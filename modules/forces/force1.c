
/*******************************************************************************
*
* File force1.c
*
* Copyright (C) 2011-2018, 2020, 2022  Stefan Schaefer, Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Twisted mass pseudo-fermion action and force.
*
*   qflt setpf1(double mu,int ipf,int icom)
*     Generates a pseudo-fermion field phi with probability proportional
*     to exp(-Spf) and returns the action Spf (see the notes).
*
*   void rotpf1(double mu,int ipf,int isp,int icr,double c1,double c2,
*               int *status)
*     Generates a pseudo-fermion field eta with probability proportional
*     to exp(-Spf) and and replaces phi by c1*phi+c2*eta (see the notes).
*     No use is made of the chronological solver stack, but the stack is
*     updated if it is not empty using the specified solver for the Dirac
*     equation with twisted mass mu. The associated solver status numbers
*     are written to the second half of the status array (the first half
*     is set to zero).
*
*   void force1(double mu,int ipf,int isp,int icr,double c,int *status)
*     Computes the force deriving from the action Spf (see the notes).
*     The calculated force is multiplied by c and added to the molecular-
*     dynamics force field. The program makes use of the chronological
*     solver stack and updates the stack on exit.
*
*   qflt action1(double mu,int ipf,int isp,int icr,int icom,int *status)
*     Returns the action Spf (see the notes). The chronological solver
*     stack is used but not updated.
*
* The quadruple-precision type qflt is defined in su3.h. See doc/qsum.pdf
* for further explanations.
*
* The pseudo-fermion action Spf is given by
*
*   Spf=(phi,(Dw^dag*Dw+mu^2)^(-1)*phi),
*
* where Dw denotes the (improved) Wilson-Dirac operator and phi the pseudo-
* fermion field.
*
* The common parameters of the programs in this module are:
*
*   mu            Twisted mass parameter in Spf.
*
*   ipf           Index of the pseudo-fermion field phi in the
*                 structure returned by mdflds() [mdflds.c].
*
*   isp           Index of the solver parameter set describing the
*                 solver to be used for the solution of the Dirac
*                 equation.
*
*   icr           Index of the chronological solver stack to use. The
*                 fields whose history is stored in the stack are the
*                 approximate solutions of the normal Dirac equation
*                 (Dw^dag*Dw+mu^2)*psi=phi. The feature can be disabled
*                 by setting icr=0.
*
*   icom          The action returned by the programs setpf1() and
*                 action1() is summed over all MPI processes if the
*                 bit (icom&0x1) is set. Otherwise the local part of
*                 the action is returned.
*
*   status        Status values returned by the solvers used for the
*                 solution of the Dirac equation.
*
* The supported solvers are CGNE, SAP_GCR and DFL_SAP_GCR. In all cases
* the status array must be of the standard length (see utils/futils.c).
*
* The bare quark mass m0 is the one set by sw_parms() [flags/lat_parms.c]
* and it is taken for granted that the parameters of the solvers have been
* set by set_solver_parms() [flags/solver_parms.c].
*
* The programs rotpf1() and force1() attempt to propagate the solutions of
* the Dirac equation along the molecular-dynamics trajectories, using the
* field stack number icr (no fields are propagated if icr=0). This feature
* assumes the program setup_chrono() [update/chrono.c] is called before
* rotpf1() or force1() is called for the first time.
*
* Some debugging information is printed to stdout on MPI process 0 if the
* macro FORCE_DBG is defined at compilation time.
*
* The required workspaces of double-precision spinor fields are
*
*                  CGNE         SAP_GCR       DFL_SAP_GCR
*   setpf1()         2             2               2
*   rotpf1()         2             2               2
*   force1()     2+(icr>0)     2+2*(icr>0)     2+2*(icr>0)
*   action1()        2             2               2
*
* These figures do not include the workspace required by the solvers.
*
*******************************************************************************/

#define FORCE1_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "mdflds.h"
#include "sw_term.h"
#include "sflds.h"
#include "dirac.h"
#include "linalg.h"
#include "sap.h"
#include "dfl.h"
#include "update.h"
#include "forces.h"
#include "global.h"

static int ifail0[2],ifail1[2],*stat0,*stat1=NULL;


static void init_stat(int *status)
{
   if (stat1==NULL)
   {
      stat0=alloc_std_status();
      stat1=alloc_std_status();
   }

   ifail0[0]=0;
   ifail0[1]=0;
   ifail1[0]=0;
   ifail1[1]=0;
   reset_std_status(stat0);
   reset_std_status(stat1);
   reset_std_status(status);
}

#if (defined FORCE_DBG)

static char *program[3]={"action1","force1","rotpf1"};


static void solver_info(int ipgm,int icr,double mu,solver_parms_t sp)
{
   if (sp.solver==CGNE)
      message("[%s]: CGNE solver, icr = %d, istop = %d, mu = %.2e\n",
              program[ipgm],icr,sp.istop,mu);
   else if (sp.solver==SAP_GCR)
      message("[%s]: SAP_GCR solver, icr = %d, istop = %d, mu = %.2e\n",
              program[ipgm],icr,sp.istop,mu);
   else if (sp.solver==DFL_SAP_GCR)
      message("[%s]: DFL_SAP_GCR solver, icr = %d, istop = %d, mu = %.2e\n",
              program[ipgm],icr,sp.istop,mu);
}


static void check_flds0(int ipgm,double mu,solver_parms_t sp,
                        spinor_dble *phi,spinor_dble *psi)
{
   double rpsi;
   qflt qnrm;
   spinor_dble *rho,**wsd;

   wsd=reserve_wsd(1);
   rho=wsd[0];

   sw_term(NO_PTS);
   Dw_dble(mu,psi,rho);
   mulg5_dble(VOLUME_TRD,2,rho);
   mulr_spinor_add_dble(VOLUME_TRD,2,rho,phi,-1.0);

   if (sp.istop)
   {
      rpsi=unorm_dble(VOLUME_TRD,3,rho);
      rpsi/=unorm_dble(VOLUME_TRD,3,phi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME_TRD,3,rho);
      rpsi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME_TRD,3,phi);
      rpsi/=qnrm.q[0];
      rpsi=sqrt(rpsi);
   }

   message("[%s]: Residue of psi = %.1e (should be <= %.1e)\n",
           program[ipgm],rpsi,sp.res);

   release_wsd();
}


static void check_flds1(int ipgm,double mu,solver_parms_t sp,
                        spinor_dble *phi,spinor_dble *psi,spinor_dble *chi)
{
   double rchi,rpsi;
   qflt qnrm;
   spinor_dble *rho,**wsd;

   wsd=reserve_wsd(1);
   rho=wsd[0];

   sw_term(NO_PTS);
   Dw_dble(-mu,chi,rho);
   mulg5_dble(VOLUME_TRD,2,rho);
   mulr_spinor_add_dble(VOLUME_TRD,2,rho,psi,-1.0);

   if (sp.istop)
   {
      rchi=unorm_dble(VOLUME_TRD,3,rho);
      rchi/=unorm_dble(VOLUME_TRD,3,psi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME_TRD,3,rho);
      rchi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME_TRD,3,psi);
      rchi/=qnrm.q[0];
      rchi=sqrt(rchi);
   }

   Dw_dble(mu,psi,rho);
   mulg5_dble(VOLUME_TRD,2,rho);
   mulr_spinor_add_dble(VOLUME_TRD,2,rho,phi,-1.0);

   if (sp.istop)
   {
      rpsi=unorm_dble(VOLUME_TRD,3,rho);
      rpsi/=unorm_dble(VOLUME_TRD,3,phi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME_TRD,3,rho);
      rpsi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME_TRD,3,phi);
      rpsi/=qnrm.q[0];
      rpsi=sqrt(rpsi);
   }

   message("[%s]: Residue of psi = %.1e (should be <= %.1e)\n",
           program[ipgm],rpsi,sp.res);
   message("[%s]: Residue of chi = %.1e (should be <= %.1e)\n",
           program[ipgm],rchi,sp.res);

   release_wsd();
}


static void check_flds2(int ipgm,double mu,solver_parms_t sp,
                        spinor_dble *phi,spinor_dble *psi)
{
   double rpsi;
   qflt qnrm;
   spinor_dble *rho,*chi,**wsd;

   wsd=reserve_wsd(2);
   rho=wsd[0];
   chi=wsd[1];

   sw_term(NO_PTS);
   Dw_dble(-mu,psi,rho);
   mulg5_dble(VOLUME_TRD,2,rho);
   Dw_dble(mu,rho,chi);
   mulg5_dble(VOLUME_TRD,2,chi);
   mulr_spinor_add_dble(VOLUME_TRD,2,chi,phi,-1.0);

   if (sp.istop)
   {
      rpsi=unorm_dble(VOLUME_TRD,3,chi);
      rpsi/=unorm_dble(VOLUME_TRD,3,phi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME_TRD,3,chi);
      rpsi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME_TRD,3,phi);
      rpsi/=qnrm.q[0];
      rpsi=sqrt(rpsi);
   }

   message("[%s]: Residue of psi = %.1e  (should be <= %.1e)\n",
           program[ipgm],rpsi,sp.res);

   release_wsd();
}

#endif

qflt setpf1(double mu,int ipf,int icom)
{
   qflt act;
   spinor_dble *phi,*eta,*psi,**wsd;
   mdflds_t *mdfs;
   tm_parms_t tm;

   icom|=0x2;
   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);

   wsd=reserve_wsd(2);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   eta=wsd[0];
   psi=wsd[1];

   random_sd(VOLUME_TRD,2,eta,1.0);
   bnd_sd2zero(ALL_PTS,eta);
   act=norm_square_dble(VOLUME_TRD,icom,eta);
   sw_term(NO_PTS);
   Dw_dble(mu,eta,psi);
   mulg5_dble(VOLUME_TRD,2,psi);
   assign_sd2sd(VOLUME_TRD,2,psi,phi);

   release_wsd();

   return act;
}


void rotpf1(double mu,int ipf,int isp,int icr,double c1,double c2,
            int *status)
{
   spinor_dble *phi,*eta,*psi,**wsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   init_stat(status);
   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);

   wsd=reserve_wsd(2);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   eta=wsd[0];
   psi=wsd[1];

   random_sd(VOLUME_TRD,2,eta,1.0);
   bnd_sd2zero(ALL_PTS,eta);
   sw_term(NO_PTS);
   Dw_dble(mu,eta,psi);
   mulg5_dble(VOLUME_TRD,2,psi);
   combine_spinor_dble(VOLUME_TRD,2,phi,psi,c1,c2);

   if (get_chrono(icr,psi))
   {
      sp=solver_parms(isp);

#if (defined FORCE_DBG)
      solver_info(2,icr,mu,sp);
#endif

      if (sp.solver==CGNE)
      {
         assign_sd2sd(VOLUME_TRD,2,phi,eta);
         tmcg(sp.nmx,sp.istop,sp.res,mu,eta,psi,ifail0,stat0);
         acc_std_status("tmcg",ifail0,stat0,1,status);

         if (ifail0[0]<0)
         {
            print_status("tmcg",ifail0,stat0);
            error_root(1,1,"rotpf1 [force1.c]","CGNE solver failed "
                       "(mu = %.4e, parameter set no %d)",mu,isp);
         }
      }
      else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
      {
         sap=sap_parms();
         set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
         mulg5_dble(VOLUME_TRD,2,eta);

         if (sp.solver==SAP_GCR)
         {
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,eta,
                    eta,ifail0,stat0);
            acc_std_status("sap_gcr",ifail0,stat0,1,status);

            if (ifail0[0]<0)
            {
               print_status("sap_gcr",ifail0,stat0);
               error_root(1,1,"rotpf1 [force1.c]","SAP_GCR solver failed "
                          "(mu = %.4e, parameter set no %d)",mu,isp);
            }
         }
         else
         {
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,eta,
                         eta,ifail0,stat0);
            acc_std_status("dfl_sap_gcr2",ifail0,stat0,1,status);

            if ((ifail0[0]<-2)||(ifail0[1]<0))
            {
               print_status("dfl_sap_gcr2",ifail0,stat0);
               error_root(1,1,"rotpf1 [force1.c]","DFL_SAP_GCR solver failed "
                          "(mu = %.4e, parameter set no %d)",mu,isp);
            }
         }

         combine_spinor_dble(VOLUME_TRD,2,psi,eta,c1,c2);
      }
      else
         error(1,1,"rotpf1 [force1.c]","Unknown solver");

      add_chrono(icr,psi);

#if (defined FORCE_DBG)
      check_flds2(2,mu,sp,phi,psi);
#endif
   }

   release_wsd();
}


void force1(double mu,int ipf,int isp,int icr,double c,int *status)
{
   double res0,res1;
   qflt rqsm;
   spinor_dble *phi,*chi,*psi,**wsd;
   spinor_dble *rho,*eta,**rsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   init_stat(status);
   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);

   wsd=reserve_wsd(2);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   psi=wsd[0];
   chi=wsd[1];

   sw_term(NO_PTS);
   sp=solver_parms(isp);

#if (defined FORCE_DBG)
   solver_info(1,icr,mu,sp);
#endif

   if (sp.solver==CGNE)
   {
      if (get_chrono(icr,chi))
      {
         rsd=reserve_wsd(1);
         rho=rsd[0];

         Dw_dble(-mu,chi,psi);
         mulg5_dble(VOLUME_TRD,2,psi);
         Dw_dble(mu,psi,rho);
         mulg5_dble(VOLUME_TRD,2,rho);
         mulr_spinor_add_dble(VOLUME_TRD,2,rho,phi,-1.0);

         if (sp.istop)
         {
            res0=unorm_dble(VOLUME_TRD,3,phi);
            res1=unorm_dble(VOLUME_TRD,3,rho)/res0;
         }
         else
         {
            rqsm=norm_square_dble(VOLUME_TRD,3,phi);
            res0=rqsm.q[0];
            rqsm=norm_square_dble(VOLUME_TRD,3,rho);
            res1=sqrt(rqsm.q[0]/res0);
         }

         if (res1<1.0)
         {
            if (res1>sp.res)
            {
               tmcg(sp.nmx,sp.istop,sp.res/res1,mu,rho,psi,ifail0,stat0);
               mulr_spinor_add_dble(VOLUME_TRD,2,chi,psi,-1.0);
            }
         }
         else
         {
            assign_sd2sd(VOLUME_TRD,2,phi,psi);
            tmcg(sp.nmx,sp.istop,sp.res,mu,psi,chi,ifail0,stat0);
         }

         release_wsd();
      }
      else
      {
         assign_sd2sd(VOLUME_TRD,2,phi,psi);
         tmcg(sp.nmx,sp.istop,sp.res,mu,psi,chi,ifail0,stat0);
      }

      acc_std_status("tmcg",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("tmcg",ifail0,stat0);
         error_root(1,1,"force1 [force1.c]","CGNE solver failed "
                    "(mu = %.4e, parameter set no %d)",mu,isp);
      }

      Dw_dble(-mu,chi,psi);
      mulg5_dble(VOLUME_TRD,2,psi);
   }
   else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      if (get_chrono(icr,chi))
      {
         rsd=reserve_wsd(2);
         rho=rsd[0];
         eta=rsd[1];

         Dw_dble(-mu,chi,psi);
         mulg5_dble(VOLUME_TRD,2,psi);
         Dw_dble(mu,psi,rho);
         mulg5_dble(VOLUME_TRD,2,rho);
         mulr_spinor_add_dble(VOLUME_TRD,2,rho,phi,-1.0);

         if (sp.istop)
         {
            res0=unorm_dble(VOLUME_TRD,3,phi);
            res1=unorm_dble(VOLUME_TRD,3,rho)/res0;
         }
         else
         {
            rqsm=norm_square_dble(VOLUME_TRD,3,phi);
            res0=rqsm.q[0];
            rqsm=norm_square_dble(VOLUME_TRD,3,rho);
            res1=sqrt(rqsm.q[0]/res0);
         }

         if (res1<1.0)
         {
            if (res1>sp.res)
            {
               mulg5_dble(VOLUME_TRD,2,rho);

               if (sp.solver==SAP_GCR)
                  sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res/res1,mu,rho,
                          eta,ifail0,stat0);
               else
                  dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res/res1,mu,rho,
                               eta,ifail0,stat0);

               mulr_spinor_add_dble(VOLUME_TRD,2,psi,eta,-1.0);

               if (sp.istop)
               {
                  res0=unorm_dble(VOLUME_TRD,3,psi);
                  res1=unorm_dble(VOLUME_TRD,3,eta)/res0;
               }
               else
               {
                  rqsm=norm_square_dble(VOLUME_TRD,3,psi);
                  res0=rqsm.q[0];
                  rqsm=norm_square_dble(VOLUME_TRD,3,eta);
                  res1=sqrt(rqsm.q[0]/res0);
               }

               if (res1<1.0)
               {
                  if (res1>sp.res)
                  {
                     mulg5_dble(VOLUME_TRD,2,eta);

                     if (sp.solver==SAP_GCR)
                        sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res/res1,-mu,
                                eta,rho,ifail1,stat1);
                     else
                        dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res/res1,-mu,
                                     eta,rho,ifail1,stat1);

                     mulr_spinor_add_dble(VOLUME_TRD,2,chi,rho,-1.0);
                  }
               }
               else
               {
                  mulg5_dble(VOLUME_TRD,2,psi);

                  if (sp.solver==SAP_GCR)
                     sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,psi,
                             chi,ifail1,stat1);
                  else
                     dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,psi,
                                  chi,ifail1,stat1);

                  mulg5_dble(VOLUME_TRD,2,psi);
               }
            }
         }
         else
         {
            assign_sd2sd(VOLUME_TRD,2,phi,chi);
            mulg5_dble(VOLUME_TRD,2,chi);

            if (sp.solver==SAP_GCR)
               sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,
                       ifail0,stat0);
            else
               dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,
                            ifail0,stat0);

            mulg5_dble(VOLUME_TRD,2,psi);

            if (sp.solver==SAP_GCR)
               sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,psi,chi,
                       ifail1,stat1);
            else
               dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,psi,chi,
                            ifail1,stat1);

            mulg5_dble(VOLUME_TRD,2,psi);
         }

         release_wsd();
      }
      else
      {
         assign_sd2sd(VOLUME_TRD,2,phi,chi);
         mulg5_dble(VOLUME_TRD,2,chi);

         if (sp.solver==SAP_GCR)
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,
                    ifail0,stat0);
         else
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,
                         ifail0,stat0);

         mulg5_dble(VOLUME_TRD,2,psi);

         if (sp.solver==SAP_GCR)
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,psi,chi,
                    ifail1,stat1);
         else
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,psi,chi,
                         ifail1,stat1);

         mulg5_dble(VOLUME_TRD,2,psi);
      }

      if (sp.solver==SAP_GCR)
      {
         acc_std_status("sap_gcr",ifail0,stat0,0,status);
         acc_std_status("sap_gcr",ifail1,stat1,1,status);

         if ((ifail0[0]<0)||(ifail1[0]<0))
         {
            print_status("sap_gcr",ifail0,stat0);
            print_status("sap_gcr",ifail1,stat1);
            error_root(1,1,"force1 [force1.c]","SAP_GCR solver failed "
                       "(mu = %.4e, parameter set no %d",mu,isp);
         }
      }
      else
      {
         acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);
         acc_std_status("dfl_sap_gcr2",ifail1,stat1,1,status);

         if (((ifail0[0]<-2)||(ifail0[1]<0))||((ifail1[0]<-2)||(ifail1[1]<0)))
         {
            print_status("dfl_sap_gcr2",ifail0,stat0);
            print_status("dfl_sap_gcr2",ifail1,stat1);
            error_root(1,1,"force1 [force1.c]","DFL_SAP_GCR solver failed "
                       "(mu = %.4e, parameter set no %d",mu,isp);
         }
      }
   }
   else
      error(1,1,"force1 [force1.c]","Unknown solver");

   if (icr)
      add_chrono(icr,chi);

   set_xt2zero();
   add_prod2xt(1.0,chi,psi);
   sw_frc(c);

   set_xv2zero();
   add_prod2xv(1.0,chi,psi);
   hop_frc(c);

#if (defined FORCE_DBG)
   check_flds1(1,mu,sp,phi,psi,chi);
#endif

   release_wsd();
}


qflt action1(double mu,int ipf,int isp,int icr,int icom,int *status)
{
   qflt rqsm,act;
   double res0,res1;
   spinor_dble *phi,*psi,*chi,**wsd;
   spinor_dble *rho,**rsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   init_stat(status);
   icom|=0x2;
   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);

   wsd=reserve_wsd(2);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   psi=wsd[0];
   chi=wsd[1];

   sw_term(NO_PTS);
   sp=solver_parms(isp);

#if (defined FORCE_DBG)
   solver_info(0,icr,mu,sp);
#endif

   if (sp.solver==CGNE)
   {
      if (get_chrono(icr,chi))
      {
         rsd=reserve_wsd(1);
         rho=rsd[0];

         Dw_dble(-mu,chi,rho);
         mulg5_dble(VOLUME_TRD,2,rho);
         Dw_dble(mu,rho,psi);
         mulg5_dble(VOLUME_TRD,2,psi);
         mulr_spinor_add_dble(VOLUME_TRD,2,psi,phi,-1.0);

         release_wsd();

         if (sp.istop)
         {
            res0=unorm_dble(VOLUME_TRD,3,phi);
            res1=unorm_dble(VOLUME_TRD,3,psi)/res0;
         }
         else
         {
            rqsm=norm_square_dble(VOLUME_TRD,3,phi);
            res0=rqsm.q[0];
            rqsm=norm_square_dble(VOLUME_TRD,3,psi);
            res1=sqrt(rqsm.q[0]/res0);
         }

         if (res1<1.0)
         {
            if (res1>sp.res)
            {
               tmcg(sp.nmx,sp.istop,sp.res/res1,mu,psi,psi,ifail0,stat0);
               mulr_spinor_add_dble(VOLUME_TRD,2,chi,psi,-1.0);
            }
         }
         else
         {
            assign_sd2sd(VOLUME_TRD,2,phi,psi);
            tmcg(sp.nmx,sp.istop,sp.res,mu,psi,chi,ifail0,stat0);
         }
      }
      else
      {
         assign_sd2sd(VOLUME_TRD,2,phi,psi);
         tmcg(sp.nmx,sp.istop,sp.res,mu,psi,chi,ifail0,stat0);
      }

      acc_std_status("tmcg",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("tmcg",ifail0,stat0);
         error_root(1,1,"action1 [force1.c]","CGNE solver failed "
                    "(mu = %.4e, parameter set no %d)",mu,isp);
      }

      Dw_dble(-mu,chi,psi);
      mulg5_dble(VOLUME_TRD,2,psi);
   }
   else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      if (get_chrono(icr,chi))
      {
         Dw_dble(-mu,chi,psi);
         mulg5_dble(VOLUME_TRD,2,psi);
         Dw_dble(mu,psi,chi);
         mulg5_dble(VOLUME_TRD,2,chi);
         mulr_spinor_add_dble(VOLUME_TRD,2,chi,phi,-1.0);

         if (sp.istop)
         {
            res0=unorm_dble(VOLUME_TRD,3,phi);
            res1=unorm_dble(VOLUME_TRD,3,chi)/res0;
         }
         else
         {
            rqsm=norm_square_dble(VOLUME_TRD,3,phi);
            res0=rqsm.q[0];
            rqsm=norm_square_dble(VOLUME_TRD,3,chi);
            res1=sqrt(rqsm.q[0]/res0);
         }

         if (res1<1.0)
         {
            if (res1>sp.res)
            {
               mulg5_dble(VOLUME_TRD,2,chi);

               if (sp.solver==SAP_GCR)
                  sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res/res1,mu,chi,
                          chi,ifail0,stat0);
               else
                  dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res/res1,mu,chi,
                               chi,ifail0,stat0);

               mulr_spinor_add_dble(VOLUME_TRD,2,psi,chi,-1.0);
            }
         }
         else
         {
            assign_sd2sd(VOLUME_TRD,2,phi,chi);
            mulg5_dble(VOLUME_TRD,2,chi);

            if (sp.solver==SAP_GCR)
               sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,ifail0,stat0);
            else
               dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,
                            ifail0,stat0);
         }
      }
      else
      {
         assign_sd2sd(VOLUME_TRD,2,phi,chi);
         mulg5_dble(VOLUME_TRD,2,chi);

         if (sp.solver==SAP_GCR)
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,ifail0,stat0);
         else
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,
                         ifail0,stat0);
      }

      if (sp.solver==SAP_GCR)
      {
         acc_std_status("sap_gcr",ifail0,stat0,0,status);

         if (ifail0[0]<0)
         {
            print_status("sap_gcr",ifail0,stat0);
            error_root(1,1,"action1 [force1.c]","SAP_GCR solver failed "
                       "(mu = %.4e, parameter set no %d",mu,isp);
         }
      }
      else
      {
         acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);

         if ((ifail0[0]<-2)||(ifail0[1]<0))
         {
            print_status("dfl_sap_gcr2",ifail0,stat0);
            error_root(1,1,"action1 [force1.c]","DFL_SAP_GCR solver failed "
                       "(mu = %.4e, parameter set no %d",mu,isp);
         }
      }
   }
   else
      error(1,1,"action1 [force1.c]","Unknown solver");

   act=norm_square_dble(VOLUME_TRD,icom,psi);

#if (defined FORCE_DBG)
   check_flds0(0,mu,sp,phi,psi);
#endif

   release_wsd();

   return act;
}
