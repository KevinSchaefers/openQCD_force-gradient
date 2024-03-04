
/*******************************************************************************
*
* File force5.c
*
* Copyright (C) 2011-2018, 2020, 2022 Stefan Schaefer, Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Hasenbusch twisted mass pseudo-fermion action and force with even-odd
* preconditioning.
*
*   qflt setpf5(double mu0,double mu1,int ipf,int isp1,int icom,
*               int *status)
*     Generates a pseudo-fermion field phi with probability proportional
*     to exp(-Spf) and returns the action Spf-(phi,phi) (see the notes).
*
*   void rotpf5(double mu0,double mu1,int ipf,int isp0,int isp1,int icr,
*               double c1,double c2,int *status)
*     Generates a pseudo-fermion field eta with probability proportional
*     to exp(-Spf) and and replaces phi by c1*phi+c2*eta (see the notes).
*     No use is made of the chronological solver stack, but the stack is
*     updated if it is not empty using the solver for twisted mass mu0.
*     The associated solver status numbers are written to the second half
*     of the status array, while the first half of the array reports the
*     numbers returned by the solver for the Dirac equation with twisted
*     mass mu1.
*
*   void force5(double mu0,int mu1,int ipf,int isp0,int icr,double c,
*               int *status)
*     Computes the force deriving from the action Spf (see the notes).
*     The calculated force is multiplied by c and added to the molecular-
*     dynamics force field. The program makes use of the chronological
*     solver stack and updates the stack on exit.
*
*   qflt action5(double mu0,double mu1,int ipf,int isp0,int icr,
*                int icom,int *status)
*     Returns the action Spf-(phi,phi) (see the notes). The chronological
*     solver stack is used but not updated.
*
* The quadruple-precision type qflt is defined in su3.h. See doc/qsum.pdf
* for further explanations.
*
* The pseudo-fermion action Spf is given by
*
*   Spf=(phi,(Dwhat^dag*Dwhat+mu1^2)(Dwhat^dag*Dwhat+mu0^2)^(-1)*phi)
*
*      =(phi,phi)+(mu1^2-mu0^2)*(phi,(Dwhat^dag*Dwhat+mu0^2)^(-1)*phi),
*
* where Dwhat denotes the even-odd preconditioned (improved) Wilson-Dirac
* operator and phi the pseudo-fermion field.
*
* The common parameters of the programs in this module are:
*
*   mu0,mu1       Twisted mass parameters in Spf.
*
*   ipf           Index of the pseudo-fermion field phi in the
*                 structure returned by mdflds() [mdflds.c].
*
*   isp0,isp1     Indices of the solver parameter sets describing the
*                 solvers to be used for the solution of the Dirac
*                 equation with twisted mass mu0 and mu1.
*
*   icr           Index of the chronological solver stack to use. The
*                 fields whose history is stored in the stack are the
*                 approximate solutions of the normal Dirac equation
*                 (Dwhat^dag*Dwhat+mu0^2)*psi=phi. The feature can be
*                 disabled by setting icr=0.
*
*   icom          The action returned by the programs setpf5() and
*                 action5() is summed over all MPI processes if the
*                 bit (icom&0x1) is set. Otherwise the local part of
*                 the action is returned.
*
*   status        Status values returned by the solvers used for the
*                 solution of the Dirac equation.
*
* The supported solvers are CGNE, SAP_GCR and DFL_SAP_GCR. In all cases
* the status array must be of the standard length (see utils/futils.c).
*
* The solver used in the case of setpf5() is for the Dirac equation with
* twisted mass mu1, while force5() and action5() use the solver for the
* equation with twisted mass mu0. In the case of rotpf5() the solver first
* used is the one for mass mu1 and the solver for mass mu0 is used only if
* icr>0. In force5() the specified solver solves the Dirac equation twice.
*
* The bare quark mass m0 is the one last set by sw_parms() [flags/lat_parms.c]
* and it is taken for granted that the parameters of the solvers have been set
* by set_solver_parms() [flags/solver_parms.c].
*
* The programs rotpf5() and force5() attempt to propagate the solutions of
* the Dirac equation along the molecular-dynamics trajectories, using the
* field stack number icr (no fields are propagated if icr=0). This feature
* assumes the program setup_chrono() [update/chrono.c] is called before
* rotpf5() or force5() is called for the first time.
*
* Some debugging information is printed to stdout on MPI process 0 if the
* macro FORCE_DBG is defined at compilation time.
*
* The required workspaces of double-precision spinor fields are
*
*                  CGNE         SAP_GCR       DFL_SAP_GCR
*   setpf5()         1             1               1
*   rotpf5()         2             2               2
*   force5()     2+(icr>0)     3+(icr>0)       3+(icr>0)
*   action5()        2             2               2
*
* These figures do not include the workspace required by the solvers.
*
*******************************************************************************/

#define FORCE5_C

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

static int ifail0[2],*stat0=NULL;


static void init_stat(int *status)
{
   if (stat0==NULL)
      stat0=alloc_std_status();

   ifail0[0]=0;
   ifail0[1]=0;
   reset_std_status(stat0);
   reset_std_status(status);
}

#if (defined FORCE_DBG)

static char *program[2]={"rotpf5","setpf5"};


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

   sw_term(ODD_PTS);
   Dwhat_dble(mu,psi,rho);
   mulg5_dble(VOLUME_TRD/2,2,rho);
   mulr_spinor_add_dble(VOLUME_TRD/2,2,rho,phi,-1.0);

   if (sp.istop)
   {
      rpsi=unorm_dble(VOLUME_TRD/2,3,rho);
      rpsi/=unorm_dble(VOLUME_TRD/2,3,phi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME_TRD/2,3,rho);
      rpsi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME_TRD/2,3,phi);
      rpsi/=qnrm.q[0];
      rpsi=sqrt(rpsi);
   }

   message("[%s]: Residue of psi = %.1e (should be <= %.1e)\n",
           program[ipgm],rpsi,sp.res);

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

   sw_term(ODD_PTS);
   Dwhat_dble(-mu,psi,rho);
   mulg5_dble(VOLUME_TRD/2,2,rho);
   Dwhat_dble(mu,rho,chi);
   mulg5_dble(VOLUME_TRD/2,2,chi);
   mulr_spinor_add_dble(VOLUME_TRD/2,2,chi,phi,-1.0);

   if (sp.istop)
   {
      rpsi=unorm_dble(VOLUME_TRD/2,3,chi);
      rpsi/=unorm_dble(VOLUME_TRD/2,3,phi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME_TRD/2,3,chi);
      rpsi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME_TRD/2,3,phi);
      rpsi/=qnrm.q[0];
      rpsi=sqrt(rpsi);
   }

   message("[%s]: Residue of psi = %.1e (should be <= %.1e)\n",
           program[ipgm],rpsi,sp.res);

   release_wsd();
}

#endif

qflt setpf5(double mu0,double mu1,int ipf,int isp1,int icom,int *status)
{
   qflt act;
   complex_dble z;
   spinor_dble *phi,*psi,**wsd;
   spinor_dble *chi,**rsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   init_stat(status);
   icom|=0x2;
   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);

   wsd=reserve_wsd(1);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   psi=wsd[0];

   random_sd(VOLUME_TRD/2,2,psi,1.0);
   bnd_sd2zero(EVEN_PTS,psi);
   assign_sd2sd(VOLUME_TRD/2,2,psi,phi);
   sp=solver_parms(isp1);

#if (defined FORCE_DBG)
   solver_info(1,0,mu1,sp);
#endif

   if (sp.solver==CGNE)
   {
      tmcgeo(sp.nmx,sp.istop,sp.res,mu1,psi,psi,ifail0,stat0);
      acc_std_status("tmcgeo",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("tmcgeo",ifail0,stat0);
         error_root(1,1,"setpf5 [force5.c]","CGNE solver failed "
                    "(mu = %.4e, parameter set no %d)",mu1,isp1);
      }

      rsd=reserve_wsd(1);
      chi=rsd[0];
      assign_sd2sd(VOLUME_TRD/2,2,psi,chi);
      Dwhat_dble(-mu1,chi,psi);
      mulg5_dble(VOLUME_TRD/2,2,psi);
      release_wsd();
   }
   else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      mulg5_dble(VOLUME_TRD/2,2,psi);
      set_sd2zero(VOLUME_TRD/2,2,psi+(VOLUME/2));

      if (sp.solver==SAP_GCR)
      {
         sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,psi,psi,
                 ifail0,stat0);
         acc_std_status("sap_gcr",ifail0,stat0,0,status);

         if (ifail0[0]<0)
         {
            print_status("sap_gcr",ifail0,stat0);
            error_root(1,1,"setpf5 [force5.c]","SAP_GCR solver failed "
                       "(mu = %.4e, parameter set no %d)",mu1,isp1);
         }
      }
      else
      {
         dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,psi,psi,
                      ifail0,stat0);
         acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);

         if ((ifail0[0]<-2)||(ifail0[1]<0))
         {
            print_status("dfl_sap_gcr2",ifail0,stat0);
            error_root(1,1,"setpf5 [force5.c]","DFL_SAP_GCR solver failed "
                       "(mu = %.4e, parameter set no %d)",mu1,isp1);
         }
      }
   }
   else
      error(1,1,"setpf5 [force5.c]","Unknown solver");

#if (defined FORCE_DBG)
   check_flds0(1,mu1,sp,phi,psi);
#endif

   z.re=0.0;
   z.im=mu0-mu1;
   mulc_spinor_add_dble(VOLUME_TRD/2,2,phi,psi,z);
   act=norm_square_dble(VOLUME_TRD/2,icom,psi);
   scl_qflt((mu1*mu1)-(mu0*mu0),act.q);

   release_wsd();

   return act;
}


void rotpf5(double mu0,double mu1,int ipf,int isp0,int isp1,int icr,
            double c1,double c2,int *status)
{
   complex_dble z;
   spinor_dble *phi,*psi,*eta,**wsd;
   spinor_dble *rho,**rsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   init_stat(status);
   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);

   wsd=reserve_wsd(2);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   psi=wsd[0];
   eta=wsd[1];

   random_sd(VOLUME_TRD/2,2,eta,1.0);
   bnd_sd2zero(EVEN_PTS,eta);
   combine_spinor_dble(VOLUME_TRD/2,2,phi,eta,c1,c2);
   sp=solver_parms(isp1);

#if (defined FORCE_DBG)
   solver_info(0,icr,mu1,sp);
#endif

   if (sp.solver==CGNE)
   {
      tmcgeo(sp.nmx,sp.istop,sp.res,mu1,eta,psi,ifail0,stat0);
      acc_std_status("tmcgeo",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("tmcgeo",ifail0,stat0);
         error_root(1,1,"rotpf5 [force5.c]","CGNE solver failed "
                    "(mu = %.4e, parameter set no %d)",mu1,isp1);
      }

      rsd=reserve_wsd(1);
      rho=rsd[0];
      assign_sd2sd(VOLUME_TRD/2,2,psi,rho);
      Dwhat_dble(-mu1,rho,psi);
      mulg5_dble(VOLUME_TRD/2,2,psi);
      release_wsd();
   }
   else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      mulg5_dble(VOLUME_TRD/2,2,eta);
      set_sd2zero(VOLUME_TRD/2,2,eta+(VOLUME/2));

      if (sp.solver==SAP_GCR)
      {
         sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,eta,psi,
                 ifail0,stat0);
         acc_std_status("sap_gcr",ifail0,stat0,0,status);

         if (ifail0[0]<0)
         {
            print_status("sap_gcr",ifail0,stat0);
            error_root(1,1,"rotpf5 [force5.c]","SAP_GCR solver failed "
                       "(mu = %.4e, parameter set no %d)",mu1,isp1);
         }
      }
      else
      {
         dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,eta,psi,
                      ifail0,stat0);
         acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);

         if ((ifail0[0]<-2)||(ifail0[1]<0))
         {
            print_status("dfl_sap_gcr2",ifail0,stat0);
            error_root(1,1,"rotpf5 [force5.c]","DFL_SAP_GCR solver failed "
                       "(mu = %.4e, parameter set no %d)",mu1,isp1);
         }
      }
   }
   else
      error(1,1,"rotpf5 [force5.c]","Unknown solver");

#if (defined FORCE_DBG)
   if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
      mulg5_dble(VOLUME_TRD/2,2,eta);
   check_flds0(0,mu1,sp,eta,psi);
#endif

   z.re=0.0;
   z.im=c2*(mu0-mu1);
   mulc_spinor_add_dble(VOLUME_TRD/2,2,phi,psi,z);

   if (get_chrono(icr,eta))
   {
      sp=solver_parms(isp0);

#if (defined FORCE_DBG)
      solver_info(0,icr,mu0,sp);
#endif

      if (sp.solver==CGNE)
      {
         assign_sd2sd(VOLUME_TRD/2,2,phi,psi);
         tmcgeo(sp.nmx,sp.istop,sp.res,mu0,psi,psi,ifail0,stat0);
         acc_std_status("tmcgeo",ifail0,stat0,1,status);

         if (ifail0[0]<0)
         {
            print_status("tmcgeo",ifail0,stat0);
            error_root(1,1,"rotpf5 [force5.c]","CGNE solver failed "
                       "(mu = %.4e, parameter set no %d)",mu0,isp0);
         }
      }
      else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
      {
         sap=sap_parms();
         set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
         mulg5_dble(VOLUME_TRD/2,2,psi);
         set_sd2zero(VOLUME_TRD/2,2,psi+(VOLUME/2));

         if (sp.solver==SAP_GCR)
         {
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,-mu0,psi,psi,
                    ifail0,stat0);
            acc_std_status("sap_gcr",ifail0,stat0,1,status);

            if (ifail0[0]<0)
            {
               print_status("sap_gcr",ifail0,stat0);
               error_root(1,1,"rotpf5 [force5.c]","SAP_GCR solver failed "
                          "(mu = %.4e, parameter set no %d)",mu0,isp0);
            }
         }
         else
         {
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,-mu0,psi,psi,
                         ifail0,stat0);
            acc_std_status("dfl_sap_gcr2",ifail0,stat0,1,status);

            if ((ifail0[0]<-2)||(ifail0[1]<0))
            {
               print_status("dfl_sap_gcr2",ifail0,stat0);
               error_root(1,1,"rotpf5 [force5.c]","DFL_SAP_GCR solver failed "
                          "(mu = %.4e, parameter set no %d)",mu0,isp0);
            }
         }

         combine_spinor_dble(VOLUME_TRD/2,2,psi,eta,c2,c1);
      }
      else
         error(1,1,"rotpf5 [force5.c]","Unknown solver");

      add_chrono(icr,psi);

#if (defined FORCE_DBG)
      check_flds2(0,mu0,sp,phi,psi);
#endif
   }

   release_wsd();
}


void force5(double mu0,double mu1,int ipf,int isp,int icr,
            double c,int *status)
{
   double dmu2;

   dmu2=(mu1*mu1)-(mu0*mu0);
   force4(mu0,ipf,0,isp,icr,dmu2*c,status);
}


qflt action5(double mu0,double mu1,int ipf,int isp,int icr,
             int icom,int *status)
{
   qflt act;

   act=action4(mu0,ipf,0,isp,icr,icom,status);
   scl_qflt((mu1*mu1)-(mu0*mu0),act.q);

   return act;
}
