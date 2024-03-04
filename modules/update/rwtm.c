
/*******************************************************************************
*
* File rwtm.c
*
* Copyright (C) 2012-2014, 2018, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Twisted-mass reweighting factors.
*
*   qflt rwtm1(double mu1,double mu2,int isp,qflt *sqn,int *status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqn and returns -ln(r1) (see the notes).
*     The twisted-mass Dirac equation is solved using the solver specified
*     by the parameter set number isp.
*
*   qflt rwtm2(double mu1,double mu2,int isp,qflt *sqn,int *status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqn and returns -ln(r2) (see the notes).
*     The twisted-mass Dirac equation is solved using the solver specified
*     by the parameter set number isp.
*
* The quadruple-precision types qflt is defined in su3.h. See doc/qsum.pdf
* for further explanations.
*
* The supported solvers are CGNE, SAP_GCR and DFL_SAP_GCR and the status
* array is assumed to have the standard length (see utils/futils.c). The
* status numbers returned by the solver for the Dirac equation with twisted
* mass mu1 and sqrt(2)*mu2 [in the case of rwtm2()] are written to the
* first and second half of the array.
*
* Twisted-mass reweighting of the quark determinant was introduced in
*
*  M. Luescher, F. Palombi: "Fluctuations and reweighting of the quark
*  determinant on large lattices", PoS LATTICE2008 (2008) 049.
*
* The values returned by the programs in this module are stochastic estimates
* of the factors in a product decomposition of the reweighting factors. See
* section 6 of the notes
*
*  M. Luescher: "Parameters of the openQCD main programs" [doc/parms.pdf].
*
* For a given random pseudo-fermion field eta with distribution proportional
* to exp{-(eta,eta)}, the factors r1 and r2 are defined by
*
*  r1=exp{-(eta,[R_1-1]*eta)},   r2=exp{-(eta,[R_2-1]*eta)},
*
*  R1=(X+mu2^2)*(X+mu1^2)^(-1),
*
*  R2=R1^2*(X+2*mu1^2)*(X+2*mu2^2)^(-1),  X=Dw^dag*Dw,
*
* where Dw denotes the massive O(a)-improved Wilson-Dirac operator. In
* both cases, the twisted masses must satisfy
*
*  0<=mu1<mu2.
*
* It is taken for granted that the solver parameters have been set by
* set_solver_parms() [flags/solver_parms.c] and that the deflation subspace
* has been properly set up if the DFL_SAP_GCR solver is used. The bare quark
* mass is the one last set by set_sw_parms() [flags/lat_parms.c]. If phase-
* periodic boundary conditions are chosen, the calling program must ensure
* that the gauge field is phase-set.
*
* The programs in this module require a workspace of 2 double-precision
* spinor fields in addition to the workspace required by the chosen solver.
*
*******************************************************************************/

#define RWTM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "sflds.h"
#include "linalg.h"
#include "sap.h"
#include "dfl.h"
#include "forces.h"
#include "update.h"
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


static void check_args(double mu1,double mu2,int isp)
{
   int iprms[1];
   double dprms[2];

   if (NPROC>1)
   {
      iprms[0]=isp;
      dprms[0]=mu1;
      dprms[1]=mu2;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,2,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((iprms[0]!=isp)||(dprms[0]!=mu1)||(dprms[1]!=mu2),1,
            "check_args [rwtm.c]","Parameters are not global");
   }

   error_root((mu1<0.0)||(mu2<=mu1),1,"check_args [rwtm.c]",
              "Twisted masses mu1,mu2 are out of range");
}


static qflt set_eta(spinor_dble *eta)
{
   random_sd(VOLUME_TRD,2,eta,1.0);
   bnd_sd2zero(ALL_PTS,eta);

   return norm_square_dble(VOLUME_TRD,3,eta);
}


qflt rwtm1(double mu1,double mu2,int isp,qflt *sqn,int *status)
{
   qflt lnr;
   spinor_dble *eta,*phi,**wsd;
   solver_parms_t sp;
   sap_parms_t sap;

   check_args(mu1,mu2,isp);
   wsd=reserve_wsd(2);
   eta=wsd[0];
   phi=wsd[1];
   (*sqn)=set_eta(eta);
   sp=solver_parms(isp);
   init_stat(status);

   if (sp.solver==CGNE)
   {
      tmcg(sp.nmx,sp.istop,sp.res,mu1,eta,phi,ifail0,stat0);
      acc_std_status("tmcg",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("tmcg",ifail0,stat0);
         error_root(1,1,"rwtm1 [rwtm.c]","CGNE solver failed "
                    "(mu = %.2e, parameter set no %d)",mu1,isp);
      }

      lnr=spinor_prod_re_dble(VOLUME_TRD,3,eta,phi);
   }
   else if (sp.solver==SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
      mulg5_dble(VOLUME_TRD,2,eta);
      sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,eta,phi,ifail0,stat0);
      acc_std_status("sap_gcr",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("sap_gcr",ifail0,stat0);
         error_root(1,1,"rwtm1 [rwtm.c]","SAP_GCR solver failed "
                    "(mu = %.2e, parameter set no %d)",mu1,isp);
      }

      lnr=norm_square_dble(VOLUME_TRD,3,phi);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
      mulg5_dble(VOLUME_TRD,2,eta);
      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,eta,phi,ifail0,stat0);
      acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);

      if ((ifail0[0]<-2)||(ifail0[1]<0))
      {
         print_status("dfl_sap_gcr2",ifail0,stat0);
         error_root(1,1,"rwtm1 [rwtm.c]","DFL_SAP_GCR solver failed "
                    "(mu = %.2e, parameter set no %d)",mu1,isp);
      }

      lnr=norm_square_dble(VOLUME_TRD,3,phi);
   }
   else
   {
      lnr.q[0]=0.0;
      lnr.q[1]=0.0;
      error_root(1,1,"rwtm1 [rwtm.c]","Unknown solver");
   }

   release_wsd();

   scl_qflt((mu2*mu2)-(mu1*mu1),lnr.q);

   return lnr;
}


qflt rwtm2(double mu1,double mu2,int isp,qflt *sqn,int *status)
{
   qflt lnr1,lnr2;
   spinor_dble *eta,*phi,**wsd;
   solver_parms_t sp;
   sap_parms_t sap;

   check_args(mu1,mu2,isp);
   wsd=reserve_wsd(2);
   eta=wsd[0];
   phi=wsd[1];
   (*sqn)=set_eta(eta);
   sp=solver_parms(isp);
   init_stat(status);

   if (sp.solver==CGNE)
   {
      tmcg(sp.nmx,sp.istop,sp.res,mu1,eta,phi,ifail0,stat0);
      acc_std_status("tmcg",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("tmcg",ifail0,stat0);
         error_root(1,1,"rwtm2 [rwtm.c]","CGNE solver failed "
                    "(mu = %.2e, parameter set no %d)",mu1,isp);
      }

      tmcg(sp.nmx,sp.istop,sp.res,sqrt(2.0)*mu2,eta,eta,ifail1,stat1);
      acc_std_status("tmcg",ifail1,stat1,1,status);

      if (ifail1[0]<0)
      {
         print_status("tmcg",ifail1,stat1);
         error_root(1,1,"rwtm2 [rwtm.c]","CGNE solver failed "
                    "(mu = %.2e, parameter set no %d)",sqrt(2.0)*mu2,isp);
      }

      if (mu1>0.0)
         lnr1=norm_square_dble(VOLUME_TRD,3,phi);
      else
      {
         lnr1.q[0]=0.0;
         lnr1.q[1]=0.0;
      }

      lnr2=spinor_prod_re_dble(VOLUME_TRD,3,eta,phi);
   }
   else if (sp.solver==SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
      mulg5_dble(VOLUME_TRD,2,eta);
      sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,eta,phi,
              ifail0,stat0);
      acc_std_status("sap_gcr",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("sap_gcr",ifail0,stat0);
         error_root(1,1,"rwtm2 [rwtm.c]","SAP_GCR solver failed "
                    "(mu = %.2e, parameter set no %d)",mu1,isp);
      }

      mulg5_dble(VOLUME_TRD,2,phi);
      sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,sqrt(2.0)*mu2,phi,eta,
              ifail1,stat1);

      if (ifail1[0]<0)
      {
         print_status("sap_gcr",ifail1,stat1);
         error_root(1,1,"rwtm2 [rwtm.c]","SAP_GCR solver failed "
                    "(mu = %.2e, parameter set no %d)",sqrt(2.0)*mu2,isp);
      }

      if (mu1>0.0)
      {
         sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,phi,phi,
                 ifail0,stat0);
         acc_std_status("sap_gcr",ifail0,stat0,0,status);

         if (ifail0[0]<0)
         {
            print_status("sap_gcr",ifail0,stat0);
            error_root(1,2,"rwtm2 [rwtm.c]","SAP_GCR solver failed "
                       "(mu = %.2e, parameter set no %d)",mu1,isp);
         }

         avg_std_status(2,status);
         lnr1=norm_square_dble(VOLUME_TRD,3,phi);
      }
      else
      {
         lnr1.q[0]=0.0;
         lnr1.q[1]=0.0;
      }

      acc_std_status("sap_gcr",ifail1,stat1,1,status);
      lnr2=norm_square_dble(VOLUME_TRD,3,eta);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
      mulg5_dble(VOLUME_TRD,2,eta);
      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,eta,phi,
                   ifail0,stat0);
      acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);

      if ((ifail0[0]<-2)||(ifail0[1]<0))
      {
         print_status("dfl_sap_gcr2",ifail0,stat0);
         error_root(1,1,"rwtm2 [rwtm.c]","DFL_SAP_GCR solver failed "
                    "(mu = %.2e, parameter set no %d)",mu1,isp);
      }

      mulg5_dble(VOLUME_TRD,2,phi);
      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,sqrt(2.0)*mu2,phi,eta,
                   ifail1,stat1);

      if ((ifail1[0]<-2)||(ifail1[1]<0))
      {
         print_status("dfl_sap_gcr2",ifail1,stat1);
         error_root(1,1,"rwtm2 [rwtm.c]","DFL_SAP_GCR solver failed "
                    "(mu = %.2e, parameter set no %d)",sqrt(2.0)*mu2,isp);
      }

      if (mu1>0.0)
      {
         dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,phi,phi,
                      ifail0,stat0);
         acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);

         if ((ifail0[0]<-2)||(ifail0[1]<0))
         {
            print_status("dfl_sap_gcr2",ifail0,stat0);
            error_root(1,2,"rwtm2 [rwtm.c]","DFL_SAP_GCR solver failed "
                       "(mu = %.2e, parameter set no %d)",mu1,isp);
         }

         avg_std_status(2,status);
         lnr1=norm_square_dble(VOLUME_TRD,3,phi);
      }
      else
      {
         lnr1.q[0]=0.0;
         lnr1.q[1]=0.0;
      }

      acc_std_status("dfl_sap_gcr2",ifail1,stat1,1,status);
      lnr2=norm_square_dble(VOLUME_TRD,3,eta);
   }
   else
   {
      lnr1.q[0]=0.0;
      lnr1.q[1]=0.0;
      lnr2.q[0]=0.0;
      lnr2.q[1]=0.0;
      error_root(1,1,"rwtm2 [rwtm.c]","Unknown solver");
   }

   release_wsd();

   mu1=mu1*mu1;
   mu2=mu2*mu2;
   scl_qflt(mu1*(mu2-mu1),lnr1.q);
   scl_qflt(2.0*mu2*mu2,lnr2.q);
   add_qflt(lnr1.q,lnr2.q,lnr1.q);
   scl_qflt((mu2-mu1)/(2.0*mu2-mu1),lnr1.q);

   return lnr1;
}
