
/*******************************************************************************
*
* File rwrat.c
*
* Copyright (C) 2012-2014, 2018, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Rational function reweighting factor.
*
*   qflt rwrat(int irp,int n,int *np,int *isp,qflt *sqn,int **status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqn and returns -ln(r) (see the notes).
*
* The quadruple-precision type qflt is defined in su3.h. See doc/qsum.pdf
* for further explanations.
*
* The computation of the reweighting factor needed to correct for the inexact
* pseudo-fermion action used for the charm and the strange quark is discussed
* in the notes "Charm and strange quark in openQCD simulations" (doc/rhmc.pdf).
*
* As explained there, an unbiased stochastic estimate r of the reweighting
* factor is obtained by choosing a pseudo-fermion field eta randomly with
* distribution proportional to exp{-(eta,eta)} and by calculating
*
*  r=exp{-(eta,[(1+Z)^(-1/2)-1]*eta)},
*
*  Z=Dwhat^dag*Dwhat*R^2-1,
*
* where Dwhat denotes the even-odd preconditioned O(a)-improved Wilson-Dirac
* operator and R the Zolotarev rational function of Dwhat^dag*Dwhat employed
* in the simulations for the heavy quark.
*
* The computation requires R to be applied a number of times to a given
* spinor field. For this calculation, R is divided into n parts according
* to
*
*  R=A*{1+R_0+R_1+..+R_{n-1}},
*
*  R_k=Sum{rmu[j]/(Dwhat^dag*Dwhat+mu[j]^2),j=l[k],..,l[k]+np[k]-1}
*
*  l[k]=Sum{np[j],j=0,..,k-1},
*
* mu[j] and rmu[j] being the poles and associated residues of R. The constant
* A is such that Z is of order delta, the approximation error of the Zolotarev
* rational function (see ratfcts/ratfcts.c).
*
* The arguments of the program rwrat() are:
*
*  irp       Index of the Zolotarev rational function R in the parameter
*            data base.
*
*  n         Number of parts R_k of R.
*
*  np        Array of the numbers np[k] of poles of the parts R_k. The poles
*            of R are ordered such that larger values come first. R_0 includes
*            the first np[0] poles, R_1 the next np[1] poles and so on.
*
*  isp[k]    Index of the solver to be used for the computation of the action
*            of R_k on a given spinor field (k=0,..,n-1; the supported solvers
*            are MSCG, SAP_GCR and DFL_SAP_GCR).
*
*  sqn       Square norm of the generated random pseudo-fermion field.
*
*  status[k] Array of the average values of the status numbers returned by
*            the solver with index isp[k] (k=0,..,n-1). The arrays must
*            have the standard size (see utils/futils.c). In the case of
*            the GCR solvers, where the Dirac equation is solved twice,
*            status[k] reports both sets of status numbers (unused array
*            elements are set to zero).
*
* It is taken for granted that the solver parameters have been set by
* set_solver_parms() [flags/solver-parms.c] and that the deflation subspace
* has been properly set up if the DFL_SAP_GCR solver is used. The bare quark
* mass is the one last set by set_sw_parms() [flags/lat_parms.c] and the
* calling program must ensure that the gauge field is phase-set.
*
* The computation of -ln(r) involves a series expansion of (1+Z)^(-1/2),
* which is stopped when the remainder of the series is estimated to be
* less than PRECISION_LIMIT (a macro defined below). The true accuracy
* of -ln(r) however also depends on the chosen solver residues.
*
* In addition to the workspaces required by the chosen solver programs,
* rwrat() requires a workspace of several double-precision spinor fields
* when applying the operator R_k:
*
*          solver           no of fields
*      -----------------------------------
*        MSCG                 3+np[k]
*        SAP_GCR              5
*        DFL_SAP_GCR          5
*      -----------------------------------
*
* The numbers of fields to be allocated must be greater or equal to those
* required for any k=0,..,n-1. For the number of fields required by the
* solvers, see forces/tmcgm.c, sap/sap_gcr.c and dfl/dfl_sap_gcr.c.
*
* Some debugging information is printed to stdout if the macro RWRAT_DBG is
* defined.
*
*******************************************************************************/

#define RWRAT_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "sflds.h"
#include "sw_term.h"
#include "dirac.h"
#include "linalg.h"
#include "sap.h"
#include "dfl.h"
#include "ratfcts.h"
#include "forces.h"
#include "update.h"
#include "global.h"

#define PRECISION_LIMIT 1.0e-8

static int nps=0,ns=0,*nsps;
static int ifail0[2],ifail1[2],*stat0,*stat1=NULL;
static double *rs;
static double cfs[4]={-0.5,0.375,-0.3125,0.2734375};


static void init_stat(int n,int **status)
{
   int k;

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

   for (k=0;k<n;k++)
      reset_std_status(status[k]);
}


static void avg_stat(int n,int nz,int *np,int *isp,int **status)
{
   int k,na;
   solver_parms_t sp;

   for (k=0;k<n;k++)
   {
      sp=solver_parms(isp[k]);

      if (sp.solver==MSCG)
         na=2*nz;
      else
         na=2*nz*np[k];

      avg_std_status(na,status[k]);
   }
}


static void set_nsps(int n,int *np,int *isp)
{
   int k;

   if (n>ns)
   {
      if (ns>0)
         free(nsps);

      nsps=malloc(2*n*sizeof(*nsps));
      error(nsps==NULL,1,"set_nsps [rwrat.c]",
            "Unable to allocate auxiliary array");
      ns=n;
   }

   for (k=0;k<n;k++)
   {
      nsps[k]=np[k];
      nsps[n+k]=isp[k];
   }
}


static qflt set_eta(spinor_dble *eta)
{
   random_sd(VOLUME_TRD/2,2,eta,1.0);
   set_sd2zero(VOLUME_TRD/2,2,eta+(VOLUME/2));
   bnd_sd2zero(EVEN_PTS,eta);

   return norm_square_dble(VOLUME_TRD/2,3,eta);
}


static void set_res(int np,double res)
{
   int k;

   if (np>nps)
   {
      if (nps>0)
         free(rs);

      rs=malloc(np*sizeof(*rs));
      error(rs==NULL,1,"set_res [rwrat.c]",
            "Unable to allocate auxiliary array");
      nps=np;
   }

   for (k=0;k<np;k++)
      rs[k]=res;
}


static void apply_Rk(int np,int isp,double *mu,double *rmu,
                     spinor_dble *eta,spinor_dble *psi,int *status)
{
   int k;
   spinor_dble **rsd;
   solver_parms_t sp;
   sap_parms_t sap;

   sp=solver_parms(isp);

   if (sp.solver==MSCG)
   {
      rsd=reserve_wsd(np);

      set_res(np,sp.res);
      tmcgm(sp.nmx,sp.istop,rs,np,mu,eta,rsd,ifail0,stat0);
      acc_std_status("tmcgm",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("tmcgm",ifail0,stat0);
         error_root(1,1,"apply_Rk [rwrat.c]","MSCG solver failed "
                    "(isp=%d)",isp);
      }

      for (k=0;k<np;k++)
         mulr_spinor_add_dble(VOLUME_TRD/2,2,psi,rsd[k],rmu[k]);

      release_wsd();
   }
   else if (sp.solver==SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      rsd=reserve_wsd(2);
      mulg5_dble(VOLUME_TRD/2,2,eta);
      set_sd2zero(VOLUME_TRD/2,2,eta+(VOLUME/2));

      for (k=0;k<np;k++)
      {
         sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu[k],eta,rsd[0],
                 ifail0,stat0);
         mulg5_dble(VOLUME_TRD/2,2,rsd[0]);
         set_sd2zero(VOLUME_TRD/2,2,rsd[0]+(VOLUME/2));
         sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,-mu[k],rsd[0],rsd[1],
                 ifail1,stat1);
         acc_std_status("sap_gcr",ifail0,stat0,0,status);
         acc_std_status("sap_gcr",ifail1,stat1,1,status);

         if ((ifail0[0]<0)||(ifail1[0]<0))
         {
            print_status("sap_gcr",ifail0,stat0);
            print_status("sap_gcr",ifail1,stat1);
            error_root(1,1,"apply_Rk [rwrat.c]","SAP_GCR solver failed "
                       "(isp=%d)",isp);
         }

         mulr_spinor_add_dble(VOLUME_TRD/2,2,psi,rsd[1],rmu[k]);
      }

      mulg5_dble(VOLUME_TRD/2,2,eta);
      release_wsd();
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      rsd=reserve_wsd(2);
      mulg5_dble(VOLUME_TRD/2,2,eta);
      set_sd2zero(VOLUME_TRD/2,2,eta+(VOLUME/2));

      for (k=0;k<np;k++)
      {
         dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu[k],eta,rsd[0],
                      ifail0,stat0);
         mulg5_dble(VOLUME_TRD/2,2,rsd[0]);
         set_sd2zero(VOLUME_TRD/2,2,rsd[0]+(VOLUME/2));
         dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,-mu[k],rsd[0],rsd[1],
                      ifail1,stat1);
         acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);
         acc_std_status("dfl_sap_gcr2",ifail1,stat1,1,status);

         if ((ifail0[0]<-2)||(ifail0[1]<0)||(ifail1[0]<-2)||(ifail1[1]<0))
         {
            print_status("dfl_sap_gcr2",ifail0,stat0);
            print_status("dfl_sap_gcr2",ifail1,stat1);
            error_root(1,1,"apply_Rk [rwrat.c]","DFL_SAP_GCR solver failed "
                       "(isp=%d)",isp);
         }

         mulr_spinor_add_dble(VOLUME_TRD/2,2,psi,rsd[1],rmu[k]);
      }

      mulg5_dble(VOLUME_TRD/2,2,eta);
      release_wsd();
   }
   else
      error_root(1,1,"apply_Rk [rwrat.c]","Unknown or unsupported solver");
}


static void apply_QR(int n,int *np,int *isp,ratfct_t *rf,
                     spinor_dble *eta,spinor_dble *psi,int **status)
{
   int k;
   double *mu,*rmu;
   spinor_dble **wsd;

   mu=(*rf).mu;
   rmu=(*rf).rmu;
   assign_sd2sd(VOLUME_TRD/2,2,eta,psi);

   for (k=0;k<n;k++)
   {
      apply_Rk(np[k],isp[k],mu,rmu,eta,psi,status[k]);
      mu+=np[k];
      rmu+=np[k];
   }

   scale_dble(VOLUME_TRD/2,2,(*rf).A,psi);

   wsd=reserve_wsd(1);
   sw_term(ODD_PTS);
   Dwhat_dble(0.0,psi,wsd[0]);
   mulg5_dble(VOLUME_TRD/2,2,wsd[0]);
   assign_sd2sd(VOLUME_TRD/2,2,wsd[0],psi);
   release_wsd();
}


static void apply_Z(int n,int *np,int *isp,ratfct_t *rf,
                    spinor_dble *eta,spinor_dble *psi,int **status)
{
   spinor_dble **wsd;

   wsd=reserve_wsd(1);

   apply_QR(n,np,isp,rf,eta,wsd[0],status);
   apply_QR(n,np,isp,rf,wsd[0],psi,status);
   mulr_spinor_add_dble(VOLUME_TRD/2,2,psi,eta,-1.0);

   release_wsd();
}


qflt rwrat(int irp,int n,int *np,int *isp,qflt *sqn,int **status)
{
   int k,l,ie,eo,irat[3],iprms[2];
   double delta,tol;
   qflt lnr,r0,r1;
   spinor_dble **wsd;
   ratfct_t rf;
   tm_parms_t tm;
   rat_parms_t rp;

#ifdef RWRAT_DBG
   message("[rwrat]: Start computation, irp = %d, n = %d\n",irp,n);
#endif

   rp=rat_parms(irp);
   error_root((rp.degree==0)||(n<1),1,"rwrat [rwrat.c]",
              "Unknown rational function or improper number n of terms");

   ie=0;
   l=0;

   for (k=0;k<n;k++)
   {
      ie|=(np[k]<1);
      l+=np[k];
   }

   error_root((ie!=0)||(l!=rp.degree),1,"rwrat [rwrat.c]",
              "Incorrect choice of the numbers np[k] of poles");

   if (NPROC>1)
   {
      iprms[0]=irp;
      iprms[1]=n;

      MPI_Bcast(iprms,2,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=irp)||(iprms[1]!=n),1,"rwrat [rwrat.c]",
            "Parameters are not global");

      set_nsps(n,np,isp);

      MPI_Bcast(nsps,2*n,MPI_INT,0,MPI_COMM_WORLD);

      for (k=0;k<n;k++)
      {
         ie|=(nsps[k]!=np[k]);
         ie|=(nsps[n+k]!=isp[k]);
      }

      error(ie!=0,2,"rwrat [rwrat.c]",
            "Parameters are not global");
   }

   tm=tm_parms();
   eo=tm.eoflg;
   if (eo!=1)
      set_tm_parms(1);

   irat[0]=irp;
   irat[1]=0;
   irat[2]=rp.degree-1;
   rf=ratfct(irat);
   delta=rf.delta;
   init_stat(n,status);

   wsd=reserve_wsd(2);
   (*sqn)=set_eta(wsd[0]);

   k=1;
   apply_Z(n,np,isp,&rf,wsd[0],wsd[1],status);
   r0=spinor_prod_re_dble(VOLUME_TRD/2,3,wsd[0],wsd[1]);
   r1=norm_square_dble(VOLUME_TRD/2,3,wsd[1]);
   tol=r1.q[0]*delta;

#ifdef RWRAT_DBG
   message("[rwrat]: delta = %.1e, precision limit = %.1e\n",
           delta,PRECISION_LIMIT);
   message("[rwrat]: sqn = %.4e, <Z> = %.4e, <Z^2> = %.4e",
           (*sqn).q[0],r0.q[0],r1.q[0]);
#endif

   scl_qflt(cfs[0],r0.q);
   scl_qflt(cfs[1],r1.q);
   add_qflt(r0.q,r1.q,lnr.q);

   if (tol>PRECISION_LIMIT)
   {
      k=2;
      apply_Z(n,np,isp,&rf,wsd[1],wsd[0],status);
      r0=spinor_prod_re_dble(VOLUME_TRD/2,3,wsd[1],wsd[0]);
      r1=norm_square_dble(VOLUME_TRD/2,3,wsd[0]);
      tol=r1.q[0]*delta;

#ifdef RWRAT_DBG
      message(", <Z^3> = %.4e, <Z^4> = %.4e",r0.q[0],r1.q[0]);
#endif

      error_root(tol>PRECISION_LIMIT,1,"rwrat [rwrat.c]",
                 "Unable to reach the required precision");

      scl_qflt(cfs[2],r0.q);
      scl_qflt(cfs[3],r1.q);
      add_qflt(r0.q,lnr.q,lnr.q);
      add_qflt(r1.q,lnr.q,lnr.q);
   }

   avg_stat(n,k,np,isp,status);
   if (eo!=1)
      set_tm_parms(eo);
   release_wsd();

#ifdef RWRAT_DBG
   message("\n");
   message("[rwrat]: -ln(r) = %.4e\n",lnr.q[0]);
   message("[rwrat]: All done\n\n");
#endif

   return lnr;
}
