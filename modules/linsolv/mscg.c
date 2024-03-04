
/*******************************************************************************
*
* File mscg.c
*
* Copyright (C) 2012, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic multi-shift CG solver program for the lattice Dirac equation.
*
*   void mscg(int vol,int nmu,double *mu,
*             void (*Dop_dble)(double mu,spinor_dble *s,spinor_dble *r),
*             spinor_dble **wsd,int nmx,int istop,double *res,
*             spinor_dble *eta,spinor_dble **psi,int *status)
*     Solution of the Dirac equation (D^dag*D+mu^2)*psi=eta for a given
*     source eta and one or more values of mu using the multi-shift CG
*     algorithm. See the notes for the explanation of the parameters of
*     the program.
*
* The algorithm implemented in this module is described in the notes
* "Multi-shift conjugate gradient algorithm" (file doc/mscg.pdf).
*
* The program Dop_dble() for the Dirac operator is assumed to have the
* following properties:
*
*   void Dop_dble(double mu,spinor_dble *s,spinor_dble *r)
*     Application of an operator Op or its Hermitian conjugate Op^dag
*     to the double-precision Dirac field s and assignment of the result
*     to r (where r is different from s). The operator must be such that
*     the identity Op^dag*Op=D^dag*D+mu^2 holds. Op and Op^dag are applied
*     alternatingly, i.e. the first call of the program applies Op, the
*     next call Op^dag, then Op again and so on. In all cases, the source
*     field s is unchanged.
*
* The other parameters of the program mscg() are:
*
*   vol     Number of elements per OpenMP thread of the spinor fields
*           on which Dop_dble() acts.
*
*   nmu     Number of shifts mu.
*
*   mu      Array of the shifts mu (nmu elements).
*
*   nmx     Maximal number of CG iterations that may be applied.
*
*   istop   Stopping criterion (0: L_2 norm based, 1: uniform norm based).
*
*   res     Array of the desired maximal relative residues of the
*           calculated solutions (nmu elements).
*
*   wsd     Array of at least 3+nmu (5 if nmu=1) double-precision spinor
*           fields (used as work space).
*
*   eta     Source field (unchanged on exit).
*
*   psi     Array of the calculated approximate solutions of the Dirac
*           equations (D^dag*D+mu^2)*psi=eta (nmu elements).
*
*   status  If the program terminates normally, status reports the total
*           number of CG iterations that were required for the solution of
*           the Dirac equations. Otherwise status is set to -1.
*
* The fields eta and psi as well as the fields in the workspace are assumed
* to be such that the operators Dop_dble() can act on them.
*
* Some debugging output is printed to stdout on process 0 if the macro
* MSCG_DBG is defined.
*
* The program mscg() is assumed to be called by the OpenMP master thread
* and on all MPI processes simultaneously.
*
* If SSE (AVX) instructions are used, the Dirac spinors must be aligned to
* a 16 (32) byte boundary.
*
*******************************************************************************/

#define MSCG_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "utils.h"
#include "sflds.h"
#include "linalg.h"
#include "linsolv.h"
#include "global.h"

typedef struct
{
   int stop;
   double s,tol;
   double ah,bh,gh,rh;
   spinor_dble *xh,*ph;
} cgsh_t;

typedef struct
{
   int k,stop;
   double mu,tol;
   double a,b;
   double rn0,rn,rnsq;
   spinor_dble *x,*r,*p,*ap,*w;
} cgs_t;

static int ns=0;
static double *dprms;
static cgs_t cgs;
static cgsh_t *cgsh;


static int alloc_cgs(int nmu,double *mu,double *res,spinor_dble **wsd,
                     spinor_dble **psi)
{
   int k,l,k0;

   if (nmu>ns)
   {
      if (ns>0)
         free(dprms);
      if (ns>1)
         free(cgsh);

      dprms=malloc(2*nmu*sizeof(*dprms));
      if (dprms==NULL)
         return 1;

      if (nmu>1)
      {
         cgsh=malloc((nmu-1)*sizeof(*cgsh));
         if (cgsh==NULL)
            return 1;
      }

      ns=nmu;
   }

   k0=0;

   for (k=1;k<nmu;k++)
   {
      if (fabs(mu[k])<fabs(mu[k0]))
         k0=k;
   }

   cgs.k=k0;
   cgs.stop=0;
   cgs.mu=mu[k0];
   cgs.a=1.0;
   cgs.b=0.0;
   cgs.tol=res[k0];

   cgs.x=psi[k0];
   cgs.r=wsd[0];
   cgs.p=wsd[1];
   cgs.ap=wsd[2];
   cgs.w=wsd[3];

   l=0;

   for (k=0;k<nmu;k++)
   {
      if (k!=k0)
      {
         cgsh[l].stop=0;
         cgsh[l].s=mu[k]*mu[k]-mu[k0]*mu[k0];
         cgsh[l].tol=res[k];
         cgsh[l].ah=1.0;
         cgsh[l].bh=0.0;
         cgsh[l].gh=1.0;
         cgsh[l].rh=1.0;

         cgsh[l].xh=psi[k];
         cgsh[l].ph=wsd[4+l];

         l+=1;
      }
   }

   return 0;
}


static void cg_init(int vol,int nmu,int istop,spinor_dble *eta)
{
   int k;
   qflt qnrm;

   set_sd2zero(vol,2,cgs.x);
   assign_sd2sd(vol,2,eta,cgs.r);
   assign_sd2sd(vol,2,eta,cgs.p);

   qnrm=norm_square_dble(vol,3,eta);
   cgs.rnsq=qnrm.q[0];

   if (istop)
      cgs.rn=unorm_dble(vol,3,eta);
   else
      cgs.rn=sqrt(cgs.rnsq);

   cgs.rn0=cgs.rn;
   cgs.tol*=cgs.rn0;

   for (k=0;k<(nmu-1);k++)
   {
      set_sd2zero(vol,2,cgsh[k].xh);
      assign_sd2sd(vol,2,eta,cgsh[k].ph);
      cgsh[k].tol*=cgs.rn0;
   }
}


static void cg_step1(int vol,int nmu,
                     void (*Dop_dble)(double mu,spinor_dble *s,spinor_dble *r))
{
   int k;
   double om;
   qflt qnrm;

   Dop_dble(cgs.mu,cgs.p,cgs.w);
   Dop_dble(cgs.mu,cgs.w,cgs.ap);

   om=cgs.b/cgs.a;
   qnrm=norm_square_dble(vol,3,cgs.w);
   cgs.a=cgs.rnsq/qnrm.q[0];
   om*=cgs.a;

   for (k=0;k<(nmu-1);k++)
   {
      if (cgsh[k].stop==0)
      {
         cgsh[k].rh=1.0/(1.0+cgsh[k].s*cgs.a+(1.0-cgsh[k].rh)*om);
         cgsh[k].ah=cgsh[k].rh*cgs.a;
      }
   }
}


static void cg_step2(int vol,int nmu)
{
   int k;

   mulr_spinor_add_dble(vol,2,cgs.x,cgs.p,cgs.a);
   mulr_spinor_add_dble(vol,2,cgs.r,cgs.ap,-cgs.a);

   for (k=0;k<(nmu-1);k++)
   {
      if (cgsh[k].stop==0)
         mulr_spinor_add_dble(vol,2,cgsh[k].xh,cgsh[k].ph,cgsh[k].ah);
   }
}


static void cg_step3(int vol,int nmu,int istop)
{
   int k;
   double rnsq,rh;
   qflt qnrm;

   qnrm=norm_square_dble(vol,3,cgs.r);
   rnsq=qnrm.q[0];
   cgs.b=rnsq/cgs.rnsq;
   cgs.rnsq=rnsq;

   if (istop)
      cgs.rn=unorm_dble(vol,3,cgs.r);
   else
      cgs.rn=sqrt(rnsq);

   for (k=0;k<(nmu-1);k++)
   {
      if (cgsh[k].stop==0)
      {
         rh=cgsh[k].rh;
         cgsh[k].bh=rh*rh*cgs.b;
         cgsh[k].gh*=rh;
      }
   }
}


static void cg_step4(int vol,int nmu)
{
   int k;

   combine_spinor_dble(vol,2,cgs.p,cgs.r,cgs.b,1.0);

   for (k=0;k<(nmu-1);k++)
   {
      if (cgsh[k].stop==0)
         combine_spinor_dble(vol,2,cgsh[k].ph,cgs.r,cgsh[k].bh,cgsh[k].gh);
   }
}


static int set_stop_flag(int nmu)
{
   int nstop,k;
   double rn;

   rn=cgs.rn;
   cgs.stop|=(rn<(0.99*cgs.tol));
   nstop=cgs.stop;

   for (k=0;k<(nmu-1);k++)
   {
      cgsh[k].stop|=((cgsh[k].gh*rn)<(0.99*cgsh[k].tol));
      nstop+=cgsh[k].stop;
   }

   return nstop;
}


static int check_res(int vol,double mu,
                     void (*Dop_dble)(double mu,spinor_dble *s,spinor_dble *r),
                     spinor_dble **wsd,int nmx,int istop,double res,
                     spinor_dble *eta,spinor_dble *psi,int *ncg)
{
   double tol,rnsq,rn,a,b;
   spinor_dble *x,*r,*p,*ap,*w;
   qflt qnrm;

   tol=res*cgs.rn0;
   r=wsd[0];
   w=wsd[1];

   Dop_dble(mu,psi,w);
   Dop_dble(mu,w,r);

   mulr_spinor_add_dble(vol,2,r,eta,-1.0);
   qnrm=norm_square_dble(vol,3,r);
   rnsq=qnrm.q[0];

   if (istop)
      rn=unorm_dble(vol,3,r);
   else
      rn=sqrt(rnsq);

   if (rn<=tol)
      return 0;
   else if (ncg[0]>=nmx)
      return 1;

   x=wsd[2];
   p=wsd[3];
   ap=wsd[4];

   set_sd2zero(vol,2,x);
   assign_sd2sd(vol,2,r,p);

   while ((rn>tol)&&(ncg[0]<nmx))
   {
      Dop_dble(mu,p,w);
      Dop_dble(mu,w,ap);

      qnrm=norm_square_dble(vol,3,w);
      a=rnsq/qnrm.q[0];
      mulr_spinor_add_dble(vol,2,x,p,a);
      mulr_spinor_add_dble(vol,2,r,ap,-a);

      qnrm=norm_square_dble(vol,3,r);
      b=qnrm.q[0]/rnsq;
      rnsq=qnrm.q[0];

      if (istop)
         rn=unorm_dble(vol,3,r);
      else
         rn=sqrt(rnsq);

      combine_spinor_dble(vol,2,p,r,b,1.0);
      ncg[0]+=1;
   }

   mulr_spinor_add_dble(vol,2,psi,x,-1.0);

   if (rn<=tol)
      return 0;
   else
      return 1;
}


void mscg(int vol,int nmu,double *mu,
          void (*Dop_dble)(double mu,spinor_dble *s,spinor_dble *r),
          spinor_dble **wsd,int nmx,int istop,double *res,
          spinor_dble *eta,spinor_dble **psi,int *status)
{
   int ncg,nstop,k,ic,ie;
   int iprms[4];

   error_loc((vol<1)||(nmu<1)||(nmx<1)||(istop<0)||(istop>1),1,
             "mscg [mscg.c]","Parameters are out of range");

   if (NPROC>1)
   {
      iprms[0]=vol;
      iprms[1]=nmu;
      iprms[2]=nmx;
      iprms[3]=istop;

      MPI_Bcast(iprms,4,MPI_INT,0,MPI_COMM_WORLD);
      error((iprms[0]!=vol)||(iprms[1]!=nmu)||(iprms[2]!=nmx)||
            (iprms[3]!=istop),1,"mscg [mscg.c]","Parameters are not global");
   }

   ie=alloc_cgs(nmu,mu,res,wsd,psi);
   error_loc(ie!=0,1,"mscg [mscg.c]","Unable to allocate auxiliary arrays");

   if (NPROC>1)
   {
      for (k=0;k<nmu;k++)
      {
         dprms[k]=mu[k];
         dprms[nmu+k]=res[k];
      }

      MPI_Bcast(dprms,2*nmu,MPI_DOUBLE,0,MPI_COMM_WORLD);

      for (k=0;k<nmu;k++)
      {
         ie|=(dprms[k]!=mu[k]);
         ie|=(dprms[nmu+k]!=res[k]);
      }

      error(ie!=0,2,"mscg [mscg.c]","Parameters are not global");
   }

   for (k=0;k<nmu;k++)
      ie|=(res[k]<=DBL_EPSILON);

   error_loc(ie!=0,1,"mscg [mscg.c]","Improper choice of residues");

#ifdef MSCG_DBG
   message("\n");
   message("[mscg]: New call, nmu = %d, mu = %.2e",nmu,cgs.mu);

   for (k=0;k<nmu;k++)
   {
      if (k!=cgs.k)
         message(", %.2e",mu[k]);
   }

   message("\n");
   message("[mscg]: res = %.1e",res[cgs.k]);

   for (k=0;k<nmu;k++)
   {
      if (k!=cgs.k)
         message(", %.1e",res[k]);
   }

   message("\n");
#endif

   cg_init(vol,nmu,istop,eta);
   nstop=set_stop_flag(nmu);
   ncg=0;

#ifdef MSCG_DBG
   message("[mscg]: tol = %.1e",cgs.tol);

   for (k=0;k<(nmu-1);k++)
      message(", %.1e",cgsh[k].tol);

   message("\n");
   message("[mscg]: ||eta|| = %.1e\n",cgs.rn);
#endif

   while ((ncg<nmx)&&(nstop<nmu))
   {
      cg_step1(vol,nmu,Dop_dble);
      cg_step2(vol,nmu);
      cg_step3(vol,nmu,istop);
      cg_step4(vol,nmu);
      ncg+=1;

#ifdef MSCG_DBG
      if (cgs.stop)
         message("[mscg]: ncg = %d, ||rho|| =        ",ncg);
      else
         message("[mscg]: ncg = %d, ||rho|| = %.1",ncg,cgs.rn);

      for (k=0;k<(nmu-1);k++)
      {
         if (cgsh[k].stop)
            message(",        ");
         else
            message(", %.1e",cgsh[k].gh*cgs.rn);
      }

      message("\n");
#endif

      nstop=set_stop_flag(nmu);
   }

#ifdef MSCG_DBG
   message("[mscg]: ncg = %d, nstop = %d\n",ncg,nstop);
#endif

   if ((ncg==nmx)&&(nstop<nmu))
   {
      status[0]=-1;
      return;
   }

   k=cgs.k;
   ie=check_res(vol,mu[k],Dop_dble,wsd,nmx,istop,res[k],eta,
                psi[k],&ncg);

#ifdef MSCG_DBG
   message("[mscg]: Check residue: ie,ncg = %d,%d",ie,ncg);
#endif

   for (k=0;k<nmu;k++)
   {
      if (k!=cgs.k)
      {
         ic=check_res(vol,mu[k],Dop_dble,wsd,nmx,istop,res[k],eta,
                      psi[k],&ncg);
         ie|=ic;

#ifdef MSCG_DBG
         message("; %d,%d",ic,ncg);
#endif
      }
   }

#ifdef MSCG_DBG
   message("\n");
#endif

   if (ie!=0)
      status[0]=-1;
   else
      status[0]=ncg;

   return;
}
