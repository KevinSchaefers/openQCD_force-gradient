
/*******************************************************************************
*
* File fgcr.c
*
* Copyright (C) 2005, 2011, 2013, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic flexible GCR solver program for the lattice Dirac equation.
*
*   double fgcr(int vol,void (*Dop)(spinor_dble *s,spinor_dble *r),
*               void (*Mop)(int k,spinor *rho,spinor *phi,spinor *chi),
*               spinor **ws,spinor_dble **wsd,int nkv,int nmx,int istop,
*               double res,spinor_dble *eta,spinor_dble *psi,int *status)
*     Solution of the Dirac equation D*psi=eta for given source eta, using
*     the preconditioned GCR algorithm.
*
* This program uses single-precision arithmetic to reduce the execution
* time, but obtains the solution with double-precision accuracy.
*
* The programs Dop() and Mop() for the operator D and the preconditioner M
* are assumed to have the following properties:
*
*   void Dop(spinor_dble *s,spinor_dble *r)
*     Application of the (global) operator D to the Dirac field s and
*     assignment of the result to r. On exit the source field s is
*     unchanged.
*
*   void Mop(int k,spinor *rho,spinor *phi,spinor *chi)
*     Approximate solution of the equation D*phi=rho in the k'th step of
*     the GCR algorithm. On exit rho is unchanged and chi=D*phi.
*
* Mop() is not required to be a linear operator and may involve an iterative
* procedure with a dynamical stopping criterion, for example. The field phi
* merely defines the next search direction and can in principle be chosen
* arbitrarily.
*
* The other parameters of the program fgcr() are:
*
*   vol     Number of elements per OpenMP thread of the spinor fields
*           on which Dop() and Mop() act.
*
*   nkv     Maximal number of Krylov vectors generated before the GCR
*           algorithm is restarted.
*
*   nmx     Maximal total number of Krylov vectors that may be generated.
*
*   istop   Stopping criterion (0: L_2 norm based, 1: uniform norm based).
*
*   res     Desired maximal relative residue of the calculated solution.
*
*   ws      Array of at least 2*nkv+1 single-precision spinor fields
*           (used as work space).
*
*   wsd     Array of at least 1 double-precision spinor field (used
*           as work space).
*
*   eta     Source field (unchanged on exit).
*
*   psi     Calculated approximate solution of the Dirac equation
*           D*psi=eta.
*
*   status  If the program terminates normally, status reports the total
*           number of Krylov vectors that were required for the solution
*           of the Dirac equation. Otherwise status is set to -1.
*
* Independently of whether the program succeeds in solving the Dirac equation
* to the desired accuracy, the program returns the norm of the residue of
* the field psi.
*
* The fields eta and psi as well as the fields in the workspaces are assumed
* to be such that the operators Dop() and Mop() can act on them.
*
* Some debugging output is printed to stdout on process 0 if the macro
* FGCR_DBG is defined.
*
* The program fgcr() is assumed to be called by the OpenMP master thread
* and on all MPI processes simultaneously.
*
* If SSE (AVX) instructions are used, the Dirac spinors must be aligned to
* a 16 (32) byte boundary.
*
*******************************************************************************/

#define FGCR_C

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

#define PRECISION_LIMIT ((double)(100.0f*FLT_EPSILON))

static int nkm=0;
static float *b;
static complex *a,*c;
static spinor **phi,**chi,*rho;
static spinor_dble *wrk;


static void alloc_arrays(int nkv)
{
   if (nkm>0)
   {
      free(a);
      free(b);
   }

   a=malloc(nkv*(nkv+1)*sizeof(*a));
   b=malloc(nkv*sizeof(*b));

   error_loc((a==NULL)||(b==NULL),1,"alloc_arrays [fgcr.c]",
             "Unable to allocate auxiliary arrays");

   c=a+nkv*nkv;
   nkm=nkv;
}


static double get_res(int vol,int istop)
{
   double rn;

   if (istop)
      rn=(double)(unorm(vol,3,rho));
   else
   {
      rn=(double)(norm_square(vol,3,rho));
      rn=sqrt(rn);
   }

   return rn;
}


static void gcr_init(int vol,int nkv,spinor **ws,spinor_dble **wsd,
                     spinor_dble *eta,spinor_dble *psi)
{
   phi=ws;
   rho=ws[nkv];
   chi=ws+nkv+1;
   wrk=wsd[0];

   set_sd2zero(vol,2,psi);
   assign_sd2s(vol,2,eta,rho);
}


static void gcr_step(int vol,int k,int nkv,
                     void (*Mop)(int k,spinor *rho,spinor *phi,spinor *chi))
{
   int l;
   complex z;

   (*Mop)(k,rho,phi[k],chi[k]);

   for (l=0;l<k;l++)
   {
      a[nkv*l+k]=spinor_prod(vol,3,chi[l],chi[k]);
      z.re=-a[nkv*l+k].re;
      z.im=-a[nkv*l+k].im;
      mulc_spinor_add(vol,2,chi[k],chi[l],z);
   }

   b[k]=normalize(vol,3,chi[k]);
   c[k]=spinor_prod(vol,3,chi[k],rho);
   z.re=-c[k].re;
   z.im=-c[k].im;
   mulc_spinor_add(vol,2,rho,chi[k],z);
}


static void update_psi(int vol,int k,int nkv,spinor_dble *eta,spinor_dble *psi,
                       void (*Dop)(spinor_dble *s,spinor_dble *r))
{
   int l,i;
   float r;
   complex z;

   for (l=k;l>=0;l--)
   {
      z.re=c[l].re;
      z.im=c[l].im;

      for (i=(l+1);i<=k;i++)
      {
         z.re-=(a[l*nkv+i].re*c[i].re-a[l*nkv+i].im*c[i].im);
         z.im-=(a[l*nkv+i].re*c[i].im+a[l*nkv+i].im*c[i].re);
      }

      r=1.0f/b[l];
      c[l].re=z.re*r;
      c[l].im=z.im*r;
   }

   set_s2zero(vol,2,rho);

   for (l=k;l>=0;l--)
      mulc_spinor_add(vol,2,rho,phi[l],c[l]);

   add_s2sd(vol,2,rho,psi);
   (*Dop)(psi,wrk);
   diff_sd2s(vol,2,eta,wrk,rho);
}


double fgcr(int vol,void (*Dop)(spinor_dble *s,spinor_dble *r),
            void (*Mop)(int k,spinor *eta,spinor *psi,spinor *chi),
            spinor **ws,spinor_dble **wsd,int nkv,int nmx,int istop,
            double res,spinor_dble *eta,spinor_dble *psi,int *status)
{
   int k,iprms[4];
   double rn,rn_old,tol,dprms[1];
#ifdef FGCR_DBG
   double xn;
#endif

   error_loc((vol<=0)||(nkv<1)||(nmx<1)||(res<=DBL_EPSILON)||
             (istop<0)||(istop>1),1,"fgcr [fgcr.c]",
             "Parameters are out of range");

   if (NPROC>1)
   {
      iprms[0]=vol;
      iprms[1]=nkv;
      iprms[2]=nmx;
      iprms[3]=istop;
      dprms[0]=res;

      MPI_Bcast(iprms,4,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((iprms[0]!=vol)||(iprms[1]!=nkv)||(iprms[2]!=nmx)||
            (iprms[3]!=istop)||(dprms[0]!=res),1,"fgcr [fgcr.c]",
            "Parameters are not global");
   }

   if (nkv>nkm)
      alloc_arrays(nkv);

   gcr_init(vol,nkv,ws,wsd,eta,psi);
   rn=get_res(vol,istop);
   tol=res*rn;
   status[0]=0;

   while (rn>tol)
   {
      rn_old=rn;

      for (k=0;;k++)
      {
         gcr_step(vol,k,nkv,Mop);
         rn=get_res(vol,istop);
         status[0]+=1;

#ifdef FGCR_DBG
         message("[fgcr]: k = %d, status = %d, rn = %.1e\n",
                 k,status[0],rn);
#endif

         if ((rn<=tol)||(rn<(PRECISION_LIMIT*rn_old))||(k==(nkv-1))||
             (status[0]>=nmx))
            break;
      }

      update_psi(vol,k,nkv,eta,psi,Dop);
      rn=get_res(vol,istop);

#ifdef FGCR_DBG
      if (istop)
         xn=unorm_dble(vol,3,psi);
      else
      {
         assign_sd2s(vol,2,psi,chi[0]);
         xn=(double)(norm_square(vol,3,chi[0]));
         xn=sqrt(xn);
      }

      message("[fgcr]: status = %d, ||psi|| = %.1e, ||rho|| = %.1e\n",
              status[0],xn,rn);
#endif

      if ((status[0]>=nmx)&&(rn>tol))
      {
         status[0]=-1;
         return rn;
      }
   }

   return rn;
}
