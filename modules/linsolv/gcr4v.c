
/*******************************************************************************
*
* File gcr4v.c
*
* Copyright (C) 2020, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic one-pass GCR solver program for the single-precision little Dirac
* equation.
*
*   float gcr4v(int vol,void (*Dop)(complex *v,complex *w),complex **wv,
*               int nmx,float res,complex *eta,complex *psi,int *status)
*     Obtains an approximate solution psi of the equation Dop*psi=eta
*     for given source eta using the GCR algorithm w/o restarting.
*
* This program is intended to be used as preconditioner in another solver
* program such as fgcr4vd().
*
* The program Dop() is assumed to have the following properties:
*
*   void Dop(complex *v,complex *w)
*     Implementation of a (global) linear operator Dop: v->w acting on
*     single-precision complex fields. On exit v is unchanged.
*
* The other parameters of the program gcr4v() are:
*
*   vol     Length per OpenMP thread of the complex fields on which
*           Dop() acts.
*
*   wv      Workspace of at least 2*nmx+1 complex fields of the type
*           on which Dop() can act.
*
*   nmx     Maximal number of Krylov vectors that may be generated
*           (a minimal value of 2 is required).
*
*   res     Desired maximal relative residue |eta-Dop*psi|/|eta| of
*           the calculated solution.
*
*   eta     Source field (unchanged on exit).
*
*   psi     Calculated approximate solution.
*
*   status  If the program is able to solve the equation to the specified
*           precision, status reports the total number of Krylov vectors
*           that were required. Otherwise status is set to -1.
*
* Independently of whether the program succeeds in solving the equation to
* the desired accuracy, the program assigns the last calculated approximate
* solution to psi and returns the norm of the residue of that field.
*
* The fields eta and psi are assumed to be such that the operator Dop()
* can act on them.
*
* Some debugging output is printed to stdout on process 0 if the macro
* GCR4V_DBG is defined at compilation time.
*
* The program gcr4v() is assumed to be called by the OpenMP master thread
* on all MPI processes simultaneously.
*
* If SSE or AVX instructions are used, the field arrays must be aligned to
* a 16 byte boundary.
*
*******************************************************************************/

#define GCR4V_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "mpi.h"
#include "utils.h"
#include "vflds.h"
#include "linalg.h"
#include "little.h"
#include "linsolv.h"
#include "global.h"

#define PRECISION_LIMIT ((float)(100.0*FLT_EPSILON))

static int nkm=0;
static float rn;
static double *rdsm0,*rdsm1,*b;
static complex *rho,**phi,**chi;
static complex_dble *cdsm0,*cdsm1,*a,*c;


static void global_dsum(int n)
{
   int l;

   if ((NPROC>1)&&(n>0))
   {
      MPI_Reduce(rdsm0,rdsm1,n,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(rdsm1,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
   {
      for (l=0;l<n;l++)
         rdsm1[l]=rdsm0[l];
   }
}


static void alloc_arrays(int nmx)
{
   error(nmx<2,1,"alloc_arrays [gcr4v.c]","nmx must be at least 2");

   if (nkm>0)
   {
      free(a);
      free(b);
      free(cdsm0);
   }

   a=malloc(nmx*(nmx+1)*sizeof(*a));
   b=malloc(nmx*sizeof(*b));
   cdsm0=malloc(2*nmx*sizeof(*cdsm0));

   error((a==NULL)||(b==NULL)||(cdsm0==NULL),1,"Alloc_arrays [gcr4v.c]",
         "Unable to allocate auxiliary arrays");

   c=a+nmx*nmx;
   cdsm1=cdsm0+nmx;
   rdsm0=(double*)(cdsm0);
   rdsm1=(double*)(cdsm1);
   nkm=nmx;
}


static void gcr_init(int vol,int nmx,complex **wv,complex *eta,complex *psi)
{
   int k;
   double rd;

   rho=wv[0];
   phi=wv+1;
   chi=wv+nmx+1;
   rd=0.0;

#pragma omp parallel private(k) reduction(+ : rd)
   {
      k=omp_get_thread_num();

      set_v2zero(vol,0,psi+k*vol);
      assign_v2v(vol,0,eta+k*vol,rho+k*vol);
      rd=vnorm_square(vol,0,eta+k*vol);
   }

   rdsm0[0]=rd;
   global_dsum(1);
   rn=(float)(sqrt(rdsm1[0]));
}


static int gcr_step(int vol,int n,int nmx,void (*Dop)(complex *v,complex *w))
{
   int k,l;
   double r,s;
   complex z;

   (*Dop)(rho,chi[n]);

   if (n>0)
   {
      for (l=0;l<n;l++)
      {
         cdsm0[l].re=0.0;
         cdsm0[l].im=0.0;
      }

#pragma omp parallel private(k,l) reduction(sum_complex_dble : cdsm0[0:n])
      {
         k=omp_get_thread_num();

         for (l=0;l<n;l++)
            cdsm0[l]=vprod(vol,0,chi[l]+k*vol,chi[n]+k*vol);
      }

      global_dsum(2*n);

      for (l=0;l<n;l++)
      {
         a[nmx*l+n].re=cdsm1[l].re;
         a[nmx*l+n].im=cdsm1[l].im;
      }
   }

   for (l=0;l<2;l++)
   {
      cdsm0[l].re=0.0;
      cdsm0[l].im=0.0;
   }

#pragma omp parallel private(k,l,z) reduction(sum_complex_dble : cdsm0[0:2])
   {
      k=omp_get_thread_num();

      for (l=0;l<n;l++)
      {
         z.re=-(float)(cdsm1[l].re);
         z.im=-(float)(cdsm1[l].im);
         mulc_vadd(vol,0,chi[n]+k*vol,chi[l]+k*vol,z);
      }

      assign_v2v(vol,0,rho+k*vol,phi[n]+k*vol);
      cdsm0[0]=vprod(vol,0,chi[n]+k*vol,rho+k*vol);
      cdsm0[1].re=vnorm_square(vol,0,chi[n]+k*vol);
   }

   global_dsum(3);

   b[n]=sqrt(cdsm1[1].re);

   if (b[n]!=0.0)
   {
      r=1.0/b[n];
      c[n].re=r*cdsm1[0].re;
      c[n].im=r*cdsm1[0].im;
      s=0.0;

#pragma omp parallel private(k,z) reduction(+ : s)
      {
         k=omp_get_thread_num();

         vscale(vol,0,(float)(r),chi[n]+k*vol);

         z.re=-(float)(c[n].re);
         z.im=-(float)(c[n].im);
         mulc_vadd(vol,0,rho+k*vol,chi[n]+k*vol,z);

         s=vnorm_square(vol,0,rho+k*vol);
      }

      rdsm0[0]=s;
      global_dsum(1);
      rn=(float)(sqrt(rdsm1[0]));

      return 0;
   }
   else
      return 1;
}


static void set_psi(int vol,int n,int nmx,complex *psi)
{
   int k,l,j;
   double r;
   complex z;
   complex_dble w;

   if (b[n]==0.0)
      n-=1;

   for (l=n;l>=0;l--)
   {
      w.re=c[l].re;
      w.im=c[l].im;

      for (j=(l+1);j<=n;j++)
      {
         w.re-=(a[l*nmx+j].re*c[j].re-a[l*nmx+j].im*c[j].im);
         w.im-=(a[l*nmx+j].re*c[j].im+a[l*nmx+j].im*c[j].re);
      }

      r=1.0/b[l];
      c[l].re=w.re*r;
      c[l].im=w.im*r;
   }

#pragma omp parallel private(k,l,z)
   {
      k=omp_get_thread_num();

      for (l=n;l>=0;l--)
      {
         z.re=(float)(c[l].re);
         z.im=(float)(c[l].im);
         mulc_vadd(vol,0,psi+k*vol,phi[l]+k*vol,z);
      }
   }
}


float gcr4v(int vol,void (*Dop)(complex *v,complex *w),complex **wv,
            int nmx,float res,complex *eta,complex *psi,int *status)
{
   int n,is;
   float rn_old,tol;

   if (nmx>nkm)
      alloc_arrays(nmx);

   gcr_init(vol,nmx,wv,eta,psi);
   tol=res*rn;
   n=0;

#ifdef GCR4V_DBG
   message("[gcr4v]: New cycle, ||rho|| = %.2e\n",rn);
#endif

   if (rn!=0.0f)
   {
      while (n<nmx)
      {
         rn_old=rn;
         is=gcr_step(vol,n,nmx,Dop);
         n+=1;

#ifdef GCR4V_DBG
         message("[gcr4v]: Status = %3d,  ||rho|| = %.2e\n",n,rn);
         if (is==1)
            message("[gcr4v]: Awhat is singular\n");
#endif

         if ((rn<=tol)||(rn<=(PRECISION_LIMIT*rn_old))||(is==1))
            break;
      }

      set_psi(vol,n-1,nmx,psi);
   }

   if (rn<=tol)
      status[0]=n;
   else
      status[0]=-1;

#ifdef GCR4V_DBG
   message("[gcr4v]: All done (status = %d)\n\n",status[0]);
#endif

   return rn;
}
