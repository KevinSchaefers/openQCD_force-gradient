
/*******************************************************************************
*
* File fgcr4vd.c
*
* Copyright (C) 2007-2013, 2018, 2020, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic flexible GCR solver program for the double-precision little Dirac
* equation.
*
*   double fgcr4vd(int vol,void (*Dop)(complex_dble *v,complex_dble *w),
*                  void (*Mop)(complex_dble *v,complex_dble *w,complex_dble *z),
*                  complex_dble **wvd,int nkv,int nmx,double res,
*                  complex_dble *eta,complex_dble *psi,int *status)
*     Obtains an approximate solution psi of the equation Dop*psi=eta
*     for given source eta using the FGCR algorithm with an arbitrary
*     preconditioner Mop.
*
* The programs Dop() and Mop() are assumed to have the following properties:
*
*   void Dop(complex_dble *v,complex_dble *w)
*     Implementation of a (global) linear operator Dop: v->w acting on
*     double-precision complex fields. On exit v is unchanged.
*
*   void Mop(complex_dble *v,complex_dble *w,complex_dble *z)
*     Computation of an approximate inverse w~Dop^(-1)*v and of the product
*     z=Dop*w. On exit v is unchanged.
*
* Mop() is not required to be a linear operator and may involve an iterative
* procedure with a dynamical stopping criterion, for example. The computed
* field w merely defines the next search direction and can in principle be
* chosen arbitrarily.
*
* The other parameters of the program fgcr4vd() are:
*
*   vol     Length per OpenMP thread of the complex fields on which Dop()
*           and Mop() act.
*
*   wvd     Workspace of at least 2*nkv+1 complex fields of the type on
*           which Dop() can act.
*
*   nkv     Maximal number of Krylov vectors generated before the GCR
*           algorithm is restarted (nkv>=2).
*
*   nmx     Maximal total number of Krylov vectors that may be generated.
*
*   res     Desired maximal relative residue |eta-Dop*psi|/|eta| of the
*           calculated solution.
*
*   eta     Source field (unchanged on exit).
*
*   psi     Calculated approximate solution.
*
*   status  If the program terminates normally, status reports the total
*           number of Krylov vectors that were required for the solution
*           of the Dirac equation. Otherwise status is set to -1.
*
* Independently of whether the program succeeds in solving the equation to
* the desired accuracy, the program assigns the last calculated approximate
* solution to psi and returns the norm of the residue of that field.
*
* The fields eta and psi are assumed to be such that the operators Dop()
* and Mop() can act on them.
*
* Some debugging output is printed to stdout on process 0 if the macro
* FGCR4VD_DBG is defined at compilation time.
*
* The program fgcr4vd() is assumed to be called by the OpenMP master thread
* on all MPI processes simultaneously.
*
* If SSE (AVX) instructions are used, the field arrays must be aligned to a
* 16 (32) byte boundary.
*
*******************************************************************************/

#define FGCR4VD_C

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

#define PRECISION_LIMIT (100.0*DBL_EPSILON)

static int nkm=0;
static double rn;
static double *rdsm0,*rdsm1,*b;
static complex_dble *rho,**phi,**chi;
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


static void alloc_arrays(int nkv)
{
   if (nkm>0)
   {
      free(a);
      free(b);
      free(cdsm0);
   }

   a=malloc(nkv*(nkv+1)*sizeof(*a));
   b=malloc(nkv*sizeof(*b));
   cdsm0=malloc(2*nkv*sizeof(*cdsm0));

   error((a==NULL)||(b==NULL)||(cdsm0==NULL),1,"Alloc_arrays [fgcr4vd.c]",
         "Unable to allocate auxiliary arrays");

   c=a+nkv*nkv;
   cdsm1=cdsm0+nkv;
   rdsm0=(double*)(cdsm0);
   rdsm1=(double*)(cdsm1);
   nkm=nkv;
}


static void gcr_init(int vol,int nkv,complex_dble **wvd,complex_dble *eta,
                     complex_dble *psi)
{
   int k;
   qflt s;

   rho=wvd[0];
   phi=wvd+1;
   chi=wvd+nkv+1;
   rn=0.0;

#pragma omp parallel private(k,s) reduction(+ : rn)
   {
      k=omp_get_thread_num();

      set_vd2zero(vol,0,psi+k*vol);
      assign_vd2vd(vol,0,eta+k*vol,rho+k*vol);
      s=vnorm_square_dble(vol,0,eta+k*vol);
      rn=s.q[0];
   }

   rdsm0[0]=rn;
   global_dsum(1);
   rn=sqrt(rdsm1[0]);
}


static int gcr_step(int vol,int m,int nkv,
                    void (*Mop)(complex_dble *v,complex_dble *w,
                                complex_dble *z))
{
   int k,l;
   double r;
   complex_dble z;
   complex_qflt w;

   (*Mop)(rho,phi[m],chi[m]);

   if (m>0)
   {
      for (l=0;l<m;l++)
      {
         cdsm0[l].re=0.0;
         cdsm0[l].im=0.0;
      }

#pragma omp parallel private(k,l,w) reduction(sum_complex_dble : cdsm0[0:m])
      {
         k=omp_get_thread_num();

         for (l=0;l<m;l++)
         {
            w=vprod_dble(vol,0,chi[l]+k*vol,chi[m]+k*vol);
            cdsm0[l].re=w.re.q[0];
            cdsm0[l].im=w.im.q[0];
         }
      }

      global_dsum(2*m);

      for (l=0;l<m;l++)
      {
         a[nkv*l+m].re=cdsm1[l].re;
         a[nkv*l+m].im=cdsm1[l].im;
      }
   }

   for (l=0;l<2;l++)
   {
      cdsm0[l].re=0.0;
      cdsm0[l].im=0.0;
   }

#pragma omp parallel private(k,l,z,w) reduction(sum_complex_dble : cdsm0[0:2])
   {
      k=omp_get_thread_num();

      for (l=0;l<m;l++)
      {
         z.re=-cdsm1[l].re;
         z.im=-cdsm1[l].im;
         mulc_vadd_dble(vol,0,chi[m]+k*vol,chi[l]+k*vol,z);
      }

      w=vprod_dble(vol,0,chi[m]+k*vol,rho+k*vol);
      cdsm0[0].re=w.re.q[0];
      cdsm0[0].im=w.im.q[0];

      w.re=vnorm_square_dble(vol,0,chi[m]+k*vol);
      cdsm0[1].re=w.re.q[0];
   }

   global_dsum(3);

   b[m]=sqrt(cdsm1[1].re);

   if (b[m]!=0.0)
   {
      r=1.0/b[m];
      c[m].re=r*cdsm1[0].re;
      c[m].im=r*cdsm1[0].im;
      rn=0.0;

#pragma omp parallel private(k,z,w) reduction(+ : rn)
      {
         k=omp_get_thread_num();

         vscale_dble(vol,0,r,chi[m]+k*vol);

         z.re=-c[m].re;
         z.im=-c[m].im;
         mulc_vadd_dble(vol,0,rho+k*vol,chi[m]+k*vol,z);

         w.re=vnorm_square_dble(vol,0,rho+k*vol);
         rn=w.re.q[0];
      }

      rdsm0[0]=rn;
      global_dsum(1);
      rn=sqrt(rdsm1[0]);

      return 0;
   }
   else
      return 1;
}


static void update_psi(int vol,int m,int nkv,int irho,
                       void (*Dop)(complex_dble *v,complex_dble *w),
                       complex_dble *eta,complex_dble *psi)
{
   int k,l,j;
   double r;
   qflt s;
   complex_dble z;

   if (b[m]==0.0)
      m-=1;

   for (l=m;l>=0;l--)
   {
      z.re=c[l].re;
      z.im=c[l].im;

      for (j=(l+1);j<=m;j++)
      {
         z.re-=(a[l*nkv+j].re*c[j].re-a[l*nkv+j].im*c[j].im);
         z.im-=(a[l*nkv+j].re*c[j].im+a[l*nkv+j].im*c[j].re);
      }

      r=1.0/b[l];
      c[l].re=z.re*r;
      c[l].im=z.im*r;
   }

#pragma omp parallel private(k,l)
   {
      k=omp_get_thread_num();

      for (l=m;l>=0;l--)
         mulc_vadd_dble(vol,0,psi+k*vol,phi[l]+k*vol,c[l]);
   }

   if (irho)
   {
      Dop(psi,rho);
      rn=0.0;

#pragma omp parallel private(k,s) reduction(+ : rn)
      {
         k=omp_get_thread_num();

         diff_vd2vd(vol,0,eta+k*vol,rho+k*vol);
         s=vnorm_square_dble(vol,0,rho+k*vol);
         rn=s.q[0];
      }

      rdsm0[0]=rn;
      global_dsum(1);
      rn=sqrt(rdsm1[0]);
   }
}


double fgcr4vd(int vol,void (*Dop)(complex_dble *v,complex_dble *w),
               void (*Mop)(complex_dble *v,complex_dble *w,complex_dble *z),
               complex_dble **wvd,int nkv,int nmx,double res,
               complex_dble *eta,complex_dble *psi,int *status)
{
   int n,m,is,iprms[3];
   double rn_old,tol,dprms[1];

   error_root((vol<=0)||(nkv<2)||(nmx<1)||(res<=DBL_EPSILON),1,
              "fgcr4vd [fgcr4vd.c]","Improper choice of vol,nkv,nmx or res");

   if (NPROC>1)
   {
      iprms[0]=vol;
      iprms[1]=nkv;
      iprms[2]=nmx;
      dprms[0]=res;

      MPI_Bcast(iprms,3,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((iprms[0]!=vol)||(iprms[1]!=nkv)||(iprms[2]!=nmx)||(dprms[0]!=res),
            1,"fgcr4vd [fgcr4vd.c]","Parameters are not global");
   }

   if (nkv>nkm)
      alloc_arrays(nkv);

   gcr_init(vol,nkv,wvd,eta,psi);
   tol=res*rn;
   n=0;

   if (rn!=0.0)
   {
      while (n<nmx)
      {
         rn_old=rn;
         m=0;
         is=0;

#ifdef FGCR4VD_DBG
         message("[fgcr4vd]: New cycle, ||rho|| = %.2e\n",rn_old);
#endif

         while ((m<nkv)&&(n<nmx))
         {
            is=gcr_step(vol,m,nkv,Mop);
            m+=1;
            n+=1;

#ifdef FGCR4VD_DBG
            message("[fgcr4vd]: Status = %3d, k = %3d, ||rho|| = %.2e\n",n,m,rn);
            if (is==1)
               message("[fgcr4vd]: Awhat is singular\n");
#endif

            if ((rn<=tol)||(rn<=(PRECISION_LIMIT*rn_old))||(is==1))
               break;
         }

         update_psi(vol,m-1,nkv,(rn>tol)&&(n<nmx)&&(is==0),Dop,eta,psi);

         if ((rn<=tol)||(is==1))
            break;
      }
   }

   if (rn<=tol)
      status[0]=n;
   else
      status[0]=-1;

#ifdef FGCR4VD_DBG
   message("[fgcr4vd]: All done (status = %d)\n\n",status[0]);
#endif

   return rn;
}
