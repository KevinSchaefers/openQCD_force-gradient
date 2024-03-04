
/*******************************************************************************
*
* File ltl_modes.c
*
* Copyright (C) 2011, 2013, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the little modes.
*
*   int set_ltl_modes(void)
*     Computes the little modes, the little-little Dirac operator and its
*     inverse, assuming the eo-preconditioned double-precision little Dirac
*     operator is up-to-date. The program returns 0 if the inversion was
*     safe and 1 if not.
*
*   complex_dble *ltl_matrix(void)
*     Returns the pointer to an Ns x Ns matrix that represents the
*     *inverse* of the double-precision little-little Dirac operator.
*
*   void dfl_Lvd(complex_dble *vd)
*     Replaces vd by P_L*vd=vd-A*P0*M*P0*vd.
*
*   void dfl_LRvd(complex_dble *vd,complex_dble *wd)
*     Sets wd to P0*M*P0*vd and replaces vd by P_L*vd.
*
*   void dfl_RLvd(complex_dble *vd,complex_dble *wd)
*     Replaces vd by vd-P0*M*P0*wd and wd by P_L*wd.
*
*   void dfl_Lv(complex *v)
*     Replaces v by P_L*v.
*
* For a description of the little Dirac operator and the associated data
* structures see README.Aw. As usual, Ns denotes the number of deflation
* modes in each block of the DFL_BLOCKS grid.
*
* In the description of the programs dfl_{Lvd,LRvd,RLvd,Lv}(), A denotes
* the eo-preconditioned little Dirac operator, P0 the orthogonal projector
* to the little modes and M the inverse of the little-little Dirac operator.
* All these programs only act on the field components associated with the
* even blocks.
*
* The inversion of a double-precision complex matrix is considered to be
* safe if and only if its Frobenius condition number is less than 10^6.
* The program set_ltl_modes() requires a workspace of 2 double-precision
* complex vector fields (see utils/wspace.c).
*
* The programs ltl_matrix() and dfl_{Lvd,LRvd,RLvd,Lv}() assume that the
* little modes are up-to-date.
*
* All programs in this module except for ltl_matrix() are assumed to be
* called by the OpenMP master thread on all MPI processes simultaneously.
*
* The program ltl_matrix() is thread-safe.
*
*******************************************************************************/

#define LTL_MODES_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "utils.h"
#include "vflds.h"
#include "linalg.h"
#include "little.h"
#include "global.h"

#define MAX_FROBENIUS 1.0e6

static int Ns=0,nvh,nvt;
static complex **vs;
static complex_dble **vds,*Ads,*B,*C;
static cmat_wsp_t *cwsp;


static void alloc_matrices(void)
{
   int nb,nmat;
   Aw_dble_t Awd;

   Awd=Awop_dble();
   Ns=Awd.Ns;
   nb=Awd.nb;

   nvh=(Ns*nb)/2;
   nvt=nvh/NTHREAD;
   nmat=Ns*Ns;

   vs=vflds();
   vds=vdflds();

   Ads=amalloc(3*nmat*sizeof(*Ads),5);
   cwsp=alloc_cmat_wsp(Ns);
   error((Ads==NULL)||(cwsp==NULL),1,"alloc_matrices [ltl_modes.c]",
         "Unable to allocate matrices");

   B=Ads+nmat;
   C=Ads+2*nmat;
}


static void sum_vdprod(int n,complex_dble *z,complex_dble *w)
{
   int l;

   if (NPROC>1)
   {
      MPI_Reduce(z,w,2*n,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(w,2*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
   {
      for (l=0;l<n;l++)
         w[l]=z[l];
   }
}


static void sum_vdnorm(int n,double *r,double *s)
{
      int l;

   if (NPROC>1)
   {
      MPI_Reduce(r,s,n,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(s,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
   {
      for (l=0;l<n;l++)
         s[l]=r[l];
   }
}


static void vs2vds(void)
{
   int k,l,j,ifail;
   double r,s;
   complex_qflt cqsm;

   ifail=0;
   r=0.0;

#pragma omp parallel private(k,l,cqsm) reduction(+ : r)
   {
      k=omp_get_thread_num();

      for (l=0;l<Ns;l++)
         assign_v2vd(nvt,0,vs[Ns+l]+k*nvt,vds[l]+k*nvt);

      cqsm.re=vnorm_square_dble(nvt,0,vds[0]+k*nvt);
      r=cqsm.re.q[0];
   }

   sum_vdnorm(1,&r,&s);

   r=sqrt(s);
   if (r==0.0)
      ifail=1;
   else
      r=1.0/r;

   for (l=1;l<Ns;l++)
   {
      for (j=0;j<l;j++)
      {
         B[j].re=0.0;
         B[j].im=0.0;
      }

#pragma omp parallel private(k,j,cqsm) reduction(sum_complex_dble : B[0:l])
      {
         k=omp_get_thread_num();

         vscale_dble(nvt,0,r,vds[l-1]+k*nvt);

         for (j=0;j<l;j++)
         {
            cqsm=vprod_dble(nvt,0,vds[j]+k*nvt,vds[l]+k*nvt);
            B[j].re=-cqsm.re.q[0];
            B[j].im=-cqsm.im.q[0];
         }
      }

      sum_vdprod(l,B,C);
      r=0.0;

#pragma omp parallel private(k,j,cqsm) reduction(+ : r)
      {
         k=omp_get_thread_num();

         for (j=0;j<l;j++)
            mulc_vadd_dble(nvt,0,vds[l]+k*nvt,vds[j]+k*nvt,C[j]);

         cqsm.re=vnorm_square_dble(nvt,0,vds[l]+k*nvt);
         r=cqsm.re.q[0];
      }

      sum_vdnorm(1,&r,&s);

      r=sqrt(s);
      if (r==0.0)
         ifail=1;
      else
         r=1.0/r;
   }

   vscale_dble(nvt,2,r,vds[Ns-1]);

   error_root(ifail,1,"vs2vsd [ltl_modes.c]",
              "Degenerate basis vector fields");
}


static void vds2Avds(void)
{
   int k,l;
   complex_dble **wvd;

   wvd=reserve_wvd(2);
   assign_vd2vd(nvt,2,vds[0],wvd[0]);

   for (l=0;l<Ns;l++)
   {
      Awhat_dble(wvd[0],wvd[1]);

#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();

         if (l>0)
            assign_vd2v(2*nvt,0,vds[l-1]+2*k*nvt,vs[l-1]+2*k*nvt);

         assign_vd2vd(nvt,0,wvd[1]+k*nvt,vds[l]+nvh+k*nvt);

         if (l<(Ns-1))
            assign_vd2vd(nvt,0,vds[l+1]+k*nvt,wvd[0]+k*nvt);
      }
   }

   assign_vd2v(2*nvt,2,vds[Ns-1],vs[Ns-1]);
   release_wvd();
}


static int set_Ads(void)
{
   int k,l,j;
   int nmat,ifail;
   double cn;
   complex_qflt cqsm;

   nmat=Ns*Ns;
   set_vd2zero(nmat,0,B);

#pragma omp parallel private(k,l,j,cqsm) \
   reduction(sum_complex_dble : B[0:nmat])
   {
      k=omp_get_thread_num();

      for (l=0;l<Ns;l++)
      {
         for (j=0;j<Ns;j++)
         {
            cqsm=vprod_dble(nvt,0,vds[l]+k*nvt,vds[j]+nvh+k*nvt);
            B[Ns*l+j].re=cqsm.re.q[0];
            B[Ns*l+j].im=cqsm.im.q[0];
         }
      }
   }

   sum_vdprod(nmat,B,C);
   ifail=cmat_inv_dble(Ns,C,cwsp,Ads,&cn);
   ifail|=(cn>MAX_FROBENIUS);

   return ifail;
}


int set_ltl_modes(void)
{
   int ifail;

   if (Ns==0)
      alloc_matrices();

   vs2vds();
   vds2Avds();
   ifail=set_Ads();

   return ifail;
}


complex_dble *ltl_matrix(void)
{
   return Ads;
}


void dfl_Lvd(complex_dble *vd)
{
   int k,l;
   complex_qflt w;

   set_vd2zero(Ns,0,B);

#pragma omp parallel private(k,l,w) reduction(sum_complex_dble : B[0:Ns])
   {
      k=omp_get_thread_num();

      for (l=0;l<Ns;l++)
      {
         w=vprod_dble(nvt,0,vds[l]+k*nvt,vd+k*nvt);
         B[l].re=-w.re.q[0];
         B[l].im=-w.im.q[0];
      }
   }

   sum_vdprod(Ns,B,C);
   cmat_vec_dble(Ns,Ads,C,B);

#pragma omp parallel private(k,l)
   {
      k=omp_get_thread_num();

      for (l=0;l<Ns;l++)
         mulc_vadd_dble(nvt,0,vd+k*nvt,vds[l]+nvh+k*nvt,B[l]);
   }
}


void dfl_LRvd(complex_dble *vd,complex_dble *wd)
{
   int k,l;
   complex_dble z;
   complex_qflt w;

   set_vd2zero(Ns,0,B);

#pragma omp parallel private(k,l,w) reduction(sum_complex_dble : B[0:Ns])
   {
      k=omp_get_thread_num();

      for (l=0;l<Ns;l++)
      {
         w=vprod_dble(nvt,0,vds[l]+k*nvt,vd+k*nvt);
         B[l].re=w.re.q[0];
         B[l].im=w.im.q[0];
      }
   }

   sum_vdprod(Ns,B,C);
   cmat_vec_dble(Ns,Ads,C,B);

#pragma omp parallel private(k,l,z)
   {
      k=omp_get_thread_num();

      set_vd2zero(nvt,0,wd+k*nvt);

      for (l=0;l<Ns;l++)
      {
         z.re=B[l].re;
         z.im=B[l].im;
         mulc_vadd_dble(nvt,0,wd+k*nvt,vds[l]+k*nvt,z);

         z.re=-z.re;
         z.im=-z.im;
         mulc_vadd_dble(nvt,0,vd+k*nvt,vds[l]+nvh+k*nvt,z);
      }
   }
}


void dfl_RLvd(complex_dble *vd,complex_dble *wd)
{
   int k,l;
   complex_qflt w;

   set_vd2zero(Ns,0,B);

#pragma omp parallel private(k,l,w) reduction(sum_complex_dble : B[0:Ns])
   {
      k=omp_get_thread_num();

      for (l=0;l<Ns;l++)
      {
         w=vprod_dble(nvt,0,vds[l]+k*nvt,wd+k*nvt);
         B[l].re=-w.re.q[0];
         B[l].im=-w.im.q[0];
      }
   }

   sum_vdprod(Ns,B,C);
   cmat_vec_dble(Ns,Ads,C,B);

#pragma omp parallel private(k,l)
   {
      k=omp_get_thread_num();

      for (l=0;l<Ns;l++)
      {
         mulc_vadd_dble(nvt,0,wd+k*nvt,vds[l]+nvh+k*nvt,B[l]);
         mulc_vadd_dble(nvt,0,vd+k*nvt,vds[l]+k*nvt,B[l]);
      }
   }
}


void dfl_Lv(complex *v)
{
   int k,l;
   complex z;

   set_vd2zero(Ns,0,B);

#pragma omp parallel private(k,l) reduction(sum_complex_dble : B[0:Ns])
   {
      k=omp_get_thread_num();

      for (l=0;l<Ns;l++)
         B[l]=vprod(nvt,0,vs[l]+k*nvt,v+k*nvt);
   }

   sum_vdprod(Ns,B,C);
   cmat_vec_dble(Ns,Ads,C,B);

#pragma omp parallel private(k,l,z)
   {
      k=omp_get_thread_num();

      for (l=0;l<Ns;l++)
      {
         z.re=-(float)(B[l].re);
         z.im=-(float)(B[l].im);
         mulc_vadd(nvt,0,v+k*nvt,vs[l]+nvh+k*nvt,z);
      }
   }
}
