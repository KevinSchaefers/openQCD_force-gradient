
/*******************************************************************************
*
* File latavg.c
*
* Copyright (C) 2017, 2018, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Lattice averages of observable fields.
*
*   void sphere_msfld(int r,double *f)
*     Sets the field f to 1 at all points x satisfying |x|<=r and to 0
*     elsewhere, where |x| denotes the Euclidean distance of x from the
*     origin.
*
*   void sphere3d_msfld(int r,double *f)
*     Sets the field f to 1 at all points x satisfying |vec(x)|<=r and
*     to 0 elsewhere, where vec(x) is the 3d vector part of x (see the
*     notes).
*
*   void sphere_sum(int dmax,double *f,double *sm)
*     Assigns the sum of the values of the field f at the points x satisfying
*     |x|<=k to sm[k] (k=0,1,..,dmax).
*
*   void sphere3d_sum(int dmax,double *f,double *sm)
*     Assigns the sum of the values of the field f at the points x satisfying
*     |vec(x)|<=k to sm[k] (k=0,1,..,dmax).
*
*   double avg_msfld(double *f)
*     Returns the global average of the field f.
*
*   double center_msfld(double *f)
*     Subtracts the average of f from f and returns the average.
*
*   void cov_msfld(int dmax,double *f,double *g,
*                complex_dble *rf,complex_dble *rg,double *w,double *cov)
*     Computes the covariance cov[k], k=0,..,dmax, of the observable fields
*     f and g *assuming these have vanishing average* (see the notes). The
*     fields rf, rg and w are used as workspace. One may set rf=rg if f=g
*     and it is always permissible to set f=w or g=w (in which case that
*     field is changed on exit).
*
*   void cov3d_msfld(int dmax,double *f,double *g,
*                  complex_dble *rf,complex_dble *rg,double *w,double *cov)
*     Same as cov_msfld() but measuring distances in 3d rather than 4d (see
*     the notes).
*
* The programs in this module assume periodic boundary conditions in all
* directions. An error occurs if this is not the case. All programs act
* on global real scalar fields f[ix], where 0<=ix<VOLUME is the index of
* the points on the local lattice defined by the geometry routines (see
* main/README.global).
*
* The Euclidean norm |x|>=0 of a lattice point x is defined by
*
*   |x|^2=sum_mu min{y[mu],N[mu]-y[mu]}^2,
*
* where y[mu]=x[mu] mod N[mu], 0<=y[mu]<N[mu], and N[mu] denotes the global
* lattice size in direction mu. For any point x with Cartesian coordinates
* (x0,x1,x2,x3),
*
*   vec(x)=(0,x1,x2,x3)
*
* denotes its 3d vector part.
*
* The covariance of f and g computed by the program cov_msfld() is defined
* by
*
*   cov[k]=(1/V^2)*sum_{|y|<=k}*sum_z*f(y+z)*g(z), k=0,1,..,dmax,
*
*   V=no of lattice points, |y|=norm in 4d or 3d (=|vec(y)|).
*
* It is up to the calling program to ensure that f and g have vanishing
* average.
*
* All programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define LATAVG_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "utils.h"
#include "msfcts.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static const int ls[4]={L0,L1,L2,L3};
static const int ns[4]={N0,N1,N2,N3};
static int dmx=0;
static double **qsmx;
static qflt *smx;


static void alloc_smx(int dmax)
{
   int d;

   if (dmax>=dmx)
   {
      if (dmx>0)
      {
         free(qsmx);
         afree(smx);
      }

      qsmx=malloc((dmax+1)*sizeof(*qsmx));
      smx=amalloc((dmax+1)*sizeof(*smx),4);
      error((qsmx==NULL)||(smx==NULL),1,"alloc_smx [latavg.c]",
            "Unable to allocate auxiliary arrays");
      dmx=dmax+1;

      for (d=0;d<=dmax;d++)
         qsmx[d]=smx[d].q;
   }
}


static int nrmsq(int i3d,int *x)
{
   int mu,z,sm;

   sm=0;

   for (mu=i3d;mu<4;mu++)
   {
      z=cpr[mu]*ls[mu]+x[mu];

      if ((2*z)>ns[mu])
         z-=ns[mu];

      sm+=z*z;
   }

   return sm;
}


static void chk_parms(int var,char *prgm)
{
   int iprms[1];

   if (NPROC>1)
   {
      iprms[0]=var;
      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);
      error(iprms[0]!=var,1,prgm,"Parameters are not global");
   }
}


static void set_sphere_msfld(int i3d,int r,double *f)
{
   int k,ofs,vol,ix,iy,x[4];
   int dsq,rsq;

   rsq=r*r;

#pragma omp parallel private(k,ofs,vol,ix,iy,x,dsq)
   {
      k=omp_get_thread_num();
      vol=VOLUME_TRD;
      ofs=k*vol;

      for (ix=ofs;ix<(ofs+vol);ix++)
      {
         iy=ix;
         x[3]=iy%L3;
         iy/=L3;
         x[2]=iy%L2;
         iy/=L2;
         x[1]=iy%L1;
         iy/=L1;
         x[0]=iy;

         dsq=nrmsq(i3d,x);
         iy=ipt[ix];

         if (dsq<=rsq)
            f[iy]=1.0;
         else
            f[iy]=0.0;
      }
   }
}


void sphere_msfld(int r,double *f)
{
   chk_parms(r,"sphere_msfld [latavg.c]");
   set_sphere_msfld(0,r,f);
}


void sphere3d_msfld(int r,double *f)
{
   chk_parms(r,"sphere3d_msfld [latavg.c]");
   set_sphere_msfld(1,r,f);
}


static void set_sphere_sum(int i3d,int dmax,double *f,double *sm)
{
   int k,ofs,vol,ix,iy,x[4];
   int d,dsq;
   double s;

   alloc_smx(dmax);

   for (d=0;d<=dmax;d++)
   {
      smx[d].q[0]=0.0;
      smx[d].q[1]=0.0;
   }

#pragma omp parallel private(k,ofs,vol,ix,iy,x,d,dsq,s) \
   reduction(sum_qflt : smx[0:(dmax+1)])
   {
      k=omp_get_thread_num();
      vol=VOLUME_TRD;
      ofs=k*vol;

      for (ix=ofs;ix<(ofs+vol);ix++)
      {
         iy=ix;
         x[3]=iy%L3;
         iy/=L3;
         x[2]=iy%L2;
         iy/=L2;
         x[1]=iy%L1;
         iy/=L1;
         x[0]=iy;

         dsq=nrmsq(i3d,x);
         s=f[ipt[ix]];

         for (d=dmax;(dsq<=(d*d))&&(d>=0);d--)
            acc_qflt(s,smx[d].q);
      }
   }

   if (NPROC>1)
      global_qsum(dmax+1,qsmx,qsmx);

   for (d=0;d<=dmax;d++)
      sm[d]=smx[d].q[0];
}


void sphere_sum(int dmax,double *f,double *sm)
{
   error_root(dmax<0,1,"sphere_sum [latavg.c]",
              "Parameter dmax is out of range");
   chk_parms(dmax,"sphere_sum [latavg.c]");
   set_sphere_sum(0,dmax,f,sm);
}


void sphere3d_sum(int dmax,double *f,double *sm)
{
   error_root(dmax<0,1,"sphere3d_sum [latavg.c]",
              "Parameter dmax is out of range");
   chk_parms(dmax,"sphere3d_sum [latavg.c]");
   set_sphere_sum(1,dmax,f,sm);
}


double avg_msfld(double *f)
{
   int k;
   double *fr,*fm;
   double sm,*qsm[1];
   qflt rqsm;

   rqsm.q[0]=0.0;
   rqsm.q[1]=0.0;

#pragma omp parallel private(k,fr,fm,sm) reduction(sum_qflt : rqsm)
   {
      k=omp_get_thread_num();

      fr=f+k*VOLUME_TRD;
      fm=fr+VOLUME_TRD;

      for (;fr<fm;fr+=4)
      {
         sm=fr[0]+fr[1]+fr[2]+fr[3];
         acc_qflt(sm,rqsm.q);
      }
   }

   if (NPROC>1)
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   return rqsm.q[0]/((double)(N0*N1)*(double)(N2*N3));
}


double center_msfld(double *f)
{
   int k;
   double *fr,*fm;
   double av;

   av=avg_msfld(f);

#pragma omp parallel private(k,fr,fm)
   {
      k=omp_get_thread_num();

      fr=f+k*VOLUME_TRD;
      fm=fr+VOLUME_TRD;

      for (;fr<fm;fr+=1)
         (*fr)-=av;
   }

   return av;
}


void cov_msfld(int dmax,double *f,double *g,
               complex_dble *rf,complex_dble *rg,double *w,double *cov)
{
   int k;
   double r;

   convolute_msfld(NULL,f,g,rf,rg,w);
   sphere_sum(dmax,w,cov);
   r=((double)(N0*N1)*(double)(N2*N3));
   r=1.0/(r*r);

   for (k=0;k<=dmax;k++)
      cov[k]*=r;
}


void cov3d_msfld(int dmax,double *f,double *g,
                 complex_dble *rf,complex_dble *rg,double *w,double *cov)
{
   int k;
   double r;

   convolute_msfld(NULL,f,g,rf,rg,w);
   sphere3d_sum(dmax,w,cov);
   r=((double)(N0*N1)*(double)(N2*N3));
   r=1.0/(r*r);

   for (k=0;k<=dmax;k++)
      cov[k]*=r;
}
