
/*******************************************************************************
*
* File valg.c
*
* Copyright (C) 2007, 2011, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic linear algebra routines for single-precision complex fields.
*
*   complex_dble vprod(int n,int icom,complex *v,complex *w)
*     Returns the scalar product of the n-vectors v and w.
*
*   double vnorm_square(int n,int icom,complex *v)
*     returns the square of the norm of the n-vector v.
*
*   void mulc_vadd(int n,int icom,complex *v,complex *w,complex z)
*     Replaces the n-vector v by v+z*w.
*
*   void vscale(int n,int icom,float r,complex_dble *v)
*     Replaces the n-vector v by r*v.
*
*   void vproject(int n,int icom,complex *v,complex *w)
*     Replaces the n-vector v by v-(w,v)*w.
*
*   double vnormalize(int n,int icom,complex *v)
*     Replaces the n-vector v by v/||v|| and returns the 2-norm ||v||.
*     The division is omitted if the norm vanishes and an error occurs
*     if icom!=0 (if icom=0 it is up to the calling program to check
*     that the norm was non-zero).
*
* All these programs operate on complex n-vectors whose base addresses are
* passed through the arguments. The length n of the arrays is specified by
* the parameter n and the meaning of the parameter icom is explained in
* the file sflds/README.icom.
*
*******************************************************************************/

#define VALG_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "utils.h"
#include "linalg.h"
#include "vflds.h"
#include "global.h"


static complex_dble loc_vprod(int n,complex *v,complex *w)
{
   complex *vm;
   complex_dble cv;

   cv.re=0.0;
   cv.im=0.0;
   vm=v+n;

   for (;v<vm;v++)
   {
         cv.re+=(double)((*v).re*(*w).re+(*v).im*(*w).im);
         cv.im+=(double)((*v).re*(*w).im-(*v).im*(*w).re);
         w+=1;
   }

   return cv;
}


static double loc_vnorm_square(int n,complex *v)
{
   complex *vm;
   double x;

   x=0.0;
   vm=v+n;

   for (;v<vm;v++)
      x+=(double)((*v).re*(*v).re+(*v).im*(*v).im);

   return x;
}


static void loc_mulc_vadd(int n,complex *v,complex *w,complex z)
{
   complex *vm;

   vm=v+n;

   for (;v<vm;v++)
   {
      (*v).re+=(z.re*(*w).re-z.im*(*w).im);
      (*v).im+=(z.re*(*w).im+z.im*(*w).re);
      w+=1;
   }
}


static void loc_vscale(int n,float r,complex *v)
{
   complex *vm;

   vm=v+n;

   for (;v<vm;v++)
   {
      (*v).re*=r;
      (*v).im*=r;
   }
}

complex_dble vprod(int n,int icom,complex *v,complex *w)
{
   int k;
   complex_dble cv,cw;

   if ((icom&0x2)==0)
      cv=loc_vprod(n,v,w);
   else
   {
      cv.re=0.0;
      cv.im=0.0;

#pragma omp parallel private(k) reduction(sum_complex_dble : cv)
      {
         k=omp_get_thread_num();
         cv=loc_vprod(n,v+k*n,w+k*n);
      }
   }

   if ((NPROC>1)&&(icom&0x1))
   {
      cw.re=cv.re;
      cw.im=cv.im;

      MPI_Reduce(&cw.re,&cv.re,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&cv.re,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }

   return cv;
}


double vnorm_square(int n,int icom,complex *v)
{
   int k;
   double x,y;

   if ((icom&0x2)==0)
      x=loc_vnorm_square(n,v);
   else
   {
      x=0.0;

#pragma omp parallel private(k) reduction(+ : x)
      {
         k=omp_get_thread_num();
         x=loc_vnorm_square(n,v+k*n);
      }
   }

   if ((NPROC>1)&&(icom&0x1))
   {
      y=x;

      MPI_Reduce(&y,&x,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&x,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }

   return x;
}


void mulc_vadd(int n,int icom,complex *v,complex *w,complex z)
{
   int k;

   if ((icom&0x2)==0)
      loc_mulc_vadd(n,v,w,z);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_mulc_vadd(n,v+k*n,w+k*n,z);
      }
   }
}


void vscale(int n,int icom,float r,complex *v)
{
   int k;

   if ((icom&0x2)==0)
      loc_vscale(n,r,v);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_vscale(n,r,v+k*n);
      }
   }
}


void vproject(int n,int icom,complex *v,complex *w)
{
   complex z;
   complex_dble zd;

   zd=vprod(n,icom,w,v);
   z.re=-(float)(zd.re);
   z.im=-(float)(zd.im);
   mulc_vadd(n,icom,v,w,z);
}


double vnormalize(int n,int icom,complex *v)
{
   double rd;

   rd=vnorm_square(n,icom,v);
   rd=sqrt(rd);

   if (rd!=0.0)
      vscale(n,icom,(float)(1.0/rd),v);
   else if (icom!=0)
      error_loc(1,1,"vnormalize [valg.c]",
                "Vector field has vanishing norm");

   return rd;
}
