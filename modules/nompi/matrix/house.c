
/*******************************************************************************
*
* File house.c
*
* Copyright (C) 2008 Martin Luescher
*
* Householder inversion of a real matrix.
*
*   double house(int n,double *amat)
*     This program computes the inverse of the real nxn matrix amat, using
*     Householder rotations, and assigns the computed matrix to amat. The
*     return value is the Frobenius condition number of amat.
*
* The (i,j)-element of the matrix amat (i,j=0,..,n-1) is assumed to be stored
* in the array element amat[i*n+j]. The inverse ainv thus satisfies
*
*    sum_k ainv[i*n+k]*amat[k*n+j] = delta_{ij}
*
* where the sum over k runs from 0 to n-1.
*
*******************************************************************************/

#define HOUSE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "utils.h"
#include "matrix.h"

static int nsv=0;
static double *dd,*rr;


static void alloc_arrays(int n)
{
   if (n>nsv)
   {
      if (nsv!=0)
         afree(dd);

      dd=amalloc(2*n*sizeof(*dd),4);
      rr=dd+n;

      error(dd==NULL,1,"alloc_arrays [house.c]",
            "Unable to allocate auxiliary arrays");

      nsv=n;
   }
}


static int check_sym(int n,double *amat)
{
   int i,j;

   for (i=0;i<n;i++)
   {
      for (j=0;j<i;j++)
      {
         if (amat[i*n+j]!=amat[j*n+i])
            return 0;
      }
   }

   return 1;
}


static void make_sym(int n,double *amat)
{
   int i,j;

   for (i=0;i<n;i++)
   {
      for (j=0;j<i;j++)
         amat[i*n+j]=amat[j*n+i];
   }
}


static double fnorm_square(int n,double *amat)
{
   int i;
   double sm;

   sm=0.0;

   for (i=0;i<(n*n);i++)
      sm+=amat[i]*amat[i];

   return sm;
}


static void fwd_house(int n,double *amat,double eps)
{
   int i,j,k,itest;
   double r1,r2,r3,r4,sm;

   itest=0;

   for (k=0;k<(n-1);k++)
   {
      r1=0.0;

      for (j=k;j<n;j++)
         r1+=amat[j*n+k]*amat[j*n+k];

      r1=sqrt(r1);

      if (r1<eps)
      {
         itest=1;
         r1=1.0;
      }

      r2=fabs(amat[k*n+k]);
      r4=1.0;

      if ((amat[k*n+k]<0.0)&&(r2>=(DBL_EPSILON*r1)))
         r4=-1.0;

      amat[k*n+k]+=r1*r4;
      r3=1.0/(r1*(r1+r2));
      rr[k]=r3;
      dd[k]=-(r1+r2)*r3*r4;

      for (j=(k+1);j<n;j++)
      {
         sm=0.0;

         for (i=k;i<n;i++)
            sm+=amat[i*n+k]*amat[i*n+j];

         sm*=r3;

         for (i=k;i<n;i++)
            amat[i*n+j]-=sm*amat[i*n+k];
      }
   }

   r1=amat[n*n-1];

   if (fabs(r1)>=eps)
      dd[n-1]=1.0/r1;
   else
      itest=1;

   error(itest!=0,1,"fwd_house [house.c]","Matrix is not invertible");
}


static void solv_sys(int n,double *amat)
{
   int i,j,k;
   double sm;

   for (k=(n-1);k>0;k--)
   {
      for (i=(k-1);i>=0;i--)
      {
         sm=amat[i*n+k]*dd[k];

         for (j=(k-1);j>i;j--)
            sm+=amat[i*n+j]*amat[j*n+k];

         amat[i*n+k]=-dd[i]*sm;
      }
   }
}


static void bck_house(int n,double *amat)
{
   int i,j,k;
   double r,sm;

   amat[n*n-1]=dd[n-1];

   for (k=(n-2);k>=0;k--)
   {
      r=dd[k];
      dd[k]=amat[k*n+k];
      amat[k*n+k]=r;

      for (j=(k+1);j<n;j++)
      {
         dd[j]=amat[j*n+k];
         amat[j*n+k]=0.0;
      }

      for (i=0;i<n;i++)
      {
         sm=0.0;

         for (j=k;j<n;j++)
            sm+=amat[i*n+j]*dd[j];

         sm*=rr[k];

         for (j=k;j<n;j++)
            amat[i*n+j]-=sm*dd[j];
      }
   }
}


double house(int n,double *amat)
{
   int isym;
   double fn1,fn2,eps;

   error(n<1,1,"house [house.c]","Parameter n is out of range");

   alloc_arrays(n);
   isym=check_sym(n,amat);
   fn1=fnorm_square(n,amat);
   eps=DBL_EPSILON*sqrt((double)(n)*fn1);

   fwd_house(n,amat,eps);
   solv_sys(n,amat);
   bck_house(n,amat);

   if (isym==1)
      make_sym(n,amat);
   fn2=fnorm_square(n,amat);

   return sqrt(fn1*fn2);
}
