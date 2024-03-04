
/*******************************************************************************
*
* File matrix.c
*
* Copyright (C) 2008, 2011, 2020 Martin Luescher, Leonardo Giusti
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Matrix operations.
*
*   void mat_vec(int n,double a[],double v[],double w[])
*     Computes w=a*v, where v and w are n-vectors and a an nxn matrix.
*
*   void mat_add(int n,double a[],double b[],double c[])
*     Computes the sum c=a+b of two nxn matrices.
*
*   void mat_sub(int n,double a[],double b[],double c[])
*     Computes the difference c=a-b of two nxn matrices.
*
*   void mat_mulr(int n,double r,double a[])
*     Multiplies the nxn matrix a by r.
*
*   void mat_mulr_add(int n,double r,double a[],double b[])
*     Adds the product r*a to b, where a and b are nxn matrices.
*
*   void mat_mul(int n,double a[],double b[],double c[])
*     Computes the product c=a*b of two nxn matrices.
*
*   void mat_mul_tr(int n,double a[],double b[],double c[])
*     Computes the product c=a*(b^t) of two nxn matrices.
*
*   void mat_sym(int n,double a[])
*     Symmetrizes the nxn matrix a[] by assigning the upper triangular
*     submatrix to the lower one.
*
*   void mat_inv(int n,double a[],double ainv[],double *k)
*     Computes the inverse ainv[] of the real symmetric nxn matrix a[]
*     and its condition number k.
*
*   double mat_trace(int n,double a[])
*     Returns the trace of the real symmetric nxn matrix a[].
*
* The convention is adopted here that the matrix element a_{ij} of a given
* nxn matrix a is stored in the element a[i*n+j] of the associated array
* where i,j=0,1,..,n-1
*
*******************************************************************************/

#define MATRIX_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "utils.h"
#include "matrix.h"


void mat_vec(int n,double a[],double v[],double w[])
{
   int i,j;
   double sm;

   for (i=0;i<n;i++)
   {
      sm=0.0;

      for (j=0;j<n;j++)
         sm+=a[i*n+j]*v[j];

      w[i]=sm;
   }
}


void mat_add(int n,double a[],double b[],double c[])
{
   int i;

   for (i=0;i<(n*n);i++)
      c[i]=a[i]+b[i];
}


void mat_sub(int n,double a[],double b[],double c[])
{
   int i;

   for (i=0;i<(n*n);i++)
      c[i]=a[i]-b[i];
}


void mat_mulr(int n,double r,double a[])
{
   int i;

   for (i=0;i<(n*n);i++)
      a[i]*=r;
}


void mat_mulr_add(int n,double r,double a[],double b[])
{
   int i;

   for (i=0;i<(n*n);i++)
      b[i]+=r*a[i];
}


void mat_mul(int n,double a[],double b[],double c[])
{
   int i,j,l;
   double sm;

   for (i=0;i<n;i++)
   {
      for (j=0;j<n;j++)
      {
         sm=0.0;

         for (l=0;l<n;l++)
            sm+=a[i*n+l]*b[l*n+j];

         c[i*n+j]=sm;
      }
   }
}


void mat_mul_tr(int n,double a[],double b[],double c[])
{
   int i,j,l;
   double sm;

   for (i=0;i<n;i++)
   {
      for (j=0;j<n;j++)
      {
         sm=0.0;

         for (l=0;l<n;l++)
            sm+=a[i*n+l]*b[j*n+l];

         c[i*n+j]=sm;
      }
   }
}


void mat_sym(int n,double a[])
{
   int i,j;

   for (i=0;i<n;i++)
   {
      for (j=0;j<i;j++)
         a[i*n+j]=a[j*n+i];
   }
}


void mat_inv(int n,double a[],double ainv[],double *k)
{
   int i,j,l;
   double *d,*v,sm,dmin,dmax,dabs;

   d=malloc(n*sizeof(double));
   v=malloc(n*n*sizeof(double));

   error((d==NULL)||(v==NULL),1,"mat_inv [matrix.c]",
         "Unable to allocate workspace");

   jacobi(n,a,d,v);

   dmin=fabs(d[0]);
   dmax=dmin;

   for (i=1;i<n;i++)
   {
      dabs=fabs(d[i]);

      if (dabs<dmin)
         dmin=dabs;
      if (dabs>dmax)
         dmax=dabs;
   }

   error(dmin<=(DBL_EPSILON*dmax),1,"mat_inv [matrix.c]",
         "Singular matrix");

   *k=dmax/dmin;

   for (i=0;i<n;i++)
      d[i]=1.0/d[i];

   for (i=0;i<n;i++)
   {
      for (j=0;j<=i;j++)
      {
         sm=0.0;

         for (l=0;l<n;l++)
            sm+=v[i*n+l]*d[l]*v[j*n+l];

         ainv[i*n+j]=sm;
         ainv[j*n+i]=sm;
      }
   }

   free(d);
   free(v);
}

double mat_trace(int n,double a[])
{
  int i;
  double tr;

  tr=0.0;
  for (i=0;i<n;i++)
    tr+=a[i*n+i];

  return (tr);
}
