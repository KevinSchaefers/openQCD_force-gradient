
/*******************************************************************************
*
* File jacobi.c
*
* Copyright (C) 1998, 2011, 2016 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Diagonalization of a real symmetric square matrix using the Jacobi method.
*
*   void jacobi(int n,double *a,double *d,double *v)
*     Computes the eigenvalues and eigenvectors of the nxn matrix a.
*
* See chapter 11 in
*
*   W.H. Press, S.A. Teukolsky, W.T. Vetterling and B.P. Flannery,
*   Numerical Recipes in FORTRAN, 2nd Edition
*   (Cambridge University Press, Cambridge, 1992)
*
* for a description of the algorithm.
*
* The matrix "a" is expected in the form of the linear array a[n*i+j] of its
* i,j elements, where the indices range from 0 to n-1. On exit the eigenvalues
* d[j] and associated orthonormal eigenvectors v[n*i+j] are such that
*
*   sum_k a[n*i+k]*v[n*k+j]=d[j]*v[n*i+j].
*
* The eigenvalues are sorted in ascending order.
*
*******************************************************************************/

#define JACOBI_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "matrix.h"

#define MAX_SWEEP 100


static void sort(int n,double *d,double *v)
{
   int i,j,k;
   double p;

   for (i=0;i<n-1;i++)
   {
      k=i;
      p=d[i];

      for (j=i+1;j<n;j++)
      {
	 if (d[j]<p)
	 {
	    k=j;
	    p=d[j];
	 }
      }

      if (k!=i)
      {
	 d[k]=d[i];
	 d[i]=p;

	 for (j=0;j<n;j++)
	 {
	    p=v[n*j+i];
	    v[n*j+i]=v[n*j+k];
	    v[n*j+k]=p;
	 }
      }
   }
}


void jacobi(int n,double *a,double *d,double *v)
{
   int k,l,j,sweep;
   double abs_sum,thresh_factor,sd_factor,thresh;
   double r1,r2,r3,r4;
   double t,e,s,c,tau,xn;

   error(n<=0,1,"jacobi [jacobi.c]","Argument n is out of range");

   xn=(double)(n);
   sd_factor=100.0;
   thresh_factor=0.2/(xn*xn);
   abs_sum=0.0;

   for (k=0;k<n;k++)
   {
      v[n*k+k]=1.0;
      d[k]=a[n*k+k];

      for (l=k+1;l<n;l++)
      {
	 v[n*k+l]=0.0;
	 v[n*l+k]=0.0;

	 error(a[n*k+l]!=a[n*l+k],1,"jacobi [jacobi.c]",
               "Matrix is not symmetric");

	 abs_sum+=fabs(a[n*k+l]);
      }
   }

   for (sweep=0;abs_sum>0.0;sweep++)
   {
      error(sweep==MAX_SWEEP,1,"jacobi [jacobi.c]",
            "Maximum number of sweeps exceeded");

      thresh=0.0;

      if (sweep<=2)
	 thresh=thresh_factor*abs_sum;

      for (k=0;k<n-1;k++)
      {
	 for (l=k+1;l<n;l++)
	 {
	    r1=sd_factor*fabs(a[n*k+l]);
	    r2=fabs(d[k]);
	    r3=fabs(d[l]);

	    if ((sweep>3)&&(r2==(r2+r1))&&(r3==(r3+r1)))
	       a[n*k+l]=0.0;

	    r1=fabs(a[n*k+l]);
	    if (r1<=thresh)
	       continue;

	    r2=d[l]-d[k];
	    r3=fabs(r2);

            if (r3==sd_factor*r1+r3)
	       t=r1/r2;
	    else
	    {
	       r4=0.5*r2/r1;

	       if (r4<0.0)
	       {
		  t=1.0/(r4-sqrt(1.0+r4*r4));
	       }
	       else
	       {
		  t=1.0/(r4+sqrt(1.0+r4*r4));
	       }
	    }

            e=1.0;
	    if (a[n*k+l]<0.0)
	       e=-1.0;
	    a[n*k+l]=0.0;

	    c=1.0/sqrt(1.0+t*t);
	    s=t*c;
	    tau=s/(1.0+c);

	    r2=t*r1;
	    d[k]-=r2;
	    d[l]+=r2;

	    for (j=0;j<n;j++)
	    {
	       if (j<k)
	       {
		  r1=a[n*j+k];
		  r2=a[n*j+l];
		  a[n*j+k]=-s*( tau*r1+e*r2)+r1;
		  a[n*j+l]= s*(-tau*r2+e*r1)+r2;
	       }
	       else if ((j>k)&&(j<l))
	       {
		  r1=a[n*k+j];
		  r2=a[n*j+l];
		  a[n*k+j]=-s*( tau*r1+e*r2)+r1;
		  a[n*j+l]= s*(-tau*r2+e*r1)+r2;
	       }
	       else if (j>l)
	       {
		  r1=a[n*k+j];
		  r2=a[n*l+j];
		  a[n*k+j]=-s*( tau*r1+e*r2)+r1;
		  a[n*l+j]= s*(-tau*r2+e*r1)+r2;
	       }

	       r1=v[n*j+k];
	       r2=v[n*j+l];
	       v[n*j+k]=-s*( tau*r1+e*r2)+r1;
	       v[n*j+l]= s*(-tau*r2+e*r1)+r2;
	    }
	 }
      }

      abs_sum=0.0;

      for (k=0;k<n-1;k++)
      {
	 for (l=k+1;l<n;l++)
	    abs_sum+=fabs(a[n*k+l]);
      }
   }

   for (k=0;k<n-1;k++)
   {
      for (l=k+1;l<n;l++)
	 a[n*k+l]=a[n*l+k];
   }

   sort(n,d,v);
}
