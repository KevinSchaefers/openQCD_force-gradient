
/*******************************************************************************
*
* File polyfit.c
*
* Copyright (C) 2016 Martin Luescher
*
* Least-squares polynomial fit of uncorrelated data.
*
*   double polyfit(fit_t dat,int pmin,int pmax,double *a,double *cov)
*     Fits the data "dat" by a polynomial a[0]*x^pmin+..+a[m-1]*x^pmax,
*     with m=pmax-pmin+1 unknown coefficients. The program returns the
*     chi^2 of the fit and assigns the covariance of the fitted values
*     to the matrix cov[i*m+j], i,j=0,..,m.
*
* The data to be fitted are passed to this program through a structure of
* type fit_t, defined in analysis.h, whose elements are
*
*  n           Number of data points.
*
*  x           Array x[0],..,x[n-1] of values of the independent variable.
*
*  y           Array y[0],..,y[n-1] of values to be fitted.
*
*  c           Statistical errors c[0],..,c[n-1] of the values
*              y[0],..,y[n-1].
*
* The fit is performed using the programs in the module lsqfit.c.
*
*******************************************************************************/

#define POLYFIT_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "analysis.h"

static int ns=0,ms=0,pms;
static double *sa;
static fit_t data;


static void alloc_arrays(int n,int m)
{
   if (ns>0)
      free(sa);

   sa=malloc(n*(n+m+2)*sizeof(*sa));
   error(sa==NULL,1,"alloc_arrays [polyfit.c]",
         "Unable to allocate data arrays");

   data.x=sa+n*m;
   data.y=data.x+n;
   data.c=data.y+n;

   ns=n;
   ms=m;
}


static double xpow(int i,double x)
{
   i+=pms;

   if (i==0)
      return 1.0;
   else if (i==1)
      return x;
   else
      return pow(x,(double)(i));
}


static double fit_data(int n,int m,double *a,double *cov)
{
   double chisq;

   chisq=least_squares(data,m,0,xpow,a,sa);
   fit_parms(data,m,sa,a,cov);

   return chisq;
}


double polyfit(fit_t dat,int pmin,int pmax,double *a,double *cov)
{
   int n,m,i,j;

   n=dat.n;
   m=pmax-pmin+1;

   error(m<1,1,"polyfit [polyfit.c]","Improper choice of paramters pmin,pmax");
   error(n<m,1,"polyfit [polyfit.c]","Insufficient number of data points");

   if ((n>ns)||(m>ms))
      alloc_arrays(n,m);

   pms=pmin;
   data.n=n;

   for (i=0;i<n;i++)
   {
      data.x[i]=dat.x[i];
      data.y[i]=dat.y[i];

      for (j=0;j<n;j++)
      {
         if (j!=i)
            data.c[i*n+j]=0.0;
         else
            data.c[i*n+j]=dat.c[i]*dat.c[i];
      }
   }

   return fit_data(n,m,a,cov);
}
