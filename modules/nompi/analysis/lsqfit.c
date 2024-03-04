
/*******************************************************************************
*
* File lsqfit.c
*
* Copyright (C) 2006, 2008, 2016 Martin Luescher
*
* General least-squares fit program.
*
*   double least_squares(fit_t dat,int m,int r,double (*f)(int i,double x),
*                        double *w,double *s)
*     Fit of the data "dat" to the function f(x)=sum_i a[i]*f(i,x), where
*     a[i], i=0,..,m-1, are the fit parameters. The r most shallow modes of
*     chi^2 are dropped from the fit. The program returns the chi^2 of the
*     fit, the singular values w[0]<=w[1]<=..<=w[m-1] of the design matrix
*     and the matrix s such that a[i]=sum_j s[i*n+j]*y[j] (see the notes).
*
*   void fit_parms(fit_t dat,int m,double *s,double *val,double *cov)
*     Computes the fit parameters val[i], i=0,..,m-1, and the associated
*     covariance matrix cov[i*m+j], given the data to be fitted and the
*     matrix s determined by the program least_squares().
*
* The data to be fitted are passed to these programs through a structure of
* type fit_t, defined in analysis.h, whose elements are
*
*  n           Number of data points.
*
*  x           Array x[0],..,x[n-1] of values of the independent variable.
*
*  y           Array y[0],..,y[n-1] of values to be fitted.
*
*  c           Covariance matrix c[i*n+j]=<dy[i]*dy[j]> of the data to be
*              fitted.
*
* The covariance matrix must be symmetric and non-singular. Its inverse is
* calculated using Jacobi diagonalization. The matrix s computed by the
* program least_squares*() must be an array with at least m*n elements.
*
* An ordinary least-squares fit is performed if r=0. Otherwise chi^2 is
* minimized in the linear subspace of the parameter space orthogonal to
* the r lowest modes of the design matrix. See chapter 15 in
*
*   W.H. Press, S.A. Teukolsky, W.T. Vetterling and B.P. Flannery,
*   Numerical Recipes in FORTRAN, 2nd Edition
*   (Cambridge University Press, Cambridge, 1992)
*
* for further explanations.
*
* Least-squares fits require the solution of a potentially ill-conditioned
* linear system. Here the system is solved using Householder reflections,
* which avoids possibly large round-off errors.
*
*******************************************************************************/

#define LSQFIT_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "utils.h"
#include "matrix.h"
#include "analysis.h"


static void set_cmat(int n,double *c,double *wmat,double *wvec,
                     double *cmat)
{
   int i,j,l;
   double sm;

   jacobi(n,c,wvec,wmat);

   error((wvec[0]<0.0)||(wvec[0]<=((double)(n)*DBL_EPSILON*wvec[n-1])),
         1,"set_cmat [lsqfit.c]","Ill-conditioned covariance matrix");

   for (l=0;l<n;l++)
      wvec[l]=1.0/sqrt(wvec[l]);

   for (i=0;i<n;i++)
   {
      for (j=0;j<=i;j++)
      {
         sm=0.0;

         for (l=0;l<n;l++)
            sm+=wmat[i*n+l]*wvec[l]*wmat[j*n+l];

         cmat[i*n+j]=sm;
         cmat[j*n+i]=sm;
      }
   }
}


static void set_amat(int n,int m,double *x,double (*f)(int i,double x),
                     double *cmat,double *wmat,double *amat)
{
   int i,j,l;
   double sm;

   for (i=0;i<n;i++)
   {
      for (j=0;j<m;j++)
         wmat[i*m+j]=f(j,x[i]);
   }

   for (i=0;i<n;i++)
   {
      for (j=0;j<m;j++)
      {
         sm=0.0;

         for (l=0;l<n;l++)
            sm+=cmat[i*n+l]*wmat[l*m+j];

         amat[i*m+j]=sm;
      }
   }
}


static void set_vmat(int n,int m,double *amat,double *wmat,
                     double *vmat,double *w)
{
   int k,l,j;
   double sm;

   for (k=0;k<m;k++)
   {
      for (l=k;l<m;l++)
      {
         sm=0.0;

         for (j=0;j<n;j++)
            sm+=amat[j*m+k]*amat[j*m+l];

         wmat[k*m+l]=sm;
         wmat[l*m+k]=sm;
      }
   }

   jacobi(m,wmat,w,vmat);

   error((w[0]<0.0)||(w[0]<=((double)(m)*DBL_EPSILON*w[m-1])),1,
         "set_vmat [lsqfit.c]","Ill-conditioned design matrix");

   for (k=0;k<m;k++)
      w[k]=sqrt(w[k]);
}


static void set_tmat(int n,int m,double *amat,double *rmat,double *tmat)
{
   int k,l,j;
   double dmax,dmin,rho;
   double r,sm,*u;

   dmax=0.0;
   dmin=0.0;

   for (k=0;k<(n*m);k++)
      tmat[k]=amat[k];

   for (k=0;k<m;k++)
   {
      u=rmat+k*n;

      for (l=0;l<k;l++)
         u[l]=0.0;

      r=0.0;

      for (l=k;l<n;l++)
      {
         u[l]=tmat[l*m+k];
         tmat[l*m+k]=0.0;
         r+=u[l]*u[l];
      }

      r=sqrt(r);

      if (r>dmax)
         dmax=r;
      if ((r<dmin)||(k==0))
         dmin=r;

      error(dmin<=(1.0e2*(double)(n)*DBL_EPSILON*dmax),1,"set_tmat [lsqfit.c]",
            "Singular least-squares problem");

      if (fabs(u[k])<(DBL_EPSILON*r))
      {
         u[k]=r;
         tmat[k*m+k]=-r;
      }
      else if (u[k]>0.0)
      {
         u[k]+=r;
         tmat[k*m+k]=-r;
      }
      else
      {
         u[k]-=r;
         tmat[k*m+k]=r;
      }

      rho=1.0/(r*fabs(u[k]));

      for (j=(k+1);j<m;j++)
      {
         sm=0.0;

         for (l=k;l<n;l++)
            sm+=u[l]*tmat[l*m+j];

         sm*=rho;

         for (l=k;l<n;l++)
            tmat[l*m+j]-=sm*u[l];
      }
   }
}


static void set_tinv(int m,double *tmat,double *tinv)
{
   int k,i,j;
   double sm;

   for (k=0;k<m;k++)
   {
      tinv[k*m+k]=1.0/tmat[k*m+k];

      for (i=(k+1);i<m;i++)
         tinv[i*m+k]=0.0;
   }

   for (k=(m-1);k>=0;k--)
   {
      for (i=(k-1);i>=0;i--)
      {
         sm=0.0;

         for (j=(i+1);j<=k;j++)
            sm+=tmat[i*m+j]*tinv[j*m+k];

         tinv[i*m+k]=-sm*tinv[i*m+i];
      }
   }
}


static void set_smat(int n,int m,double *cmat,double *rmat,double *tinv,
                     double *wmat,double *s)
{
   int k,l,j;
   double nrm,sm,*u;

   for (k=0;k<(n*n);k++)
      wmat[k]=cmat[k];

   for (k=0;k<m;k++)
   {
      u=rmat+k*n;
      nrm=0.0;

      for (l=k;l<n;l++)
         nrm+=u[l]*u[l];

      for (j=0;j<n;j++)
      {
         sm=0.0;

         for (l=k;l<n;l++)
            sm+=u[l]*wmat[l*n+j];

         sm=2.0*sm/nrm;

         for (l=k;l<n;l++)
            wmat[l*n+j]-=sm*u[l];
      }
   }

   for (k=0;k<m;k++)
   {
      for (j=0;j<n;j++)
      {
         sm=0.0;

         for (l=0;l<m;l++)
            sm+=tinv[k*m+l]*wmat[l*n+j];

         s[k*n+j]=sm;
      }
   }
}


static void project_smat(int n,int m,int r,double *vmat,double *wmat,
                         double *wvec,double *s)
{
   int k,l,j;
   double sm;

   for (k=0;k<m;k++)
   {
      for (l=k;l<m;l++)
      {
         sm=0.0;

         for (j=r;j<m;j++)
            sm+=vmat[k*m+j]*vmat[l*m+j];

         wmat[k*m+l]=sm;
         wmat[l*m+k]=sm;
      }
   }

   for (j=0;j<n;j++)
   {
      for (k=0;k<m;k++)
      {
         sm=0.0;

         for (l=0;l<m;l++)
            sm+=wmat[k*m+l]*s[l*n+j];

         wvec[k]=sm;
      }

      for (k=0;k<m;k++)
         s[k*n+j]=wvec[k];
   }
}


double least_squares(fit_t dat,int m,int r,double (*f)(int i,double x),
                     double *w,double *s)
{
   int n,k,l;
   double *x,*y,*c;
   double *cmat,*wmat,*wvec;
   double *amat,*vmat,*rmat,*tmat,*tinv;
   double sm,chsq;

   n=dat.n;
   x=dat.x;
   y=dat.y;
   c=dat.c;

   error(n<m,1,"least_squares [lsqfit.c]",
         "Number of data points is too small");

   error((r<0)||(r>=m),1,"least_squares [lsqfit.c]",
         "Parameter r is out of range");

   cmat=malloc((2*n*n+n+3*n*m+2*m*m)*sizeof(*cmat));
   error(cmat==NULL,1,"least_squares [lsqfit.c]",
         "Unable to allocate workspace");

   wmat=cmat+n*n;
   vmat=wmat+n*n;
   wvec=vmat+m*m;
   amat=wvec+n;
   rmat=amat+n*m;
   tmat=rmat+n*m;
   tinv=tmat+n*m;

   set_cmat(n,c,wmat,wvec,cmat);
   set_amat(n,m,x,f,cmat,wmat,amat);
   set_vmat(n,m,amat,wmat,vmat,w);

   set_tmat(n,m,amat,rmat,tmat);
   set_tinv(m,tmat,tinv);
   set_smat(n,m,cmat,rmat,tinv,wmat,s);

   if (r>0)
      project_smat(n,m,r,vmat,wmat,wvec,s);

   for (k=0;k<m;k++)
   {
      sm=0.0;

      for (l=0;l<n;l++)
         sm+=s[k*n+l]*y[l];

      wvec[k]=sm;
   }

   chsq=0.0;

   for (k=0;k<n;k++)
   {
      sm=0.0;

      for (l=0;l<m;l++)
         sm+=amat[k*m+l]*wvec[l];

      for (l=0;l<n;l++)
         sm-=cmat[k*n+l]*y[l];

      chsq+=sm*sm;
   }

   free(cmat);

   return chsq;
}


void fit_parms(fit_t dat,int m,double *s,double *val,double *cov)
{
   int n,k,l,j;
   double *y,*c,*sc,sm;

   n=dat.n;
   y=dat.y;
   c=dat.c;

   sc=malloc(n*m*sizeof(*sc));
   error(sc==NULL,1,"fit_parms [lsqfit.c]",
         "Unable to allocate work space");

   for (k=0;k<m;k++)
   {
      sm=0.0;

      for (l=0;l<n;l++)
         sm+=s[k*n+l]*y[l];

      val[k]=sm;
   }

   for (k=0;k<m;k++)
   {
      for (j=0;j<n;j++)
      {
         sm=0.0;

         for (l=0;l<n;l++)
            sm+=s[k*n+l]*c[l*n+j];

         sc[k*n+j]=sm;
      }
   }

   for (k=0;k<m;k++)
   {
      for (j=0;j<=k;j++)
      {
         sm=0.0;

         for (l=0;l<n;l++)
            sm+=sc[k*n+l]*s[j*n+l];

         cov[k*m+j]=sm;
         cov[j*m+k]=sm;
      }
   }

   free(sc);
}
