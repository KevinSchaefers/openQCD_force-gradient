
/*******************************************************************************
*
* File bootstrap.c
*
* Copyright (C) 2017 Martin Luescher
*
* Resampling programs.
*
*   int sample_gauss(int n,double *avg,double *cov,int nms,int ndc,
*                    double *ev,jdat_t **jdat)
*     Generates a representative sample of nms Gaussian n-component vectors
*     with mean avg[i] and variance cov[i*n+j] (i,j=0,..,n-1). The jackknife
*     samples jdat[i] of the components v[i] are then formed with bin size 1
*     and decimation level up to ndc. The program assigns the eigenvalues of
*     the covariance matrix to ev[0],..,ev[n-1] and returns the number nev
*     of negative eigenvalues. If nev>0 the covariance matrix is truncated
*     to its positive part before generating the jackknife samples.
*
*******************************************************************************/

#define BOOTSTRAP_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "utils.h"
#include "random.h"
#include "matrix.h"
#include "analysis.h"

static int ns=0,nmx=0;
static double *wmat,*wvec,*cmat;
static double **ms;


static void alloc_wmat(int n,int nms)
{
   int i;
   double *p;

   if ((n*nms)>nmx)
   {
      if (nmx>0)
         free(ms[0]);

      nmx=n*nms;
      p=malloc(nmx*sizeof(*p));
      error(p==NULL,1,"alloc_wmat [bootstrap.c]",
            "Unable to allocate auxiliary arrays");
   }
   else
      p=ms[0];

   if (n>ns)
   {
      if (ns>0)
      {
         free(wmat);
         free(ms);
      }

      wmat=malloc((n+2*n*n)*sizeof(*wmat));
      ms=malloc(n*sizeof(*ms));
      error((wmat==NULL)||(ms==NULL),1,"alloc_wmat [bootstrap.c]",
            "Unable to allocate auxiliary arrays");
      ns=n;
   }

   wvec=wmat+n*n;
   cmat=wvec+n;

   for (i=0;i<(n+2*n*n);i++)
      wmat[i]=0.0;

   for (i=0;i<(n*nms);i++)
      p[i]=0.0;

   for (i=0;i<n;i++)
   {
      ms[i]=p;
      p+=nms;
   }
}


static int set_cmat(int n,double *cov,double *ev)
{
   int i,j,nev;

   jacobi(n,cov,ev,wmat);
   nev=0;

   for (i=0;i<n;i++)
   {
      if (ev[i]<0.0)
      {
         nev+=1;
         wvec[i]=0.0;
      }
      else
         wvec[i]=sqrt(2.0*ev[i]);
   }

   for (i=0;i<n;i++)
   {
      for (j=0;j<n;j++)
         cmat[i*n+j]=wmat[i*n+j]*wvec[j];
   }

   return nev;
}


int sample_gauss(int n,double *avg,double *cov,int nms,int ndc,
                 double *ev,jdat_t **jdat)
{
   int i,j,k,nev;
   double sm;

   error((n<1)||(nms<1)||(ndc<1),1,"sample_gauss [bootstrap.c]",
         "Parameter n,nms or ndc is out of range");

   alloc_wmat(n,nms);
   nev=set_cmat(n,cov,ev);

   for (i=0;i<nms;i++)
   {
      gauss_dble(wvec,n);

      for (j=0;j<n;j++)
      {
         sm=0.0;

         for (k=0;k<n;k++)
            sm+=cmat[n*j+k]*wvec[k];

         ms[j][i]=avg[j]+sm;
      }
   }

   for (j=0;j<n;j++)
      jackbin(nms,1,ndc,ms[j],jdat[j]);

   return nev;
}
