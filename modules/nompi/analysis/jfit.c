
/*******************************************************************************
*
* File jfit.c
*
* Copyright (C) 2008, 2019 Martin Luescher
*
* Generic programs intended for fits of data available in the form of multi-
* elimination jackknife samples.
*
*   void jfit_cov(int ndat,jdat_t **jdat,jdat_t *jcov)
*     Computes the jackknife samples jcov[i*ndat+j] of the covariance matrix
*     of the data samples jdat[i], i,j=0,..,ndat-1. This program descends to
*     all jackknife elimination levels of the input data structures and
*     allocates or reallocates the output structures as required.
*
*   double jfit_icov(int ndat,jdat_t **jdat,jdat_t *jcov)
*     Computes the jackknife samples jcov[i*ndat+j] of the *inverse* of the
*     covariance matrix of the data samples jdat[i], i=0,..,ndat-1. This
*     program descends to all jackknife elimination levels of the input data
*     structures and allocates or reallocates the output structures as
*     required. On output the maximum of the Frobenius condition numbers of
*     the inverted matrices is returned.
*
*   double jfit_const(int type,int ndat,jdat_t **jdat,double *k,jdat_t *jpar)
*     Least-squares fit of the data jdat[0],..,jdat[ndat-1] by a constant.
*     The statistical fluctuations of the covariance matrix are taken into
*     account if type=1, while if type=0 the matrix is set to its average
*     value. On output k reports the maximal Frobenius condition number of
*     the covariance matrices used and the jackknife samples of the fitted
*     constant are assigned to jpar (which allocated or reallocated if so
*     required). All this is done to as many jackknife elimination levels
*     as possible. The program returns the chi^2 of the fit.
*
*   double jfit(int ndat,jdat_t **jdat,int npar,jdat_t **jpar,
*               double (*lhf)(int ndat,double *dat,int npar,double *par),
*               double *x0,double *x1,double *x2,
*               int imx,double omega1,double omega2,int *status)
*     Determines the parameters jpar[0],..,jpar[npar-1] by minimizing the
*     function lhf() of the statistical data jdat[0],..,jdat[ndat-1]. The
*     parameters x0,x1,...,status are passed to the program powell() in the
*     module extras/fsolve.c which is used for the minimization.
*      If the minimization was successful, the program returns the value of
*     lhf() at the minimum and status reports the average number of iterations
*     preformed by the minimization program. Otherwise 0 is returned, status
*     reports an error code and the fit parameters are set to the empty jdat_t
*     structure. This program descends to all jackknife elimination levels of
*     the input data structures.
*
* The program jfit() can be used to perform general fits, where a given
* likelihood function lhf() is to be minimized. This includes the case of
* linear and non-linear least-squares fits. For linear least-squares fits,
* the programs in the module analysis/lsqfit.c can alternatively be used.
*
* The program jfit_icov() calls jfit_cov() and then inverts the covariance
* matrices at each jackknife elimination level using Householder rotations
* (see nompi/matrix/house.c).
*
*******************************************************************************/

#define JFIT_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "extras.h"
#include "matrix.h"
#include "analysis.h"

static int nsv=0,ndsv=0,nxsv=0,isv,icnt;
static double *csv,*dsv,*xsv,*xmn;
static double (*fsv)(int ndat,double *dat,int npar,double *par);


static int nolev(jdat_t *jdat)
{
   int n;
   jdat_t *p;

   n=0;

   for (p=jdat;p!=NULL;p=(*p).next)
      n+=1;

   return n;
}


static double cov(jdat_t *jdat0,jdat_t *jdat1)
{
   int n,i;
   double fbar0,fbar1,*f0,*f1,sm;

   fbar0=(*jdat0).fbar;
   fbar1=(*jdat1).fbar;
   f0=(*jdat0).f;
   f1=(*jdat1).f;
   n=(*jdat0).n;
   sm=0.0;

   for (i=0;i<n;i++)
      sm+=((f0[i]-fbar0)*(f1[i]-fbar1));

   return sm;
}


static void jfit_cov_r(int ndc,jdat_t ***pdat,jdat_t ***pcov)
{
   int n,i;
   double r;
   jdat_t *jdat0,*jdat1,*jcov;

   jdat0=pdat[ndc-1][0];
   jdat1=pdat[ndc-1][1];
   jcov=pcov[ndc-1][0];

   (*jcov).fbar=cov(jdat0,jdat1);
   (*jcov).sigf=0.0;
   n=(*jdat0).n;

   for (i=0;i<n;i++)
   {
      (*jcov).f[i]=cov((*jdat0).next+i,(*jdat1).next+i);
      r=(*jcov).f[i]-(*jcov).fbar;
      (*jcov).sigf+=r*r;
   }

   (*jcov).sigf=sqrt((*jcov).sigf);

   if (ndc>2)
   {
      for (i=0;i<n;i++)
      {
         pdat[ndc-2][0]=(*jdat0).next+i;
         pdat[ndc-2][1]=(*jdat1).next+i;
         pcov[ndc-2][0]=(*jcov).next+i;

         jfit_cov_r(ndc-1,pdat,pcov);
      }
   }
}


void jfit_cov(int ndat,jdat_t **jdat,jdat_t *jcov)
{
   int i,j,k;
   int n,ndc;
   jdat_t ***pdat,***pcov,**p;

   error(ndat<1,1,"jfit_cov [jfit.c]",
         "Parameter ndat is out of range");

   k=0;

   for (i=0;i<ndat;i++)
   {
      j=jdat[i]-jcov;

      if ((j>=0)&&(j<(ndat*ndat)))
         k=1;
   }

   error(k!=0,1,"jfit_cov [jfit.c]",
         "Attempt to overwrite input data");

   check_jdat(ndat,jdat,1);
   n=(*(jdat[0])).n;
   ndc=nolev(jdat[0]);

   for (i=1;i<ndat;i++)
   {
      k=nolev(jdat[i]);

      if (k<ndc)
         ndc=k;
   }

   error(ndc<2,1,"jfit_cov [jfit.c]",
         "Insufficient jackknife elimination depth");

   for (i=0;i<(ndat*ndat);i++)
      alloc_jdat(n,ndc-1,jcov+i);

   pdat=amalloc(2*ndc*sizeof(*pdat),3);
   p=amalloc(3*ndc*sizeof(*p),3);
   error((pdat==NULL)||(p==NULL),1,"jfit_cov [jfit.c]",
         "Unable to allocate auxiliary arrays");

   pcov=pdat+ndc;
   pdat[0]=p;
   pcov[0]=p+2*ndc;

   for (i=1;i<ndc;i++)
   {
      pdat[i]=pdat[i-1]+2;
      pcov[i]=pcov[i-1]+1;
   }

   for (i=0;i<ndat;i++)
   {
      pdat[ndc-1][0]=jdat[i];

      for (j=i;j<ndat;j++)
      {
         pdat[ndc-1][1]=jdat[j];
         pcov[ndc-1][0]=jcov+i*ndat+j;

         jfit_cov_r(ndc,pdat,pcov);

         if (j>i)
            copy_jdat(jcov+i*ndat+j,jcov+j*ndat+i);
      }
   }

   afree(p);
   afree(pdat);
}


static void jfit_icov_r(int ndc,int ndat,double *amat,jdat_t ***pcov,
                        double *kmax)
{
   int i,j,n,nmat;
   double k,r;
   jdat_t **jcov;

   jcov=pcov[ndc-1];
   n=(*(jcov[0])).n;
   nmat=ndat*ndat;

   for (i=0;i<nmat;i++)
      amat[i]=(*(jcov[i])).fbar;

   k=house(ndat,amat);

   if (k>(*kmax))
      (*kmax)=k;

   for (i=0;i<nmat;i++)
   {
      (*(jcov[i])).fbar=amat[i];
      (*(jcov[i])).sigf=0.0;
   }

   for (i=0;i<n;i++)
   {
      for (j=0;j<nmat;j++)
         amat[j]=(*(jcov[j])).f[i];

      k=house(ndat,amat);

      if (k>(*kmax))
         (*kmax)=k;

      for (j=0;j<nmat;j++)
      {
         (*(jcov[j])).f[i]=amat[j];
         r=(*(jcov[j])).f[i]-(*(jcov[j])).fbar;
         (*(jcov[j])).sigf+=(r*r);
      }
   }

   for (i=0;i<nmat;i++)
      (*(jcov[i])).sigf=sqrt((*(jcov[i])).sigf);

   if (ndc>1)
   {
      for (i=0;i<n;i++)
      {
         for (j=0;j<nmat;j++)
            pcov[ndc-2][j]=(*(jcov[j])).next+i;

         jfit_icov_r(ndc-1,ndat,amat,pcov,kmax);
      }
   }
}


double jfit_icov(int ndat,jdat_t **jdat,jdat_t *jcov)
{
   int i,ndc,nmat;
   double kmax,*amat;
   jdat_t ***pcov,**p;

   jfit_cov(ndat,jdat,jcov);

   nmat=ndat*ndat;
   ndc=nolev(jcov);

   amat=amalloc(nmat*sizeof(*amat),4);
   pcov=amalloc(ndc*sizeof(*pcov),3);
   p=amalloc(ndc*nmat*sizeof(*p),3);
   error((amat==NULL)||(pcov==NULL)||(p==NULL),1,"jfit_icov [jfit.c]",
         "Unable to allocate auxiliary arrays");

   pcov[0]=p;

   for (i=1;i<ndc;i++)
      pcov[i]=pcov[i-1]+nmat;

   for (i=0;i<nmat;i++)
      pcov[ndc-1][i]=jcov+i;

   kmax=0.0;
   jfit_icov_r(ndc,ndat,amat,pcov,&kmax);

   afree(p);
   afree(pcov);
   afree(amat);

   return kmax;
}


static void alloc_csv(int n)
{
   if (n!=nsv)
   {
      if (nsv>0)
         afree(csv);

      nsv=n;
      csv=amalloc(nsv*nsv*sizeof(*csv),4);

      error(csv==NULL,1,"alloc_csv [jfit.c]",
            "Unable to allocate auxiliary array");
   }
}


static double fc0(int n,double *obs)
{
   int i,j;
   double r0,r1;

   r0=0.0;
   r1=0.0;

   for (i=0;i<nsv;i++)
   {
      for (j=0;j<nsv;j++)
      {
         r0+=csv[i*nsv+j]*obs[j];
         r1+=csv[i*nsv+j];
      }
   }

   return r0/r1;
}


static double fc1(int n,double *obs)
{
   int i,j;
   double r0,r1,*c;

   c=obs+nsv;
   r0=0.0;
   r1=0.0;

   for (i=0;i<nsv;i++)
   {
      for (j=0;j<nsv;j++)
      {
         r0+=c[i*nsv+j]*obs[j];
         r1+=c[i*nsv+j];
      }
   }

   return r0/r1;
}


static double chisq(int ndat,jdat_t **jdat,double c)
{
   int i,j;
   double r0,r1;

   r0=0.0;

   for (i=0;i<ndat;i++)
   {
      r1=0.0;

      for (j=0;j<ndat;j++)
         r1+=csv[i*ndat+j]*((*(jdat[j])).fbar-c);

      r0+=((*(jdat[i])).fbar-c)*r1;
   }

   return r0;
}


double jfit_const(int type,int ndat,jdat_t **jdat,double *k,jdat_t *jpar)
{
   int i,j;
   jdat_t **jall,*jcov;

   error((type<0)||(type>1)||(ndat<2),1,"jfit_const [jfit.c]",
         "Parameter type or ndat is out of range");

   alloc_csv(ndat);

   for (i=0;i<ndat;i++)
   {
      for (j=i;j<ndat;j++)
      {
         csv[i*ndat+j]=cov(jdat[i],jdat[j]);

         if (j!=i)
            csv[j*ndat+i]=csv[i*ndat+j];
      }
   }

   (*k)=house(ndat,csv);

   if (type==0)
      newobs(ndat,jdat,fc0,jpar);
   else
   {
      jall=amalloc((ndat+ndat*ndat)*sizeof(*jall),3);
      jcov=amalloc(ndat*ndat*sizeof(*jcov),3);

      error((jall==NULL)||(jcov==NULL),1,"jfit_const [jfit.c]",
            "Unable to allocate jdat structures");

      for (i=0;i<ndat;i++)
         jall[i]=jdat[i];

      for (i=0;i<(ndat*ndat);i++)
         jall[ndat+i]=jcov+i;

      (*k)=jfit_icov(ndat,jdat,jcov);
      newobs(ndat+ndat*ndat,jall,fc1,jpar);

      for (i=0;i<(ndat*ndat);i++)
         free_jdat(jcov+i);

      afree(jcov);
      afree(jall);
   }

   return chisq(ndat,jdat,(*jpar).fbar);
}


static void alloc_dsv(int ndat,int npar)
{
   if ((ndat!=ndsv)||(npar!=nxsv))
   {
      if ((ndsv>0)||(nxsv>0))
         afree(dsv);

      ndsv=ndat;
      nxsv=npar;
      dsv=amalloc((ndsv+2*nxsv)*sizeof(*dsv),3);
      xsv=dsv+ndsv;
      xmn=xsv+nxsv;

      error(dsv==NULL,1,"alloc_dsv [jfit.c]",
            "Unable to allocate auxiliary arrays");
   }
}


static double fm(int n,double *x)
{
   return fsv(ndsv,dsv,n,x);
}


static void jfit_r(int ndc,int ndat,jdat_t ***pdat,int npar,jdat_t ***ppar,
                   double *x0,double *x1,double *x2,
                   int imx,double omega1,double omega2,double *sall)
{
   int i,j,n,stat;
   double r;
   jdat_t **jdat,**jpar;

   if ((*sall)<0.0)
      return;

   jdat=pdat[ndc-1];
   jpar=ppar[ndc-1];
   n=(*(jdat[0])).n;

   for (i=0;i<ndat;i++)
      dsv[i]=(*(jdat[i])).fbar;

   if (isv==1)
      powell(npar,x0,xsv,x2,fm,imx,omega1,omega2,xmn,&stat);
   else
   {
      powell(npar,x0,x1,x2,fm,imx,omega1,omega2,xmn,&stat);

      for (j=0;j<npar;j++)
         xsv[j]=xmn[j];

      isv=1;
   }

   icnt+=1;

   if (stat>=0)
      (*sall)+=(double)(stat);
   else
   {
      (*sall)=(double)(stat);
      return;
   }

   for (j=0;j<npar;j++)
   {
      (*(jpar[j])).fbar=xmn[j];
      (*(jpar[j])).sigf=0.0;
   }

   for (i=0;i<n;i++)
   {
      for (j=0;j<ndat;j++)
         dsv[j]=(*(jdat[j])).f[i];

      powell(npar,x0,xsv,x2,fm,imx,omega1,omega2,xmn,&stat);
      icnt+=1;

      if (stat>=0)
         (*sall)+=(double)(stat);
      else
      {
         (*sall)=(double)(stat);
         return;
      }

      for (j=0;j<npar;j++)
      {
         (*(jpar[j])).f[i]=xmn[j];
         r=(*(jpar[j])).f[i]-(*(jpar[j])).fbar;
         (*(jpar[j])).sigf+=(r*r);
      }
   }

   for (j=0;j<npar;j++)
      (*(jpar[j])).sigf=sqrt((*(jpar[j])).sigf);

   if (ndc>1)
   {
      for (i=0;i<n;i++)
      {
         for (j=0;j<ndat;j++)
            pdat[ndc-2][j]=(*(jdat[j])).next+i;

         for (j=0;j<npar;j++)
            ppar[ndc-2][j]=(*(jpar[j])).next+i;

         jfit_r(ndc-1,ndat,pdat,npar,ppar,x0,x1,x2,imx,omega1,omega2,sall);
      }
   }
}


double jfit(int ndat,jdat_t **jdat,int npar,jdat_t **jpar,
            double (*lhf)(int ndat,double *dat,int npar,double *par),
            double *x0,double *x1,double *x2,
            int imx,double omega1,double omega2,int *status)
{
   int i,j,k;
   int n,ndc;
   double sall;
   jdat_t ***pdat,***ppar,**p;

   error((ndat<1)||(npar<1),1,"jfit [jfit.c]",
         "The parameters ndat and npar must be positive");

   alloc_dsv(ndat,npar);
   fsv=lhf;
   isv=0;
   icnt=0;
   sall=0.0;
   k=0;

   for (i=0;i<ndat;i++)
   {
      for (j=0;j<npar;j++)
      {
         if (jdat[i]==jpar[j])
            k=1;
      }
   }

   error(k!=0,1,"jfit [jfit.c]",
         "Attempt to overwrite input data");

   check_jdat(ndat,jdat,1);
   n=(*(jdat[0])).n;
   ndc=nolev(jdat[0]);

   for (i=1;i<ndat;i++)
   {
      k=nolev(jdat[i]);

      if (k<ndc)
         ndc=k;
   }

   for (i=0;i<npar;i++)
      alloc_jdat(n,ndc,jpar[i]);

   pdat=amalloc(2*ndc*sizeof(*pdat),3);
   p=amalloc(ndc*(ndat+npar)*sizeof(*p),3);
   error((pdat==NULL)||(p==NULL),1,"jfit [jfit.c]",
         "Unable to allocate auxiliary arrays");

   ppar=pdat+ndc;
   pdat[0]=p;
   ppar[0]=p+ndc*ndat;

   for (i=1;i<ndc;i++)
   {
      pdat[i]=pdat[i-1]+ndat;
      ppar[i]=ppar[i-1]+npar;
   }

   for (i=0;i<ndat;i++)
      pdat[ndc-1][i]=jdat[i];

   for (i=0;i<npar;i++)
      ppar[ndc-1][i]=jpar[i];

   jfit_r(ndc,ndat,pdat,npar,ppar,x0,x1,x2,imx,omega1,omega2,&sall);

   afree(p);
   afree(pdat);

   if (sall>=0)
   {
      sall/=(double)(icnt);
      (*status)=(int)(sall);

      for (i=0;i<ndat;i++)
         dsv[i]=(*(jdat[i])).fbar;

      for (i=0;i<npar;i++)
         xsv[i]=(*(jpar[i])).fbar;

      return lhf(ndat,dsv,npar,xsv);
   }
   else
   {
      (*status)=(int)(sall);

      for (i=0;i<npar;i++)
         alloc_jdat(0,1,jpar[i]);

      return 0.0;
   }
}
