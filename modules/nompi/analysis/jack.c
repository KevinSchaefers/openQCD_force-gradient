
/*******************************************************************************
*
* File jack.c
*
* Copyright (C) 2007,2008 Leonardo Giusti, Martin Luescher
*
* Basic jackknife error analysis programs
*
* The externally accessible functions are
*
*   void check_jdat(int nobs,jdat_t **jdat,int flag)
*     Performs various checks on the structures jdat[0],..,jdat[nobs-1]
*     depending on the value of flag. If flag>=0, the program checks that
*     the structures are all registered as allocated. If flag>=1 it also
*     checks that the number of data elements are the same and if flag>=2
*     additionally verifies that the number of jackknife decimation levels
*     does not vary
*
*   void free_jdat(jdat_t *jdat)
*     Frees the data arrays in the structure jdat and sets the elements of
*     the structure to zero. If the structure is not registered as being
*     allocated, the program does nothing. This program descends to the
*     jdat_t substructures of jdat and frees these recursively
*
*   void alloc_jdat(int n,int ndc,jdat_t *jdat)
*     Sets (*jdat).n to n and allocates the data array (*jdat).f accordingly.
*     The structure is then internally registered as being allocated and all
*     data elements as well as (*jdat).fbar and (*jdat).sigf are set to zero.
*     After that the program recursively decends to the jackknife decimation
*     levels where up to ndc elements are removed from the underlying data
*     series and allocates the corresponding jdat_t structures in the same
*     way. At the lowest level, the pointer *.next is set to NULL. If the
*     structure is already allocated, the arrays are freed and reallocated
*     only where needed
*
*   void jackbin(int n,int bs,int ndc,double *a,jdat_t *jdat)
*     Divides the array elements a[0],..,a[n-1] into bins of bs successive
*     elements and computes the bin averages b[0],..,b[n/bs-1]. The jackknife
*     samples of this series, decimated by 1 up to ndc elements, as well as
*     the corresponding averages and standard errors are then assigned to the
*     appropriate data elements in the jdat structure and its substructures
*
*   void newobs(int nobs,jdat_t **jdat,double (*f)(int nobs,double *obs),
*               jdat_t *jnew)
*     Computes the average, the statistical error and the jackknife samples
*     of a function f() of nobs observables whose jackknife samples are in
*     the structures jdat[0],..,jdat[nobs-1]. The results of the computation
*     are then assigned to the elements of the structure jnew, which is
*     allocated or reallocated if so required. All this is done down to all
*     available jackknife elimination levels
*
*   void jackcov(int nobs,jdat_t **jdat,double *cov)
*     Computes the covariance matrix cov[i*nobs+j] (i,j=0,..,nobs-1) of
*     nobs observables, assuming the associated jackknife samples are
*     contained in the structures jdat[0],..,jdat[nobs-1]
*
*   double jackbias(jdat_t *jdat)
*     Returns the bias of the observable whose jackknife samples are
*     contained in the jdat structure
*
*   void fwrite_jdat(jdat_t *jdat,FILE *out)
*     Writes the contents of the structure jdat to the stream "out"
*
*   void fread_jdat(jdat_t *jdat,FILE *in)
*     Reads the contents of the structure jdat from  the stream "in". The
*     structure is allocated or reallocated as required
*
*   void copy_jdat(jdat_t *jdin,jdat_t *jdout)
*     Copies the contents of the structure jdin to the structure jdout.
*     The latter is allocated or reallocated as required
*
*   void embed_jdat(int n,int offset,jdat_t *jdin,jdat_t *jdout)
*     Allocates jdout (if so required) with size parameter set to n and
*     embeds jdin in jdout at the specified offset. The array elements
*     jdout.f[offset+i] are thus set to jdin.f[i], i=0,..,jdin.n-1, and
*     otherwise to jdin.fbar. The averages jdout.fbar and jdout.sigf
*     are also copied from jdin and (*jdout).next is set to NULL
*
* Notes:
*
* For a given series a[0],..,a[n-1] of values of a primary stochastic
* variable, the elements of the associated jdat_t structure are
*
*  jdat.n                      array length
*  jdat.fbar                   average of the array elements
*  jdat.sigf                   statistical error of the average
*  jdat.f[0],..,jdat.f[n-1]    jackknife samples
*  jdat.next                   Pointer to the n-dimensional array of
*                              jdat_t structures at the next jackknife
*                              elimination level
*
* The single-elimination jackknife samples
*
*  jdat.f[i]=1/(n-1)*(sum_k{a[k]}-a[i])
*
* are defined as usual, where the sum goes from 0 to n-1. The samples with
* two decimations, first eliminating a[i] and then a[j+(i<=j)], j=0,..,n-2,
* are stored in the array elements
*
*  (jdat.next[i]).f[j]=1/(n-2)*(sum_k{a[k]}-a[i]-a[j+(i<=j)])
*
* where the sum goes from 0 to n-1. At this level
*
*  (jdat.next[i]).n=n-1
*  (jdat.next[i]).fbar=1/(n-1)*(sum_k{a[k]}-a[i])
*  (jdat.next[i]).sigf=sqrt{sum_j{((jdat.next[i]).fbar-(jdat.next[i]).f[j])^2}}
*
* The higher-elimination samples are defined recursively in the obvious way.
* Jackknife structures may contain just the single-elimination samples in
* which case jdat.next is set to NULL. In general, the number of available
* levels in the structure can be determined by following the *.next pointers
* until *.next=NULL
*
* The program jackbin serves to form the jackknife samples from a series
* of measurements of a primary observables. However, a jackknife structure
* jdat may also contain the jackknife samples of a function of the primary
* observables. The elements jdat.fbar and jdat.sigf are then the (biased)
* standard estimate and jackknife error of the observable
*
* A jdat_t structure with array length equal to zero is referred to
* as empty. The program alloc_jdat(), for example, may be called with
* size parameter n=0, in which case all elements of the structure are
* set to zero
*
* The jackknife error covariance of any two observables is given by
*
*  cov=sum_i{(jdat1.f[i]-jdat1.fbar)*(jdat2.f[i]-jdat2.fbar)}
*
* and the bias of an observable is
*
*  bias=sum_i{jdat.f[i]-jdat.fbar}
*
*  unbiased average=jdat.fbar-bias
*
* where the sums over i run from 0 to n-1
*
*******************************************************************************/

#define JACK_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "analysis.h"


struct ajdat_t
{
   jdat_t *j;
   struct ajdat_t *last,*next;
};

static struct ajdat_t *rpos=NULL;


static void ins_jdat(jdat_t *j)
{
   struct ajdat_t *p,*q;

   p=malloc(sizeof(*p));
   error(p==NULL,1,"ins_jdat [jack.c]",
         "Unable to register jdat_t structure");
   (*p).j=j;

   if (rpos!=NULL)
   {
      q=(*rpos).next;

      (*p).next=q;
      (*rpos).next=p;
      (*q).last=p;
      (*p).last=rpos;
   }
   else
   {
      (*p).next=p;
      (*p).last=p;
   }

   rpos=p;
}


static void rmv_jdat(jdat_t *j)
{
   struct ajdat_t *p,*pn,*pl;

   if (rpos!=NULL)
   {
      p=rpos;

      for (;;)
      {
         if ((*p).j==j)
         {
            pn=(*p).next;
            pl=(*p).last;

            if (pn!=p)
            {
               (*pl).next=pn;
               (*pn).last=pl;
               rpos=pl;
            }
            else
               rpos=NULL;

            free(p);
            return;
         }

         p=(*p).next;
         if (p==rpos)
            return;
      }
   }
}


static int fnd_jdat(jdat_t *j)
{
   struct ajdat_t *p;

   if (rpos!=NULL)
   {
      p=rpos;

      for (;;)
      {
         if ((*p).j==j)
         {
            rpos=p;
            return 1;
         }

         p=(*p).next;
         if (p==rpos)
            return 0;
      }
   }

   return 0;
}


static int nolev(jdat_t *jdat)
{
   int n;
   jdat_t *p;

   n=0;

   for (p=jdat;p!=NULL;p=(*p).next)
      n+=1;

   return n;
}


void check_jdat(int nobs,jdat_t **jdat,int flag)
{
   int n,i,it;

   error(nobs<1,1,"check_jdat [jack.c]","Argument nobs is out of range");

   it=0;

   if (flag>=0)
   {
      for (i=0;i<nobs;i++)
      {
         if (fnd_jdat(jdat[i])==0)
            it=1;
      }

      error(it!=0,1,"check_jdat [jack.c]",
            "Attempt to access unallocated jdat_t structure");
   }

   if (flag>=1)
   {
      n=(*(jdat[0])).n;

      for (i=1;i<nobs;i++)
      {
         if ((*(jdat[i])).n!=n)
            it=1;
      }

      error(it!=0,1,"check_jdat [jack.c]","Varying sample sizes");
   }

   if (flag>=2)
   {
      n=nolev(jdat[0]);

      for (i=1;i<nobs;i++)
      {
         if (nolev(jdat[i])!=n)
            it=1;
      }

      error(it!=0,1,"check_jdat [jack.c]","Varying decimation depth");
   }
}


void free_jdat(jdat_t *jdat)
{
   int i;

   if (fnd_jdat(jdat)==0)
      return;

   if ((*jdat).n>0)
   {
      afree((*jdat).f);

      if ((*jdat).next!=NULL)
      {
         for (i=0;i<(*jdat).n;i++)
            free_jdat((*jdat).next+i);
         afree((*jdat).next);
      }
   }

   (*jdat).n=0;
   (*jdat).fbar=0.0;
   (*jdat).sigf=0.0;
   (*jdat).f=NULL;
   (*jdat).next=NULL;

   rmv_jdat(jdat);
}


void alloc_jdat(int n,int ndc,jdat_t *jdat)
{
   int i;
   double *p;

   error((n<0)||(ndc<1)||((n>0)&&(ndc>=n)),1,"alloc_jdat [jack.c]",
         "Argument n or ndc is out of range");

   if (fnd_jdat(jdat)!=0)
   {
      if (n!=(*jdat).n)
         free_jdat(jdat);
      else
      {
         (*jdat).fbar=0.0;
         (*jdat).sigf=0.0;

         for (i=0;i<n;i++)
            (*jdat).f[i]=0.0;

         if (ndc>1)
         {
            if ((*jdat).next==NULL)
            {
               (*jdat).next=amalloc(n*sizeof(*jdat),3);
               error((*jdat).next==NULL,1,"alloc_jdat [jack.c]",
                     "Unable to allocate array of jdat_t structures");
            }

            for (i=0;i<n;i++)
               alloc_jdat(n-1,ndc-1,(*jdat).next+i);
         }
         else if ((*jdat).next!=NULL)
         {
            for (i=0;i<n;i++)
               free_jdat((*jdat).next+i);

            (*jdat).next=NULL;
         }

         return;
      }
   }

   if (n>0)
   {
      p=amalloc(n*sizeof(*p),4);
      error(p==NULL,1,"alloc_jdat [jack.c]","Unable to allocate data array");

      for (i=0;i<n;i++)
         p[i]=0.0;
   }
   else
      p=NULL;

   (*jdat).n=n;
   (*jdat).fbar=0.0;
   (*jdat).sigf=0.0;
   (*jdat).f=p;

   if ((n>1)&&(ndc>1))
   {
      (*jdat).next=amalloc(n*sizeof(*jdat),3);
      error((*jdat).next==NULL,1,"alloc_jdat [jack.c]",
            "Unable to allocate array of jdat_t structures");

      for (i=0;i<n;i++)
         alloc_jdat(n-1,ndc-1,(*jdat).next+i);
   }
   else
      (*jdat).next=NULL;

   ins_jdat(jdat);
}


static void jackbin_r(int n,int ndc,jdat_t *jdat)
{
   int i,j;
   double fbar,sigma,r,s,sm;
   jdat_t *p;

   if (ndc>1)
   {
      p=(*jdat).next;

      for (i=0;i<n;i++)
      {
         for (j=0;j<(n-1);j++)
            (*p).f[j]=(*jdat).f[j+(i<=j)];

         jackbin_r(n-1,ndc-1,p);
         p+=1;
      }
   }

   sm=0.0;

   for (i=0;i<n;i++)
      sm+=(*jdat).f[i];

   fbar=sm/(double)(n);
   s=1.0/(double)(n-1);
   sigma=0.0;

   for (i=0;i<n;i++)
   {
      (*jdat).f[i]=s*(sm-(*jdat).f[i]);
      r=fbar-(*jdat).f[i];
      sigma+=(r*r);
   }

   (*jdat).fbar=fbar;
   (*jdat).sigf=sqrt(sigma);
}


void jackbin(int n,int bs,int ndc,double *a,jdat_t *jdat)
{
   int i,j,nbin;
   double sm,s;

   error((bs<1)||((2*bs)>n),1,"jackbin [jack.c]",
         "Argument n or bs is out of range");

   nbin=n/bs;
   alloc_jdat(nbin,ndc,jdat);

   s=1.0/(double)(bs);

   for (i=0;i<nbin;i++)
   {
      sm=0.0;

      for (j=0;j<bs;j++)
         sm+=a[i*bs+j];

      (*jdat).f[i]=s*sm;
   }

   jackbin_r(nbin,ndc,jdat);
}


static void newobs_r(int nobs,double *obs,int ndc,jdat_t ***pp,
                     double (*f)(int nobs,double *obs),jdat_t *jnew)
{
   int n,i,j;
   double fbar,sigma,r;
   jdat_t **p;

   p=pp[ndc-1];
   n=(*(p[0])).n;

   for (j=0;j<nobs;j++)
     obs[j]=(*(p[j])).fbar;

   fbar=f(nobs,obs);
   sigma=0.0;

   for (i=0;i<n;i++)
   {
     for (j=0;j<nobs;j++)
       obs[j]=(*(p[j])).f[i];

     (*jnew).f[i]=f(nobs,obs);
     r=(*jnew).f[i]-fbar;
     sigma+=(r*r);
   }

   (*jnew).fbar=fbar;
   (*jnew).sigf=sqrt(sigma);

   if (ndc>1)
   {
     for (i=0;i<n;i++)
     {
       for (j=0;j<nobs;j++)
	 pp[ndc-2][j]=(*(p[j])).next+i;

       newobs_r(nobs,obs,ndc-1,pp,f,(*jnew).next+i);
     }
   }
}


void newobs(int nobs,jdat_t **jdat,double (*f)(int nobs,double *obs),
            jdat_t *jnew)
{
   int j,n,ndc;
   double *obs;
   jdat_t ***pp,**p;

   check_jdat(nobs,jdat,1);
   n=0;

   for (j=0;j<nobs;j++)
   {
      if (jnew==jdat[j])
         n=1;
   }

   error(n!=0,1,"newobs [jack.c]",
         "Attempt to overwrite the input observables");

   ndc=nolev(jdat[0]);

   for (j=1;j<nobs;j++)
   {
      n=nolev(jdat[j]);

      if (n<ndc)
         ndc=n;
   }

   n=jdat[0][0].n;
   alloc_jdat(n,ndc,jnew);

   obs=amalloc(nobs*sizeof(*obs),4);
   pp=amalloc(ndc*sizeof(*pp),3);
   p=amalloc(ndc*nobs*sizeof(*p),3);
   error((obs==NULL)||(pp==NULL)||(p==NULL),1,"newobs [jack.c]",
         "Unable to allocate auxiliary arrays");

   pp[0]=p;

   for (j=1;j<ndc;j++)
      pp[j]=pp[j-1]+nobs;

   for (j=0;j<nobs;j++)
      pp[ndc-1][j]=jdat[j];

   newobs_r(nobs,obs,ndc,pp,f,jnew);

   afree(p);
   afree(pp);
   afree(obs);
}


void jackcov(int nobs,jdat_t **jdat,double *cov)
{
   int n,i,j,k;
   double sm,*f1,*f2,fbar1,fbar2;

   check_jdat(nobs,jdat,1);
   n=(*(jdat[0])).n;

   for (i=0;i<nobs;i++)
   {
      for (j=i;j<nobs;j++)
      {
         sm=0.0;
         f1=(*(jdat[i])).f;
         f2=(*(jdat[j])).f;
         fbar1=(*(jdat[i])).fbar;
         fbar2=(*(jdat[j])).fbar;

         for (k=0;k<n;k++)
            sm+=((f1[k]-fbar1)*(f2[k]-fbar2));

         cov[i*nobs+j]=sm;
         cov[j*nobs+i]=sm;
      }
   }
}


double jackbias(jdat_t *jdat)
{
   int n,i;
   double fbar,*f,sm;

   error(fnd_jdat(jdat)==0,1,"jackbias [jack.c]",
         "Attempt to access unallocated jdat_t structure");

   error((*jdat).n==0,1,"jackbias [jack.c]","Empty jdat_t structure");

   n=(*jdat).n;
   fbar=(*jdat).fbar;
   f=(*jdat).f;
   sm=0.0;

   for (i=0;i<n;i++)
      sm+=(f[i]-fbar);

   return sm;
}


void fwrite_jdat(jdat_t *jdat,FILE *out)
{
   int i,n,flag,iw;

   error(fnd_jdat(jdat)==0,1,"fwrite_jdat [jack.c]",
         "Attempt to access unallocated jdat_t structure");

   n=(*jdat).n;
   iw=fwrite(&n,sizeof(int),1,out);
   iw+=fwrite(&((*jdat).fbar),sizeof(double),1,out);
   iw+=fwrite(&((*jdat).sigf),sizeof(double),1,out);
   iw+=fwrite((*jdat).f,sizeof(double),n,out);

   if ((*jdat).next!=NULL)
   {
      flag=1;
      iw+=fwrite(&flag,sizeof(int),1,out);
      for (i=0;i<n;i++)
         fwrite_jdat((*jdat).next+i,out);
   }
   else
   {
      flag=0;
      iw+=fwrite(&flag,sizeof(int),1,out);
   }

   error(iw!=(4+n),1,"fwrite_jdat [jack.c]",
         "Incorrect write count");
}


void fread_jdat(jdat_t *jdat,FILE *in)
{
   int i,n,flag,ir;

   ir=fread(&n,sizeof(int),1,in);
   error(ir!=1,1,"fread_jdat [jack.c]",
         "Unexpected end of file");

   alloc_jdat(n,1,jdat);
   ir+=fread(&((*jdat).fbar),sizeof(double),1,in);
   ir+=fread(&((*jdat).sigf),sizeof(double),1,in);
   ir+=fread((*jdat).f,sizeof(double),n,in);
   ir+=fread(&flag,sizeof(int),1,in);

   error(ir!=(4+n),1,"fread_jdat [jack.c]",
         "Incorrect read count");

   if (flag==1)
   {
      (*jdat).next=amalloc(n*sizeof(*jdat),3);

      error((*jdat).next==NULL,1,"fread_jdat [jack.c]",
            "Unable to allocate jdat structures");

      for (i=0;i<n;i++)
         fread_jdat((*jdat).next+i,in);
   }
}


void copy_jdat(jdat_t *jdin,jdat_t *jdout)
{
   int i,n;

   error(fnd_jdat(jdin)==0,1,"copy_jdat [jack.c]",
         "Attempt to access unallocated jdat_t structure");

   error(jdout==jdin,1,"copy_jdat [jack.c]",
         "Attempt to overwrite the input observable");

   n=(*jdin).n;
   alloc_jdat(n,1,jdout);

   (*jdout).fbar=(*jdin).fbar;
   (*jdout).sigf=(*jdin).sigf;

   for (i=0;i<n;i++)
      (*jdout).f[i]=(*jdin).f[i];

   if ((*jdin).next!=NULL)
   {
      (*jdout).next=amalloc(n*sizeof(*jdout),3);

      error((*jdout).next==NULL,1,"copy_jdat [jack.c]",
            "Unable to allocate jdat_t array");

      for (i=0;i<n;i++)
         copy_jdat((*jdin).next+i,(*jdout).next+i);
   }
}


void embed_jdat(int n,int offset,jdat_t *jdin,jdat_t *jdout)
{
   int i,j;

   error(fnd_jdat(jdin)==0,1,"embed_jdat [jack.c]",
         "Attempt to access unallocated jdat_t structure");

   error((offset<0)||(n<(offset+(*jdin).n)),1,"embed_jdat [jack.c]",
         "Incorrect offset or size parameter");

   alloc_jdat(n,1,jdout);

   (*jdout).fbar=(*jdin).fbar;
   (*jdout).sigf=(*jdin).sigf;

   for (i=0;i<n;i++)
   {
      j=i-offset;

      if ((j>=0)&&(j<(*jdin).n))
         (*jdout).f[i]=(*jdin).f[j];
      else
         (*jdout).f[i]=(*jdin).fbar;
   }
}
