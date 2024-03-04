
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2015, 2017 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of small DFT's (n=1,2,3,4,5).
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "utils.h"
#include "random.h"
#include "dft.h"

static complex_dble *w1,*w2,*w4,*w3,*w5;
static complex_dble wt[5],wt0[5]={{1.0,0.0},{1.0,0.0},{1.0,0.0},{1.0,0.0},
                                  {1.0,0.0}};
static complex_dble **f=NULL,**ft,**ftt;


static void set_wn(void)
{
   int i;
   double twopi;

   w1=amalloc(20*sizeof(*w1),4);
   error(w1==NULL,1,"set_wn [check1.c]",
         "Unable to allocate auxiliary arrays");

   w2=w1+2;
   w3=w2+3;
   w4=w3+4;
   w5=w4+5;
   twopi=8.0*atan(1.0);

   for (i=0;i<=1;i++)
   {
      w1[i].re=1.0;
      w1[i].im=0.0;
   }

   for (i=0;i<=2;i++)
   {
      w2[i].re=cos((double)(i)*twopi/2.0);
      w2[i].im=sin((double)(i)*twopi/2.0);
   }

   for (i=0;i<=3;i++)
   {
      w3[i].re=cos((double)(i)*twopi/3.0);
      w3[i].im=sin((double)(i)*twopi/3.0);
   }

   for (i=0;i<=4;i++)
   {
      w4[i].re=cos((double)(i)*twopi/4.0);
      w4[i].im=sin((double)(i)*twopi/4.0);
   }

   for (i=0;i<=5;i++)
   {
      w5[i].re=cos((double)(i)*twopi/5.0);
      w5[i].im=sin((double)(i)*twopi/5.0);
   }

   gauss_dble((double*)(wt),10);
}


static void set_fn(int n,int nfc)
{
   int i;
   complex_dble *f0;

   if (f!=NULL)
   {
      afree(f[0]);
      free(f);
   }

   f=malloc(3*n*sizeof(*f));
   f0=amalloc(3*n*nfc*sizeof(*f0),4);
   error((f==NULL)||(f0==NULL),1,"set_fn [check1.c]",
         "Unable to allocate function arrays");

   gauss_dble((double*)(f0),6*n*nfc);

   for (i=0;i<(3*n);i++)
      f[i]=f0+i*nfc;

   ft=f+n;
   ftt=ft+n;
}


static double check_dft(int s,int n,int nfc)
{
   int i,j,k,l;
   double d,dmx,r,rmx;
   complex_dble sm,wf,*w,*z;

   if (n==1)
      w=w1;
   else if (n==2)
      w=w2;
   else if (n==3)
      w=w3;
   else if (n==4)
      w=w4;
   else
      w=w5;

   dmx=0.0;
   rmx=0.0;

   for (i=0;i<nfc;i++)
   {
      r=0.0;

      for (j=0;j<n;j++)
      {
         r+=fabs(f[j][i].re)+fabs(f[j][i].im);

         sm.re=0.0;
         sm.im=0.0;

         for (k=0;k<n;k++)
         {
            l=(j*k)%n;

            if (s==-1)
               l=n-l;

            z=f[k]+i;
            wf.re=wt[k].re*(*z).re-wt[k].im*(*z).im;
            wf.im=wt[k].re*(*z).im+wt[k].im*(*z).re;

            sm.re+=(w[l].re*wf.re-w[l].im*wf.im);
            sm.im+=(w[l].re*wf.im+w[l].im*wf.re);
         }

         d=fabs(sm.re-ft[j][i].re)+fabs(sm.im-ft[j][i].im);

         if (d>dmx)
            dmx=d;
      }

      if (r>rmx)
         rmx=r;
   }

   return dmx/rmx;
}


static double cmp_fn(int n,int nfc,double c)
{
   int i,j;
   double d,dmx,r,rmx;
   complex_dble z1,z2;

   dmx=0.0;
   rmx=0.0;

   for (i=0;i<nfc;i++)
   {
      for (j=0;j<n;j++)
      {
         z1=f[j][i];
         z2=ftt[j][i];

         r=fabs(z1.re)+fabs(z1.im);
         if (r>rmx)
            rmx=r;

         d=fabs(z1.re-c*z2.re)+fabs(z1.im-c*z2.im);
         if (d>dmx)
            dmx=d;
      }
   }

   return dmx/rmx;
}


static double cmp_wfn(int n,int nfc,double c)
{
   int i,j;
   double d,dmx,r,rmx;
   complex_dble wf,z1,z2;

   dmx=0.0;
   rmx=0.0;

   for (i=0;i<nfc;i++)
   {
      for (j=0;j<n;j++)
      {
         z1=f[j][i];
         z2=ftt[j][i];

         wf.re=wt[j].re*z1.re-wt[j].im*z1.im;
         wf.im=wt[j].re*z1.im+wt[j].im*z1.re;

         r=fabs(wf.re)+fabs(wf.im);
         if (r>rmx)
            rmx=r;

         d=fabs(wf.re-c*z2.re)+fabs(wf.im-c*z2.im);
         if (d>dmx)
            dmx=d;
      }
   }

   return dmx/rmx;
}


int main(void)
{
   int n,nfc;
   dft_wsp_t *dwsp;

   printf("\n");
   printf("Small DFT's (n=1,2,3,4,5)\n");
   printf("-------------------------\n\n");

   rlxd_init(1,0,1234,1);
   dwsp=alloc_dft_wsp();
   error(dwsp==NULL,1,"main [check1.c]","Unable to allocate workspace");
   set_wn();

   while(1)
   {
      printf("Specify number n of function values and nfc of functions: ");
      (void)(scanf("%d %d",&n,&nfc));
      error(n>5,1,"main [check1.c]","n must be less than or equal to 5");

      set_fn(n,nfc);
      error(small_dft(1,n,nfc,wt,f,dwsp,ft)!=0,1,"main [check1.c]",
            "small_dft() failed");
      printf("Maximal relative deviation (s=+1) = %.1e\n",check_dft(1,n,nfc));

      set_fn(n,nfc);
      (void)(small_dft(-1,n,nfc,wt,f,dwsp,ft));
      printf("Maximal relative deviation (s=-1) = %.1e\n",check_dft(-1,n,nfc));

      (void)(small_dft(1,n,nfc,wt0,ft,dwsp,ftt));
      printf("Maximal relative deviation (inverse FT) = %.1e\n",
             cmp_wfn(n,nfc,1.0/(double)(n)));

      set_fn(n,nfc);
      (void)(small_dft(1,n,nfc,wt,f,dwsp,ftt));
      (void)(small_dft(1,n,nfc,wt,f,dwsp,f));
      printf("Maximal relative deviation (in place FT) = %.1e\n\n",
             cmp_fn(n,nfc,1.0));
   }

   exit(0);
}
