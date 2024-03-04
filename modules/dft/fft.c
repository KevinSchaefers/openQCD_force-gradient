
/*******************************************************************************
*
* File fft.c
*
* Copyright (C) 2015, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Fast Fourier transform.
*
*   int fft(dft_parms_t *dp,int nf,complex_dble **f,complex_dble **ft)
*     Applies the Fourier transform specified by the parameter set dp to the
*     functions f and assigns the calculated functions to ft (see the notes).
*
*   int inv_fft(dft_parms_t *dp,int nf,complex_dble **ft,complex_dble **f)
*     Applies the inverse of the Fourier transform specified by the parameter
*     set dp to the functions ft and assigns the calculated functions to f
*     (see the notes).
*
* These programs operate on arrays f[x][i] and ft[k][i] of functions labeled
* by the index i=0,1,..,nf-1. The range of the position x and the momentum k
* depend on the type of Fourier transform that is to be applied. The Fourier
* transform cannot be performed in place, i.e. the data arrays pointed to by
* f and ft may not overlap. On exit the pointer arrays f and ft are unchanged
* and so are the input data.
*
* If (*dp).type=EXP, the ranges are
*
*  x=0,1,..,n-1,   k=0,1,..,n-1,
*
* where n=(*dp).n. The Fourier transform and its inverse are applied according
* to eqs.(2.4) and (2.6) in the notes
*
*  M. Luescher: "Discrete Fourier transform", January 2015, doc/dft.pdf,
*
* in this case.
*
* If (*dp).type=SIN or (*dp).type=COS, the ranges are
*
*  x=0,1,..,n,   k=0,1,..,n.
*
* The input data at the endpoints of these ranges are not used if they are
* determined by the constraints (3.2)-(3.4), but the calculated functions
* are guaranteed to have the correct values at the endpoints. Depending on
* the shift bits b,c and the transformation type, one of the transformations
* (3.5) or (3.6) is applied in these cases.
*
* The programs in this module are thread safe. Any errors occuring in the
* course of their execution result in a non-zero return value and incorrect
* results.
*
*******************************************************************************/

#define FFT_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "dft.h"

#if (defined x64)
#include "sse2.h"
#endif


static void set_fs(dft_parms_t *dp,complex_dble **f,dft_wsp_t *dwsp)
{
   int n,c,*r;
   int N,i,j;
   complex_dble **fs;
   dft_type_t t;

   t=(*dp).type;
   n=(*dp).n;
   c=(*dp).c;
   r=(*dp).r;

   if (t==EXP)
      N=n;
   else
      N=2*n;

   fs=(*dwsp).fs;

   if (t==EXP)
   {
      for (i=0;i<N;i++)
         fs[i]=f[r[i]];
   }
   else
   {
      for (i=0;i<N;i++)
      {
         j=r[i];

         if (j<n)
            fs[i]=f[j];
         else
            fs[i]=f[N-j-c];
      }
   }
}


static void set_fts(dft_parms_t *dp,int nf,complex_dble **ft,dft_wsp_t *dwsp)
{
   int n,i;
   complex_dble **fts;

   n=(*dp).n;
   fts=(*dwsp).fts;

   for (i=0;i<n;i++)
      fts[i]=ft[i];

   if ((*dp).type!=EXP)
   {
      fts[n]=ft[n];

      if (n>1)
      {
         fts[n+1]=(*dwsp).buf;

         for (i=(n+2);i<(2*n);i++)
            fts[i]=fts[i-1]+nf;
      }
   }
}


static void apply_wc(dft_parms_t *dp,int nf,complex_dble **fts)
{
   int n,b,c,d;
   int i,j;
   complex_dble *wc,w,z;

   n=(*dp).n;
   b=(*dp).b;
   c=(*dp).c;
   d=((*dp).type==SIN);

   if (c==1)
   {
      wc=(*dp).wc;

      for (i=0;i<=n;i++)
      {
         if ((i<n)||((*dp).type!=EXP))
         {
            w.re=wc[i].re;
            w.im=wc[i].im;

            for (j=0;j<nf;j++)
            {
               z.re=fts[i][j].re;
               z.im=fts[i][j].im;
               fts[i][j].re=w.re*z.re-w.im*z.im;
               fts[i][j].im=w.re*z.im+w.im*z.re;
            }
         }
      }
   }

   if ((*dp).type!=EXP)
   {
      if (b==0)
      {
         if (d==1)
         {
            for (j=0;j<nf;j++)
            {
               fts[0][j].re=0.0;
               fts[0][j].im=0.0;
            }
         }

         if ((c+d)==1)
         {
            for (j=0;j<nf;j++)
            {
               fts[n][j].re=0.0;
               fts[n][j].im=0.0;
            }
         }
      }
      else
      {
         if ((c+d)==1)
         {
            for (j=0;j<nf;j++)
            {
               fts[n][j].re=-fts[n-1][j].re;
               fts[n][j].im=-fts[n-1][j].im;
            }
         }
         else
         {
            for (j=0;j<nf;j++)
            {
               fts[n][j].re=fts[n-1][j].re;
               fts[n][j].im=fts[n-1][j].im;
            }
         }
      }
   }
}

#if (defined x64)

static void fft_step(int nf,complex_dble *w,complex_dble *fe,complex_dble *fo)
{
   complex_dble *fem;

   _sse_load_cmplx_dble(*w);

   fem=fe+nf-(nf%4);

   for (;fe<fem;fe+=4)
   {
      __asm__ __volatile__ ("movapd %0, %%xmm0 \n\t"
                            "movapd %1, %%xmm1 \n\t"
                            "movapd %2, %%xmm2 \n\t"
                            "movapd %3, %%xmm3 \n\t"
                            "movapd %%xmm0, %%xmm4 \n\t"
                            "movapd %%xmm1, %%xmm5 \n\t"
                            "movapd %%xmm2, %%xmm8 \n\t"
                            "movapd %%xmm3, %%xmm9"
                            :
                            :
                            "m" (fo[0]),
                            "m" (fo[1]),
                            "m" (fo[2]),
                            "m" (fo[3])
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5", "xmm8", "xmm9");

      __asm__ __volatile__ ("mulpd %%xmm6, %%xmm0 \n\t"
                            "mulpd %%xmm6, %%xmm1 \n\t"
                            "mulpd %%xmm6, %%xmm2 \n\t"
                            "mulpd %%xmm6, %%xmm3 \n\t"
                            "shufpd $0x1, %%xmm4, %%xmm4 \n\t"
                            "shufpd $0x1, %%xmm5, %%xmm5 \n\t"
                            "shufpd $0x1, %%xmm8, %%xmm8 \n\t"
                            "shufpd $0x1, %%xmm9, %%xmm9 \n\t"
                            "mulpd %%xmm7, %%xmm4 \n\t"
                            "mulpd %%xmm7, %%xmm5 \n\t"
                            "mulpd %%xmm7, %%xmm8 \n\t"
                            "mulpd %%xmm7, %%xmm9 \n\t"
                            "addpd %%xmm4, %%xmm0 \n\t"
                            "addpd %%xmm5, %%xmm1 \n\t"
                            "addpd %%xmm8, %%xmm2 \n\t"
                            "addpd %%xmm9, %%xmm3"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5", "xmm8", "xmm9");

      __asm__ __volatile__ ("movapd %0, %%xmm10 \n\t"
                            "movapd %1, %%xmm11 \n\t"
                            "movapd %2, %%xmm12 \n\t"
                            "movapd %3, %%xmm13 \n\t"
                            "movapd %%xmm10, %%xmm4 \n\t"
                            "movapd %%xmm11, %%xmm5 \n\t"
                            "movapd %%xmm12, %%xmm8 \n\t"
                            "movapd %%xmm13, %%xmm9"
                            :
                            :
                            "m" (fe[0]),
                            "m" (fe[1]),
                            "m" (fe[2]),
                            "m" (fe[3])
                            :
                            "xmm4", "xmm5", "xmm8", "xmm9",
                            "xmm10", "xmm11", "xmm12", "xmm13");

      __asm__ __volatile__ ("subpd %%xmm0, %%xmm10 \n\t"
                            "subpd %%xmm1, %%xmm11 \n\t"
                            "subpd %%xmm2, %%xmm12 \n\t"
                            "subpd %%xmm3, %%xmm13 \n\t"
                            "addpd %%xmm0, %%xmm4 \n\t"
                            "addpd %%xmm1, %%xmm5 \n\t"
                            "addpd %%xmm2, %%xmm8 \n\t"
                            "addpd %%xmm3, %%xmm9"
                            :
                            :
                            :
                            "xmm4", "xmm5", "xmm8", "xmm9",
                            "xmm10", "xmm11", "xmm12", "xmm13");

      __asm__ __volatile__ ("movapd %%xmm10, %0 \n\t"
                            "movapd %%xmm11, %1 \n\t"
                            "movapd %%xmm12, %2 \n\t"
                            "movapd %%xmm13, %3 \n\t"
                            "movapd %%xmm4, %4 \n\t"
                            "movapd %%xmm5, %5 \n\t"
                            "movapd %%xmm8, %6 \n\t"
                            "movapd %%xmm9, %7"
                            :
                            "=m" (fo[0]),
                            "=m" (fo[1]),
                            "=m" (fo[2]),
                            "=m" (fo[3]),
                            "=m" (fe[0]),
                            "=m" (fe[1]),
                            "=m" (fe[2]),
                            "=m" (fe[3]));

      fo+=4;
   }

   fem+=(nf%4);

   for (;fe<fem;fe++)
   {
      __asm__ __volatile__ ("movapd %0, %%xmm0 \n\t"
                            "movapd %%xmm0, %%xmm1"
                            :
                            :
                            "m" (fo[0])
                            :
                            "xmm0", "xmm1");

      __asm__ __volatile__ ("mulpd %%xmm6, %%xmm0 \n\t"
                            "shufpd $0x1, %%xmm1, %%xmm1 \n\t"
                            "mulpd %%xmm7, %%xmm1 \n\t"
                            "addpd %%xmm1, %%xmm0 \n\t"
                            "movapd %0, %%xmm2 \n\t"
                            "movapd %%xmm2, %%xmm3 \n\t"
                            "subpd %%xmm0, %%xmm2 \n\t"
                            "addpd %%xmm0, %%xmm3"
                            :
                            :
                            "m" (fe[0])
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3");

      __asm__ __volatile__ ("movapd %%xmm2, %0 \n\t"
                            "movapd %%xmm3, %1"
                            :
                            "=m" (fo[0]),
                            "=m" (fe[0]));

      fo+=1;
   }
}

#else

static void fft_step(int nf,complex_dble *w,complex_dble *fe,complex_dble *fo)
{
   int i;
   complex_dble z;

   for (i=0;i<nf;i++)
   {
      z.re=(*w).re*(*fo).re-(*w).im*(*fo).im;
      z.im=(*w).re*(*fo).im+(*w).im*(*fo).re;

      (*fo).re=(*fe).re-z.re;
      (*fo).im=(*fe).im-z.im;

      (*fe).re+=z.re;
      (*fe).im+=z.im;

      fe+=1;
      fo+=1;
   }
}

#endif

static void fft_blk(dft_parms_t *dp,dft_wsp_t *dwsp,int N,int m,int p,int nf)
{
   int i,j,k;
   int mb,mhb,nmb;
   complex_dble **fs,**fts,*w,*wb;

   w=(*dp).w;
   wb=(*dp).wb;
   fs=(*dwsp).fs;
   fts=(*dwsp).fts;

   for (i=0;i<N;i+=m)
      (void)(small_dft(1,m,nf,wb+i,fs+i,dwsp,fts+i));

   mb=m;
   nmb=N/mb;

   for (i=0;i<p;i++)
   {
      mhb=mb;
      mb*=2;
      nmb/=2;

      for (j=0;j<N;j+=mb)
      {
         for (k=0;k<mhb;k++)
            fft_step(nf,w+k*nmb,fts[j+k],fts[j+k+mhb]);
      }
   }

   apply_wc(dp,nf,fts);
}


int fft(dft_parms_t *dp,int nf,complex_dble **f,complex_dble **ft)
{
   int N,n,m,p,i;
   int nb,ib;
   complex_dble **fs,**fts;
   dft_wsp_t *dwsp;

   n=(*dp).n;

   if ((*dp).type!=EXP)
      N=2*n;
   else
      N=n;

   p=0;
   m=N;

   while ((m>4)&&((m&0x1)==0))
   {
      p+=1;
      m/=2;
   }

   nb=512/n;
   if (nb<4)
      nb=4;

   dwsp=alloc_dft_wsp();

   if ((dwsp==NULL)||
       ((m>5)&&(set_dft_wsp0(m,dwsp)==NULL))||
       (set_dft_wsp1(N,dwsp)==NULL)||
       (((*dp).type!=EXP)&&(set_dft_wsp2((n-1)*nb,dwsp)==NULL)))
   {
      free_dft_wsp(dwsp);
      return 1;
   }

   set_fs(dp,f,dwsp);
   set_fts(dp,nb,ft,dwsp);

   fs=(*dwsp).fs;
   fts=(*dwsp).fts;

   for (ib=0;ib<nf;ib+=nb)
   {
      if ((ib+nb)>nf)
         nb=nf-ib;

      fft_blk(dp,dwsp,N,m,p,nb);

      for (i=0;i<N;i++)
         fs[i]+=nb;

      for (i=0;i<n;i++)
         fts[i]+=nb;

      if ((*dp).type!=EXP)
         fts[n]+=nb;
   }

   free_dft_wsp(dwsp);

   return 0;
}


static void inv_set_fts(dft_parms_t *dp,complex_dble **ft,dft_wsp_t *dwsp)
{
   int n,b,*r;
   int N,i,j;
   complex_dble **fts;
   dft_type_t t;

   t=(*dp).type;
   n=(*dp).n;
   b=(*dp).b;
   r=(*dp).r;

   if (t==EXP)
      N=n;
   else
      N=2*n;

   fts=(*dwsp).fts;

   if (t==EXP)
   {
      for (i=0;i<N;i++)
         fts[i]=ft[r[i]];
   }
   else
   {
      for (i=0;i<N;i++)
      {
         j=r[i];

         if (j<n)
            fts[i]=ft[j];
         else
            fts[i]=ft[N-j-b];
      }
   }
}


static void inv_set_fs(dft_parms_t *dp,int nf,complex_dble **f,dft_wsp_t *dwsp)
{
   int n,i;
   complex_dble **fs;

   n=(*dp).n;
   fs=(*dwsp).fs;

   for (i=0;i<n;i++)
      fs[i]=f[i];

   if ((*dp).type!=EXP)
   {
      fs[n]=f[n];

      if (n>1)
      {
         fs[n+1]=(*dwsp).buf;

         for (i=(n+2);i<(2*n);i++)
            fs[i]=fs[i-1]+nf;
      }
   }
}


static void inv_apply_wb(dft_parms_t *dp,int nf,complex_dble **fs)
{
   int n,b,c,d;
   int i,j;
   complex_dble *iwb,w,z;

   n=(*dp).n;
   b=(*dp).b;
   c=(*dp).c;
   d=((*dp).type==SIN);

   if (b==1)
   {
      iwb=(*dp).iwb;

      for (i=0;i<=n;i++)
      {
         if ((i<n)||((*dp).type!=EXP))
         {
            w.re=iwb[i].re;
            w.im=iwb[i].im;

            for (j=0;j<nf;j++)
            {
               z.re=fs[i][j].re;
               z.im=fs[i][j].im;
               fs[i][j].re=w.re*z.re-w.im*z.im;
               fs[i][j].im=w.re*z.im+w.im*z.re;
            }
         }
      }
   }

   if ((*dp).type!=EXP)
   {
      if (c==0)
      {
         if (d==1)
         {
            for (j=0;j<nf;j++)
            {
               fs[0][j].re=0.0;
               fs[0][j].im=0.0;
            }
         }

         if ((b+d)==1)
         {
            for (j=0;j<nf;j++)
            {
               fs[n][j].re=0.0;
               fs[n][j].im=0.0;
            }
         }
      }
      else
      {
         if ((b+d)==1)
         {
            for (j=0;j<nf;j++)
            {
               fs[n][j].re=-fs[n-1][j].re;
               fs[n][j].im=-fs[n-1][j].im;
            }
         }
         else
         {
            for (j=0;j<nf;j++)
            {
               fs[n][j].re=fs[n-1][j].re;
               fs[n][j].im=fs[n-1][j].im;
            }
         }
      }
   }
}


static void inv_fft_blk(dft_parms_t *dp,dft_wsp_t *dwsp,
                        int N,int m,int p,int nf)
{
   int i,j,k;
   int mb,mhb,nmb;
   complex_dble z,*w,*iwc;
   complex_dble **fs,**fts;

   w=(*dp).w;
   iwc=(*dp).iwc;
   fs=(*dwsp).fs;
   fts=(*dwsp).fts;

   for (i=0;i<N;i+=m)
      (void)(small_dft(-1,m,nf,iwc+i,fts+i,dwsp,fs+i));

   mb=m;
   nmb=N/mb;

   for (i=0;i<p;i++)
   {
      mhb=mb;
      mb*=2;
      nmb/=2;

      for (j=0;j<N;j+=mb)
      {
         for (k=0;k<mhb;k++)
         {
            z.re=w[k*nmb].re;
            z.im=-w[k*nmb].im;
            fft_step(nf,&z,fs[j+k],fs[j+k+mhb]);
         }
      }
   }

   inv_apply_wb(dp,nf,fs);
}


int inv_fft(dft_parms_t *dp,int nf,complex_dble **ft,complex_dble **f)
{
   int N,n,m,p,i;
   int nb,ib;
   complex_dble **fs,**fts;
   dft_wsp_t *dwsp;

   n=(*dp).n;

   if ((*dp).type!=EXP)
      N=2*n;
   else
      N=n;

   p=0;
   m=N;

   while ((m>4)&&((m&0x1)==0))
   {
      p+=1;
      m/=2;
   }

   nb=512/n;
   if (nb<4)
      nb=4;

   dwsp=alloc_dft_wsp();

   if ((dwsp==NULL)||
       ((m>5)&&(set_dft_wsp0(m,dwsp)==NULL))||
       (set_dft_wsp1(N,dwsp)==NULL)||
       (((*dp).type!=EXP)&&(set_dft_wsp2((n-1)*nb,dwsp)==NULL)))
   {
      free_dft_wsp(dwsp);
      return 1;
   }

   inv_set_fts(dp,ft,dwsp);
   inv_set_fs(dp,nb,f,dwsp);

   fts=(*dwsp).fts;
   fs=(*dwsp).fs;

   for (ib=0;ib<nf;ib+=nb)
   {
      if ((ib+nb)>nf)
         nb=nf-ib;

      inv_fft_blk(dp,dwsp,N,m,p,nb);

      for (i=0;i<N;i++)
         fts[i]+=nb;

      for (i=0;i<n;i++)
         fs[i]+=nb;

      if ((*dp).type!=EXP)
         fs[n]+=nb;
   }

   free_dft_wsp(dwsp);

   return 0;
}
