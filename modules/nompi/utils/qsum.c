
/*******************************************************************************
*
* File qsum.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Quadruple-precision arithmetic.
*
*   void acc_qflt(double u,double *qr)
*     Adds the double-precision number u to the quadruple-precision
*     number qr. The addition is performed in quadruple precision.
*
*   void add_qflt(double *qu,double *qv,double *qr)
*     Assigns the sum of the quadruple-precision numbers qu and qv to qr.
*     The addition is performed in quadruple precision. It is permissible
*     to set qr to qu or qv.
*
*   void scl_qflt(double u,double *qr)
*     Multiplies the quadruple-precision number qr by the double-precision
*     number u. The multiplication is performed in quadruple precision.
*
*   void mul_qflt(double *qu,double *qv,double *qr)
*     Assigns the product of the quadruple-precision numbers qu and qv to qr.
*     The multiplication is performed in quadruple precision. It is permissible
*     to set qr to qu or qv.
*
* Quadruple-precision numbers are represented by pairs q[2] of double-precision
* numbers such that the sum q[0]+q[1] calculated by the FPU coincides with q[0].
* See
*
*  M. Luescher: Quadruple-precision summation in openQCD
*
* [doc/qsum.pdf] for further explanations. The arguments qx of the programs
* acc_qflt() and add_qflt() are thus assumed to be arrays of double numbers
* of length 2 (or more).
*
* Compliance with the IEEE-754 standard for double-precision data, additions,
* subtractions and multiplications (with rounding to nearest and ties to even)
* is taken for granted.
*
*******************************************************************************/

#define QSUM_C

#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

static double mm=134217729.0;

#if ((defined AVX)||(defined x64))

void acc_qflt(double u,double *qr)
{
   __asm__ __volatile__ ("movsd %0, %%xmm0 \n\t"
                         "movsd %1, %%xmm1 \n\t"
                         "movsd %2, %%xmm2 \n\t"
                         "movapd %%xmm0, %%xmm3 \n\t"
                         "movapd %%xmm1, %%xmm4"
                         :
                         :
                         "m" (u),
                         "m" (qr[0]),
                         "m" (qr[1])
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4");

   __asm__ __volatile__ ("addsd %%xmm0, %%xmm1 \n\t"
                         "movapd %%xmm1, %%xmm5 \n\t"
                         "subsd %%xmm1, %%xmm3 \n\t"
                         "addsd %%xmm3, %%xmm4 \n\t"
                         "addsd %%xmm1, %%xmm3 \n\t"
                         "subsd %%xmm3, %%xmm0 \n\t"
                         "addsd %%xmm4, %%xmm0"
                         :
                         :
                         :
                         "xmm0", "xmm1", "xmm3", "xmm4",
                         "xmm5");

   __asm__ __volatile__ ("addsd %%xmm2, %%xmm0 \n\t"
                         "addsd %%xmm0, %%xmm1 \n\t"
                         "subsd %%xmm1, %%xmm5 \n\t"
                         "addsd %%xmm5, %%xmm0 \n\t"
                         "movsd %%xmm1, %0 \n\t"
                         "movsd %%xmm0, %1"
                         :
                         "=m" (qr[0]),
                         "=m" (qr[1])
                         :
                         :
                         "xmm0", "xmm1", "xmm5");
}


void add_qflt(double *qu,double *qv,double *qr)
{
   __asm__ __volatile__ ("movupd %0, %%xmm0 \n\t"
                         "movupd %2, %%xmm1 \n\t"
                         "movapd %%xmm0, %%xmm2 \n\t"
                         "movapd %%xmm1, %%xmm3"
                         :
                         :
                         "m" (qu[0]),
                         "m" (qu[1]),
                         "m" (qv[0]),
                         "m" (qv[1])
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3");

   __asm__ __volatile__ ("addpd %%xmm1, %%xmm0 \n\t"
                         "movapd %%xmm0, %%xmm4 \n\t"
                         "subpd %%xmm0, %%xmm3 \n\t"
                         "addpd %%xmm3, %%xmm2 \n\t"
                         "addpd %%xmm4, %%xmm3 \n\t"
                         "subpd %%xmm3, %%xmm1 \n\t"
                         "addpd %%xmm2, %%xmm1 \n\t"
                         "movhlps %%xmm0, %%xmm5 \n\t"
                         "movhlps %%xmm1, %%xmm6"
                         :
                         :
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3",
                         "xmm4", "xmm5", "xmm6");

   __asm__ __volatile__ ("addsd %%xmm5, %%xmm1 \n\t"
                         "addsd %%xmm1, %%xmm4 \n\t"
                         "subsd %%xmm4, %%xmm0 \n\t"
                         "movapd %%xmm4, %%xmm2 \n\t"
                         "addsd %%xmm0, %%xmm1 \n\t"
                         "addsd %%xmm6, %%xmm1 \n\t"
                         "addsd %%xmm1, %%xmm2 \n\t"
                         "subsd %%xmm2, %%xmm4 \n\t"
                         "addsd %%xmm4, %%xmm1 \n\t"
                         "movsd %%xmm2, %0 \n\t"
                         "movsd %%xmm1, %1"
                         :
                         "=m" (qr[0]),
                         "=m" (qr[1])
                         :
                         :
                         "xmm0", "xmm1", "xmm2", "xmm4");
}


static void exact_prod(double u,double v,double *qr)
{
   __asm__ __volatile__ ("movsd %0, %%xmm0 \n\t"
                         "mulsd %1, %%xmm0 \n\t"
                         "movddup %2, %%xmm1 \n\t"
                         "movlpd %0, %%xmm2 \n\t"
                         "movhpd %1, %%xmm2 \n\t"
                         "movapd %%xmm2, %%xmm3"
                         :
                         :
                         "m" (u),
                         "m" (v),
                         "m" (mm)
                         :
                         "xmm0", "xmm1", "xmm2", "xmm3");

   __asm__ __volatile__ ("mulpd %%xmm2, %%xmm1 \n\t"
                         "subpd %%xmm1, %%xmm3 \n\t"
                         "addpd %%xmm3, %%xmm1 \n\t"
                         "subpd %%xmm1, %%xmm2 \n\t"
                         "movddup %%xmm1, %%xmm4 \n\t"
                         "movddup %%xmm2, %%xmm5 \n\t"
                         "movhlps %%xmm1, %%xmm2 \n\t"
                         "mulpd %%xmm2, %%xmm4 \n\t"
                         "mulpd %%xmm2, %%xmm5 \n\t"
                         "movhlps %%xmm4, %%xmm6 \n\t"
                         "movhlps %%xmm5, %%xmm7"
                         :
                         :
                         :
                         "xmm1", "xmm2", "xmm3", "xmm4",
                         "xmm5", "xmm6", "xmm7");

   __asm__ __volatile__ ("subsd %%xmm0, %%xmm4 \n\t"
                         "addsd %%xmm4, %%xmm6 \n\t"
                         "addsd %%xmm6, %%xmm5 \n\t"
                         "addsd %%xmm5, %%xmm7 \n\t"
                         "movsd %%xmm0, %0 \n\t"
                         "movsd %%xmm7, %1"
                         :
                         "=m" (qr[0]),
                         "=m" (qr[1])
                         :
                         :
                         "xmm4", "xmm5", "xmm6", "xmm7");
}


void scl_qflt(double u,double *qr)
{
   double s;

   __asm__ __volatile__ ("movsd %1, %%xmm8 \n\t"
                         "mulsd %2, %%xmm8 \n\t"
                         "movsd %%xmm8, %0"
                         :
                         "=m" (s)
                         :
                         "m" (u),
                         "m" (qr[1])
                         :
                         "xmm8");

   exact_prod(u,qr[0],qr);
   acc_qflt(s,qr);
}


void mul_qflt(double *qu,double *qv,double *qr)
{
   double s;

   __asm__ __volatile__ ("movsd %1, %%xmm8 \n\t"
                         "movsd %2, %%xmm9 \n\t"
                         "mulsd %3, %%xmm8 \n\t"
                         "mulsd %4, %%xmm9 \n\t"
                         "addsd %%xmm9, %%xmm8 \n\t"
                         "movsd %%xmm8, %0"
                         :
                         "=m" (s)
                         :
                         "m" (qu[0]),
                         "m" (qu[1]),
                         "m" (qv[1]),
                         "m" (qv[0])
                         :
                         "xmm8", "xmm9");

   exact_prod(qu[0],qv[0],qr);
   acc_qflt(s,qr);
}

#else

void acc_qflt(double u,double *qr)
{
   double a,b,qp,up;
   double c,d;

   a=qr[0]+u;
   qp=a-u;
   up=a-qp;
   b=(qr[0]-qp)+(u-up);

   c=qr[1]+b;
   d=a+c;

   qr[0]=d;
   qr[1]=c-(d-a);
}


void add_qflt(double *qu,double *qv,double *qr)
{
   double a,b,up,vp;
   double c,d,rp,sp;
   double e,f,w;

   a=qu[0]+qv[0];
   c=qu[1]+qv[1];

   up=a-qv[0];
   rp=c-qv[1];
   vp=a-up;
   sp=c-rp;

   b=(qu[0]-up)+(qv[0]-vp);
   d=(qu[1]-rp)+(qv[1]-sp);

   b+=c;
   e=a+b;
   f=b-(e-a);

   f+=d;
   w=e+f;

   qr[0]=w;
   qr[1]=f-(w-e);
}


static void exact_prod(double u,double v,double *qr)
{
   double ux,uy,vx,vy;
   double uh,ul;
   double vh,vl;
   double x,y,c,d,e;

   ux=mm*u;
   vx=mm*v;
   uy=ux-u;
   vy=vx-v;
   uh=ux-uy;
   vh=vx-vy;
   ul=u-uh;
   vl=v-vh;

   x=u*v;
   c=uh*vh-x;
   d=uh*vl+c;
   e=ul*vh+d;
   y=ul*vl+e;

   qr[0]=x;
   qr[1]=y;
}


void scl_qflt(double u,double *qr)
{
   double s;

   s=u*qr[1];
   exact_prod(u,qr[0],qr);
   acc_qflt(s,qr);
}


void mul_qflt(double *qu,double *qv,double *qr)
{
   double s;

   s=(qu[0]*qv[1])+(qu[1]*qv[0]);
   exact_prod(qu[0],qv[0],qr);
   acc_qflt(s,qr);
}

#endif
