
/*******************************************************************************
*
* File gauss.c
*
* Copyright (C) 2005, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generation of Gaussian random numbers.
*
*   void gauss(float *r,int n)
*     Generates n single-precision Gaussian random numbers x with distribution
*     proportional to exp(-x^2) and assigns them to r[0],..,r[n-1].
*
*   void gauss_dble(double *r,int n)
*     Generates n double-precision Gaussian random numbers x with distribution
*     proportional to exp(-x^2) and assigns them to r[0],..,r[n-1].
*
* The programs in this module are thread-safe and can be locally called.
*
*******************************************************************************/

#define GAUSS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "random.h"

static const double twopi=6.2831853071795865;


void gauss(float *r,int n)
{
   float s1,*rm;
   double rho,r0,r1;

   ranlxs(r,n);
   rm=r+n-(n&0x1);

   for (;r<rm;r+=2)
   {
      r0=(double)(r[0]);
      r1=(double)(r[1]-0.5);
      rho=-log(1.0-r0);
      rho=sqrt(rho);
      r1*=twopi;
      r[0]=(float)(rho*sin(r1));
      r[1]=(float)(rho*cos(r1));
   }

   if (n&0x1)
   {
      r0=(double)(r[0]);
      rho=-log(1.0-r[0]);
      rho=sqrt(rho);
      ranlxs(&s1,1);
      r1=(double)(s1-0.5);
      r[0]=(double)(rho*sin(twopi*r1));
   }
}


void gauss_dble(double *r,int n)
{
   double rho,r1,*rm;

   ranlxd(r,n);
   rm=r+n-(n&0x1);

   for (;r<rm;r+=2)
   {
      rho=-log(1.0-r[0]);
      rho=sqrt(rho);
      r[1]=twopi*(r[1]-0.5);
      r[0]=rho*sin(r[1]);
      r[1]=rho*cos(r[1]);
   }

   if (n&0x1)
   {
      rho=-log(1.0-r[0]);
      rho=sqrt(rho);
      ranlxd(&r1,1);
      r[0]=rho*sin(twopi*(r1-0.5));
   }
}
