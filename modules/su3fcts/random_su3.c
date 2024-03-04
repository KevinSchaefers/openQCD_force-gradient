
/*******************************************************************************
*
* File random_su3.c
*
* Copyright (C) 2004, 2009, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generation of uniformly distributed single- and double-precision
* SU(3) matrices.
*
*   void random_su3(su3 *u)
*     Generates a random single-precision SU(3) matrix and assigns it to *u
*
*   void random_su3_dble(su3_dble *u)
*     Generates a random double-precision SU(3) matrix and assigns it to *u
*
* The random matrices are uniformly distributed over SU(3) to a precision
* given by the number of significant bits of the random numbers returned by
* ranlxs and ranlxd respectively. Rougly speaking one can expect the matrices
* to be uniformly random up to systematic deviations from 1 at the level of
* 10^(-7) and 10^(-14) in the single- and double-precision programs.
*
* The programs in this module do not perform any communications and are
* thread-safe.
*
*******************************************************************************/

#define RANDOM_SU3_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "su3.h"
#include "random.h"
#include "su3fcts.h"

typedef union
{
   su3_vector v;
   float r[6];
} vector_t;

typedef union
{
   su3_vector_dble v;
   double r[6];
} vector_dble_t;

typedef union
{
   su3 u;
   su3_vector v[3];
} matrix_t;

typedef union
{
   su3_dble u;
   su3_vector_dble v[3];
} matrix_dble_t;


static void random_su3_vector(su3_vector *v)
{
   float norm,fact,*r;
   vector_t *w;

   w=(vector_t*)(v);
   r=(*w).r;
   norm=0.0f;

   while (norm<=0.1f)
   {
      gauss(r,6);
      norm=r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+
           r[3]*r[3]+r[4]*r[4]+r[5]*r[5];
   }

   fact=(double)(1.0/sqrt((double)(norm)));

   r[0]*=fact;
   r[1]*=fact;
   r[2]*=fact;
   r[3]*=fact;
   r[4]*=fact;
   r[5]*=fact;
}


void random_su3(su3 *u)
{
   float norm,fact;
   su3_vector *v;
   matrix_t *m;

   m=(matrix_t*)(u);
   v=(*m).v;

   random_su3_vector(v);
   norm=0.0f;

   while (norm<=0.1f)
   {
      random_su3_vector(v+1);
      _vector_cross_prod(v[2],v[0],v[1]);
      norm=_vector_prod_re(v[2],v[2]);
   }

   fact=(double)(1.0/sqrt((double)(norm)));

   v[2].c1.re*=fact;
   v[2].c1.im*=fact;
   v[2].c2.re*=fact;
   v[2].c2.im*=fact;
   v[2].c3.re*=fact;
   v[2].c3.im*=fact;

   _vector_cross_prod(v[1],v[2],v[0]);
}


static void random_su3_vector_dble(su3_vector_dble *v)
{
   double norm,fact,*r;
   vector_dble_t *w;

   w=(vector_dble_t*)(v);
   r=(*w).r;
   norm=0.0;

   while (norm<=0.1)
   {
      gauss_dble(r,6);
      norm=r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+
           r[3]*r[3]+r[4]*r[4]+r[5]*r[5];
   }

   fact=1.0/sqrt(norm);

   r[0]*=fact;
   r[1]*=fact;
   r[2]*=fact;
   r[3]*=fact;
   r[4]*=fact;
   r[5]*=fact;
}


void random_su3_dble(su3_dble *u)
{
   double norm,fact;
   su3_vector_dble *v;
   matrix_dble_t *m;

   m=(matrix_dble_t*)(u);
   v=(*m).v;

   random_su3_vector_dble(v);
   norm=0.0;

   while (norm<=0.1)
   {
      random_su3_vector_dble(v+1);
      _vector_cross_prod(v[2],v[0],v[1]);
      norm=_vector_prod_re(v[2],v[2]);
   }

   fact=1.0/sqrt(norm);

   v[2].c1.re*=fact;
   v[2].c1.im*=fact;
   v[2].c2.re*=fact;
   v[2].c2.im*=fact;
   v[2].c3.re*=fact;
   v[2].c3.im*=fact;

   _vector_cross_prod(v[1],v[2],v[0]);
}
