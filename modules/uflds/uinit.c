
/*******************************************************************************
*
* File uinit.c
*
* Copyright (C) 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic assignment and initialization programs for arrays of SU(3) matrices.
*
*   void set_u2unity(int vol,int icom,su3 *u)
*     Sets the elements of the array u of single-precision SU(3) matrices
*     to unity.
*
*   void set_ud2unity(int vol,int icom,su3_dble *ud)
*     Sets the elements of the array ud of double-precision SU(3) matrices
*     to unity.
*
*   void assign_ud2ud(int vol,int icom,su3_dble *ud,su3_dble *vd)
*     Copies the array ud of double-precision SU(3) matrices to the
*     array vd.
*
* All these programs operate on arrays of SU(3) matrices, whose base address
* is passed through the arguments. The length of the arrays is specified by
* the parameter vol and the meaning of the parameter icom is explained in
* the file sflds/README.icom.
*
* Since no MPI communications are performed, all programs in this file can be
* called locally. If SSE2 instructions are used, the fields must be aligned
* to a 16 byte boundary.
*
*******************************************************************************/

#define UINIT_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "su3fcts.h"
#include "utils.h"
#include "uflds.h"
#include "global.h"


static void loc_set_u2unity(int vol,su3 *u)
{
   su3 u0={{0.0f,0.0f}};
   su3 *um;

   u0.c11.re=1.0f;
   u0.c22.re=1.0f;
   u0.c33.re=1.0f;

   um=u+vol;

   for (;u<um;u++)
      (*u)=u0;
}


void set_u2unity(int vol,int icom,su3 *u)
{
   int k;

   if (icom&0x2)
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_set_u2unity(vol,u+k*vol);
      }
   }
   else
      loc_set_u2unity(vol,u);
}


void set_ud2unity(int vol,int icom,su3_dble *ud)
{
   int k;

   if (icom&0x2)
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         cm3x3_unity(vol,ud+k*vol);
      }
   }
   else
      cm3x3_unity(vol,ud);
}


void assign_ud2ud(int vol,int icom,su3_dble *ud,su3_dble *vd)
{
   int k;

   if (icom&0x2)
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         cm3x3_assign(vol,ud+k*vol,vd+k*vol);
      }
   }
   else
      cm3x3_assign(vol,ud,vd);
}
