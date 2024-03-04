
/*******************************************************************************
*
* File salg_dble.c
*
* Copyright (C) 2005-2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic linear algebra routines for double-precision Dirac fields.
*
*   complex_qflt spinor_prod_dble(int vol,int icom,spinor_dble *s,
*                                 spinor_dble *r)
*     Returns the scalar product of the fields s and r.
*
*   qflt spinor_prod_re_dble(int vol,int icom,spinor_dble *s,
*                            spinor_dble *r)
*     Returns the real part of the scalar product of the fields s and r.
*
*   complex_qflt spinor_prod5_dble(int vol,int icom,spinor_dble *s,
*                                  spinor_dble *r)
*     Returns the scalar product of the fields s and gamma_5*r.
*
*   qflt norm_square_dble(int vol,int icom,spinor_dble *s)
*     Returns the square of the 2-norm of the field s.
*
*   void mulc_spinor_add_dble(int vol,int icom,spinor_dble *s,spinor_dble *r,
*                             complex_dble z)
*     Replaces the field s by s+z*r.
*
*   void mulr_spinor_add_dble(int vol,int icom,spinor_dble *s,spinor_dble *r,
*                             double c)
*     Replaces the field s by s+c*r.
*
*   void combine_spinor_dble(int vol,int icom,spinor_dble *s,spinor_dble *r,
*                            double cs,double cr)
*     Replaces the field s by cs*s+cr*r.
*
*   void scale_dble(int vol,int icom,double c,spinor_dble *s)
*     Replaces the field s by c*s.
*
*   void project_dble(int vol,int icom,spinor_dble *s,spinor_dble *r)
*     Replaces the field s by s-(r,s)*r.
*
*   double normalize_dble(int vol,int icom,spinor_dble *s)
*     Replaces the field s by s/||s|| and returns the 2-norm ||s||. The
*     division is omitted if the norm vanishes and an error occurs if
*     icom!=0 (if icom=0 it is up to the calling program to check that
*     the norm was non-zero).
*
*   void mulg5_dble(int vol,int icom,spinor_dble *s)
*     Multiplies the field s with gamma_5.
*
*   void mulmg5_dble(int vol,int icom,spinor_dble *s)
*     Multiplies the field s with -gamma_5.
*
* The quadruple-precision types qflt and complex_qflt are defined in su3.h.
* See doc/qsum.pdf for further explanations.
*
* All these programs act on arrays of spinor fields whose base address
* is passed through the arguments. The length of the arrays is specified
* by the parameter vol and the meaning of the parameter icom is explained
* in the file sflds/README.icom.
*
* If SSE (AVX) instructions are used, the spinor fields must be aligned
* to a 16 (32) byte boundary.
*
*******************************************************************************/

#define SALG_DBLE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "sflds.h"
#include "linalg.h"
#include "global.h"

#if (defined AVX)
#include "avx.h"

#if (defined FMA3)

static complex_qflt loc_spinor_prod_dble(int vol,spinor_dble *s,spinor_dble *r)
{
   double *qsm[2];
   complex_dble smz;
   complex_qflt cqsm;
   spinor_dble *sm,*smb;

   qsm[0]=cqsm.re.q;
   qsm[1]=cqsm.im.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   qsm[1][0]=0.0;
   qsm[1][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      __asm__ __volatile__ ("vxorpd %%ymm0, %%ymm0, %%ymm0 \n\t"
                            "vxorpd %%ymm1, %%ymm1, %%ymm1 \n\t"
                            "vxorpd %%ymm2, %%ymm2, %%ymm2 \n\t"
                            "vxorpd %%ymm3, %%ymm3, %%ymm3 \n\t"
                            "vxorpd %%ymm4, %%ymm4, %%ymm4 \n\t"
                            "vxorpd %%ymm5, %%ymm5, %%ymm5"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      for (;s<smb;s++)
      {
         __asm__ __volatile__ ("vmovapd %0, %%ymm6 \n\t"
                               "vmovapd %2, %%ymm7 \n\t"
                               "vmovapd %4, %%ymm8"
                               :
                               :
                               "m" ((*r).c1.c1),
                               "m" ((*r).c1.c2),
                               "m" ((*r).c1.c3),
                               "m" ((*r).c2.c1),
                               "m" ((*r).c2.c2),
                               "m" ((*r).c2.c3)
                               :
                               "xmm6", "xmm7", "xmm8");

         __asm__ __volatile__ ("vmovddup %0, %%ymm9 \n\t"
                               "vmovddup %2, %%ymm10 \n\t"
                               "vmovddup %4, %%ymm11"
                               :
                               :
                               "m" ((*s).c1.c1.re),
                               "m" ((*s).c1.c2.re),
                               "m" ((*s).c1.c3.re),
                               "m" ((*s).c2.c1.re),
                               "m" ((*s).c2.c2.re),
                               "m" ((*s).c2.c3.re)
                               :
                               "xmm9", "xmm10", "xmm11");

         __asm__ __volatile__ ("vmovddup %0, %%ymm12 \n\t"
                               "vmovddup %2, %%ymm13 \n\t"
                               "vmovddup %4, %%ymm14"
                               :
                               :
                               "m" ((*s).c1.c1.im),
                               "m" ((*s).c1.c2.im),
                               "m" ((*s).c1.c3.im),
                               "m" ((*s).c2.c1.im),
                               "m" ((*s).c2.c2.im),
                               "m" ((*s).c2.c3.im)
                               :
                               "xmm12", "xmm13", "xmm14");

         __asm__ __volatile__ ("vfmadd231pd %%ymm6, %%ymm9, %%ymm0 \n\t"
                               "vfmadd231pd %%ymm7, %%ymm10, %%ymm1 \n\t"
                               "vfmadd231pd %%ymm8, %%ymm11, %%ymm2 \n\t"
                               "vfnmadd231pd %%ymm6, %%ymm12, %%ymm3 \n\t"
                               "vfnmadd231pd %%ymm7, %%ymm13, %%ymm4 \n\t"
                               "vfnmadd231pd %%ymm8, %%ymm14, %%ymm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3",
                               "xmm4", "xmm5");

         __asm__ __volatile__ ("vmovapd %0, %%ymm6 \n\t"
                               "vmovapd %2, %%ymm7 \n\t"
                               "vmovapd %4, %%ymm8"
                               :
                               :
                               "m" ((*r).c3.c1),
                               "m" ((*r).c3.c2),
                               "m" ((*r).c3.c3),
                               "m" ((*r).c4.c1),
                               "m" ((*r).c4.c2),
                               "m" ((*r).c4.c3)
                               :
                               "xmm6", "xmm7", "xmm8");

         __asm__ __volatile__ ("vmovddup %0, %%ymm9 \n\t"
                               "vmovddup %2, %%ymm10 \n\t"
                               "vmovddup %4, %%ymm11"
                               :
                               :
                               "m" ((*s).c3.c1.re),
                               "m" ((*s).c3.c2.re),
                               "m" ((*s).c3.c3.re),
                               "m" ((*s).c4.c1.re),
                               "m" ((*s).c4.c2.re),
                               "m" ((*s).c4.c3.re)
                               :
                               "xmm9", "xmm10", "xmm11");

         __asm__ __volatile__ ("vmovddup %0, %%ymm12 \n\t"
                               "vmovddup %2, %%ymm13 \n\t"
                               "vmovddup %4, %%ymm14"
                               :
                               :
                               "m" ((*s).c3.c1.im),
                               "m" ((*s).c3.c2.im),
                               "m" ((*s).c3.c3.im),
                               "m" ((*s).c4.c1.im),
                               "m" ((*s).c4.c2.im),
                               "m" ((*s).c4.c3.im)
                               :
                               "xmm12", "xmm13", "xmm14");

         __asm__ __volatile__ ("vfmadd231pd %%ymm6, %%ymm9, %%ymm0 \n\t"
                               "vfmadd231pd %%ymm7, %%ymm10, %%ymm1 \n\t"
                               "vfmadd231pd %%ymm8, %%ymm11, %%ymm2 \n\t"
                               "vfnmadd231pd %%ymm6, %%ymm12, %%ymm3 \n\t"
                               "vfnmadd231pd %%ymm7, %%ymm13, %%ymm4 \n\t"
                               "vfnmadd231pd %%ymm8, %%ymm14, %%ymm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3",
                               "xmm4", "xmm5");

         r+=1;
      }

      __asm__ __volatile__ ("vaddpd %%ymm1, %%ymm0, %%ymm0 \n\t"
                            "vaddpd %%ymm4, %%ymm3, %%ymm3 \n\t"
                            "vaddpd %%ymm2, %%ymm0, %%ymm0 \n\t"
                            "vaddpd %%ymm5, %%ymm3, %%ymm3 \n\t"
                            "vpermilpd $0x5, %%ymm3, %%ymm3 \n\t"
                            "vaddsubpd %%ymm3, %%ymm0, %%ymm0 \n\t"
                            "vextractf128 $0x1, %%ymm0, %%xmm1 \n\t"
                            "vaddpd %%xmm1, %%xmm0, %%xmm0 \n\t"
                            "vmovupd %%xmm0, %0"
                            :
                            "=m" (smz)
                            :
                            :
                            "xmm0", "xmm1", "xmm3");

      _avx_zeroupper();

      acc_qflt(smz.re,qsm[0]);
      acc_qflt(smz.im,qsm[1]);
   }

   return cqsm;
}


static qflt loc_spinor_prod_re_dble(int vol,spinor_dble *s,spinor_dble *r)
{
   double smx,*qsm[1];
   qflt rqsm;
   spinor_dble *sm,*smb;

   qsm[0]=rqsm.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      __asm__ __volatile__ ("vxorpd %%ymm0, %%ymm0, %%ymm0 \n\t"
                            "vxorpd %%ymm1, %%ymm1, %%ymm1 \n\t"
                            "vxorpd %%ymm2, %%ymm2, %%ymm2"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2");

      for (;s<smb;s++)
      {
         __asm__ __volatile__ ("vmovapd %0, %%ymm3 \n\t"
                               "vmovapd %2, %%ymm4 \n\t"
                               "vmovapd %4, %%ymm5"
                               :
                               :
                               "m" ((*s).c1.c1),
                               "m" ((*s).c1.c2),
                               "m" ((*s).c1.c3),
                               "m" ((*s).c2.c1),
                               "m" ((*s).c2.c2),
                               "m" ((*s).c2.c3)
                               :
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("vmovapd %0, %%ymm6 \n\t"
                               "vmovapd %2, %%ymm7 \n\t"
                               "vmovapd %4, %%ymm8"
                               :
                               :
                               "m" ((*r).c1.c1),
                               "m" ((*r).c1.c2),
                               "m" ((*r).c1.c3),
                               "m" ((*r).c2.c1),
                               "m" ((*r).c2.c2),
                               "m" ((*r).c2.c3)
                               :
                               "xmm6", "xmm7", "xmm8");

         __asm__ __volatile__ ("vfmadd231pd %%ymm3, %%ymm6, %%ymm0 \n\t"
                               "vfmadd231pd %%ymm4, %%ymm7, %%ymm1 \n\t"
                               "vfmadd231pd %%ymm5, %%ymm8, %%ymm2"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2");

         __asm__ __volatile__ ("vmovapd %0, %%ymm9 \n\t"
                               "vmovapd %2, %%ymm10 \n\t"
                               "vmovapd %4, %%ymm11"
                               :
                               :
                               "m" ((*s).c3.c1),
                               "m" ((*s).c3.c2),
                               "m" ((*s).c3.c3),
                               "m" ((*s).c4.c1),
                               "m" ((*s).c4.c2),
                               "m" ((*s).c4.c3)
                               :
                               "xmm9", "xmm10", "xmm11");

         __asm__ __volatile__ ("vmovapd %0, %%ymm12 \n\t"
                               "vmovapd %2, %%ymm13 \n\t"
                               "vmovapd %4, %%ymm14"
                               :
                               :
                               "m" ((*r).c3.c1),
                               "m" ((*r).c3.c2),
                               "m" ((*r).c3.c3),
                               "m" ((*r).c4.c1),
                               "m" ((*r).c4.c2),
                               "m" ((*r).c4.c3)
                               :
                               "xmm12", "xmm13", "xmm14");

         __asm__ __volatile__ ("vfmadd231pd %%ymm9, %%ymm12, %%ymm0 \n\t"
                               "vfmadd231pd %%ymm10, %%ymm13, %%ymm1 \n\t"
                               "vfmadd231pd %%ymm11, %%ymm14, %%ymm2"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2");

         r+=1;
      }

      __asm__ __volatile__ ("vaddpd %%ymm1, %%ymm0, %%ymm0 \n\t"
                            "vaddpd %%ymm2, %%ymm0, %%ymm0 \n\t"
                            "vextractf128 $0x1, %%ymm0, %%xmm1 \n\t"
                            "vaddpd %%xmm1, %%xmm0, %%xmm0 \n\t"
                            "vhaddpd %%xmm0, %%xmm0, %%xmm2 \n\t"
                            "vmovsd %%xmm2, %0"
                            :
                            "=m" (smx)
                            :
                            :
                            "xmm0", "xmm1", "xmm2");

      _avx_zeroupper();

      acc_qflt(smx,qsm[0]);
   }

   return rqsm;
}


static complex_qflt loc_spinor_prod5_dble(int vol,spinor_dble *s,spinor_dble *r)
{
   double *qsm[2];
   complex_dble smz;
   complex_qflt cqsm;
   spinor_dble *sm,*smb;

   qsm[0]=cqsm.re.q;
   qsm[1]=cqsm.im.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   qsm[1][0]=0.0;
   qsm[1][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      __asm__ __volatile__ ("vxorpd %%ymm0, %%ymm0, %%ymm0 \n\t"
                            "vxorpd %%ymm1, %%ymm1, %%ymm1 \n\t"
                            "vxorpd %%ymm2, %%ymm2, %%ymm2 \n\t"
                            "vxorpd %%ymm3, %%ymm3, %%ymm3 \n\t"
                            "vxorpd %%ymm4, %%ymm4, %%ymm4 \n\t"
                            "vxorpd %%ymm5, %%ymm5, %%ymm5"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      for (;s<smb;s++)
      {
         __asm__ __volatile__ ("vmovapd %0, %%ymm6 \n\t"
                               "vmovapd %2, %%ymm7 \n\t"
                               "vmovapd %4, %%ymm8"
                               :
                               :
                               "m" ((*r).c1.c1),
                               "m" ((*r).c1.c2),
                               "m" ((*r).c1.c3),
                               "m" ((*r).c2.c1),
                               "m" ((*r).c2.c2),
                               "m" ((*r).c2.c3)
                               :
                               "xmm6", "xmm7", "xmm8");

         __asm__ __volatile__ ("vmovddup %0, %%ymm9 \n\t"
                               "vmovddup %2, %%ymm10 \n\t"
                               "vmovddup %4, %%ymm11"
                               :
                               :
                               "m" ((*s).c1.c1.re),
                               "m" ((*s).c1.c2.re),
                               "m" ((*s).c1.c3.re),
                               "m" ((*s).c2.c1.re),
                               "m" ((*s).c2.c2.re),
                               "m" ((*s).c2.c3.re)
                               :
                               "xmm9", "xmm10", "xmm11");

         __asm__ __volatile__ ("vmovddup %0, %%ymm12 \n\t"
                               "vmovddup %2, %%ymm13 \n\t"
                               "vmovddup %4, %%ymm14"
                               :
                               :
                               "m" ((*s).c1.c1.im),
                               "m" ((*s).c1.c2.im),
                               "m" ((*s).c1.c3.im),
                               "m" ((*s).c2.c1.im),
                               "m" ((*s).c2.c2.im),
                               "m" ((*s).c2.c3.im)
                               :
                               "xmm12", "xmm13", "xmm14");

         __asm__ __volatile__ ("vfmadd231pd %%ymm6, %%ymm9, %%ymm0 \n\t"
                               "vfmadd231pd %%ymm7, %%ymm10, %%ymm1 \n\t"
                               "vfmadd231pd %%ymm8, %%ymm11, %%ymm2 \n\t"
                               "vfnmadd231pd %%ymm6, %%ymm12, %%ymm3 \n\t"
                               "vfnmadd231pd %%ymm7, %%ymm13, %%ymm4 \n\t"
                               "vfnmadd231pd %%ymm8, %%ymm14, %%ymm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3",
                               "xmm4", "xmm5");

         __asm__ __volatile__ ("vmovapd %0, %%ymm6 \n\t"
                               "vmovapd %2, %%ymm7 \n\t"
                               "vmovapd %4, %%ymm8"
                               :
                               :
                               "m" ((*r).c3.c1),
                               "m" ((*r).c3.c2),
                               "m" ((*r).c3.c3),
                               "m" ((*r).c4.c1),
                               "m" ((*r).c4.c2),
                               "m" ((*r).c4.c3)
                               :
                               "xmm6", "xmm7", "xmm8");

         __asm__ __volatile__ ("vmovddup %0, %%ymm9 \n\t"
                               "vmovddup %2, %%ymm10 \n\t"
                               "vmovddup %4, %%ymm11"
                               :
                               :
                               "m" ((*s).c3.c1.re),
                               "m" ((*s).c3.c2.re),
                               "m" ((*s).c3.c3.re),
                               "m" ((*s).c4.c1.re),
                               "m" ((*s).c4.c2.re),
                               "m" ((*s).c4.c3.re)
                               :
                               "xmm9", "xmm10", "xmm11");

         __asm__ __volatile__ ("vmovddup %0, %%ymm12 \n\t"
                               "vmovddup %2, %%ymm13 \n\t"
                               "vmovddup %4, %%ymm14"
                               :
                               :
                               "m" ((*s).c3.c1.im),
                               "m" ((*s).c3.c2.im),
                               "m" ((*s).c3.c3.im),
                               "m" ((*s).c4.c1.im),
                               "m" ((*s).c4.c2.im),
                               "m" ((*s).c4.c3.im)
                               :
                               "xmm12", "xmm13", "xmm14");

         __asm__ __volatile__ ("vfnmadd231pd %%ymm6, %%ymm9, %%ymm0 \n\t"
                               "vfnmadd231pd %%ymm7, %%ymm10, %%ymm1 \n\t"
                               "vfnmadd231pd %%ymm8, %%ymm11, %%ymm2 \n\t"
                               "vfmadd231pd %%ymm6, %%ymm12, %%ymm3 \n\t"
                               "vfmadd231pd %%ymm7, %%ymm13, %%ymm4 \n\t"
                               "vfmadd231pd %%ymm8, %%ymm14, %%ymm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2", "xmm3",
                               "xmm4", "xmm5");

         r+=1;
      }

      __asm__ __volatile__ ("vaddpd %%ymm1, %%ymm0, %%ymm0 \n\t"
                            "vaddpd %%ymm4, %%ymm3, %%ymm3 \n\t"
                            "vaddpd %%ymm2, %%ymm0, %%ymm0 \n\t"
                            "vaddpd %%ymm5, %%ymm3, %%ymm3 \n\t"
                            "vpermilpd $0x5, %%ymm3, %%ymm3 \n\t"
                            "vaddsubpd %%ymm3, %%ymm0, %%ymm0 \n\t"
                            "vextractf128 $0x1, %%ymm0, %%xmm1 \n\t"
                            "vaddpd %%xmm1, %%xmm0, %%xmm0 \n\t"
                            "vmovupd %%xmm0, %0"
                            :
                            "=m" (smz)
                            :
                            :
                            "xmm0", "xmm1", "xmm3");

      _avx_zeroupper();

      acc_qflt(smz.re,qsm[0]);
      acc_qflt(smz.im,qsm[1]);
   }

   return cqsm;
}


static qflt loc_norm_square_dble(int vol,spinor_dble *s)
{
   double smx,*qsm[1];
   qflt rqsm;
   spinor_dble *sm,*smb;

   qsm[0]=rqsm.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      __asm__ __volatile__ ("vxorpd %%ymm6, %%ymm6, %%ymm6 \n\t"
                            "vxorpd %%ymm7, %%ymm7, %%ymm7 \n\t"
                            "vxorpd %%ymm8, %%ymm8, %%ymm8 \n\t"
                            "vxorpd %%ymm9, %%ymm9, %%ymm9 \n\t"
                            "vxorpd %%ymm10, %%ymm10, %%ymm10 \n\t"
                            "vxorpd %%ymm11, %%ymm11, %%ymm11"
                            :
                            :
                            :
                            "xmm6", "xmm7", "xmm8", "xmm9",
                            "xmm10", "xmm11");

      for (;s<smb;s++)
      {
         _avx_spinor_load_dble(*s);

         __asm__ __volatile__ ("vfmadd231pd %%ymm0, %%ymm0, %%ymm6 \n\t"
                               "vfmadd231pd %%ymm1, %%ymm1, %%ymm7 \n\t"
                               "vfmadd231pd %%ymm2, %%ymm2, %%ymm8 \n\t"
                               "vfmadd231pd %%ymm3, %%ymm3, %%ymm9 \n\t"
                               "vfmadd231pd %%ymm4, %%ymm4, %%ymm10 \n\t"
                               "vfmadd231pd %%ymm5, %%ymm5, %%ymm11 \n\t"
                               :
                               :
                               :
                               "xmm6", "xmm7", "xmm8", "xmm9",
                               "xmm10", "xmm11");

      }

      __asm__ __volatile__ ("vaddpd %%ymm6, %%ymm7, %%ymm7 \n\t"
                            "vaddpd %%ymm8, %%ymm9, %%ymm9 \n\t"
                            "vaddpd %%ymm10, %%ymm11, %%ymm11 \n\t"
                            "vaddpd %%ymm7, %%ymm9, %%ymm9 \n\t"
                            "vaddpd %%ymm9, %%ymm11, %%ymm11 \n\t"
                            "vextractf128 $0x1, %%ymm11, %%xmm12 \n\t"
                            "vaddpd %%xmm11, %%xmm12, %%xmm12 \n\t"
                            "vhaddpd %%xmm12, %%xmm12, %%xmm13 \n\t"
                            "vmovsd %%xmm13, %0 \n\t"
                            :
                            "=m" (smx)
                            :
                            :
                            "xmm0", "xmm1", "xmm2",
                            "xmm7", "xmm9", "xmm11");

      _avx_zeroupper();

      acc_qflt(smx,qsm[0]);
   }

   return rqsm;
}

#else

static complex_qflt loc_spinor_prod_dble(int vol,spinor_dble *s,spinor_dble *r)
{
   double *qsm[2];
   complex_dble smz;
   complex_qflt cqsm;
   spinor_dble *sm,*smb;

   qsm[0]=cqsm.re.q;
   qsm[1]=cqsm.im.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   qsm[1][0]=0.0;
   qsm[1][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      __asm__ __volatile__ ("vxorpd %%ymm0, %%ymm0, %%ymm0 \n\t"
                            "vxorpd %%ymm1, %%ymm1, %%ymm1 \n\t"
                            "vxorpd %%ymm2, %%ymm2, %%ymm2 \n\t"
                            "vxorpd %%ymm3, %%ymm3, %%ymm3 \n\t"
                            "vxorpd %%ymm4, %%ymm4, %%ymm4 \n\t"
                            "vxorpd %%ymm5, %%ymm5, %%ymm5"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2",
                            "xmm3", "xmm4", "xmm5");

      for (;s<smb;s++)
      {
         __asm__ __volatile__ ("vmovapd %0, %%ymm6 \n\t"
                               "vmovapd %2, %%ymm7 \n\t"
                               "vmovapd %4, %%ymm8"
                               :
                               :
                               "m" ((*s).c1.c1),
                               "m" ((*s).c1.c2),
                               "m" ((*s).c1.c3),
                               "m" ((*s).c2.c1),
                               "m" ((*s).c2.c2),
                               "m" ((*s).c2.c3)
                               :
                               "xmm6", "xmm7", "xmm8");

         __asm__ __volatile__ ("vmovapd %0, %%ymm12 \n\t"
                               "vmovapd %2, %%ymm13 \n\t"
                               "vmovapd %4, %%ymm14"
                               :
                               :
                               "m" ((*r).c1.c1),
                               "m" ((*r).c1.c2),
                               "m" ((*r).c1.c3),
                               "m" ((*r).c2.c1),
                               "m" ((*r).c2.c2),
                               "m" ((*r).c2.c3)
                               :
                               "xmm12", "xmm13", "xmm14");

         __asm__ __volatile__ ("vpermilpd $0x5, %%ymm6, %%ymm9 \n\t"
                               "vpermilpd $0x5, %%ymm7, %%ymm10 \n\t"
                               "vpermilpd $0x5, %%ymm8, %%ymm11 \n\t"
                               "vmulpd %%ymm12, %%ymm6, %%ymm6 \n\t"
                               "vmulpd %%ymm13, %%ymm7, %%ymm7 \n\t"
                               "vmulpd %%ymm14, %%ymm8, %%ymm8 \n\t"
                               "vmulpd %%ymm12, %%ymm9, %%ymm9 \n\t"
                               "vmulpd %%ymm13, %%ymm10, %%ymm10 \n\t"
                               "vmulpd %%ymm14, %%ymm11, %%ymm11 \n\t"
                               "vaddpd %%ymm6, %%ymm0, %%ymm0 \n\t"
                               "vaddpd %%ymm7, %%ymm1, %%ymm1 \n\t"
                               "vaddpd %%ymm8, %%ymm2, %%ymm2 \n\t"
                               "vaddsubpd %%ymm9, %%ymm3, %%ymm3 \n\t"
                               "vaddsubpd %%ymm10, %%ymm4, %%ymm4 \n\t"
                               "vaddsubpd %%ymm11, %%ymm5, %%ymm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5",
                               "xmm6", "xmm7", "xmm8",
                               "xmm9", "xmm10", "xmm11");

         __asm__ __volatile__ ("vmovapd %0, %%ymm6 \n\t"
                               "vmovapd %2, %%ymm7 \n\t"
                               "vmovapd %4, %%ymm8"
                               :
                               :
                               "m" ((*s).c3.c1),
                               "m" ((*s).c3.c2),
                               "m" ((*s).c3.c3),
                               "m" ((*s).c4.c1),
                               "m" ((*s).c4.c2),
                               "m" ((*s).c4.c3)
                               :
                               "xmm6", "xmm7", "xmm8");

         __asm__ __volatile__ ("vmovapd %0, %%ymm12 \n\t"
                               "vmovapd %2, %%ymm13 \n\t"
                               "vmovapd %4, %%ymm14"
                               :
                               :
                               "m" ((*r).c3.c1),
                               "m" ((*r).c3.c2),
                               "m" ((*r).c3.c3),
                               "m" ((*r).c4.c1),
                               "m" ((*r).c4.c2),
                               "m" ((*r).c4.c3)
                               :
                               "xmm12", "xmm13", "xmm14");

         __asm__ __volatile__ ("vpermilpd $0x5, %%ymm6, %%ymm9 \n\t"
                               "vpermilpd $0x5, %%ymm7, %%ymm10 \n\t"
                               "vpermilpd $0x5, %%ymm8, %%ymm11 \n\t"
                               "vmulpd %%ymm12, %%ymm6, %%ymm6 \n\t"
                               "vmulpd %%ymm13, %%ymm7, %%ymm7 \n\t"
                               "vmulpd %%ymm14, %%ymm8, %%ymm8 \n\t"
                               "vmulpd %%ymm12, %%ymm9, %%ymm9 \n\t"
                               "vmulpd %%ymm13, %%ymm10, %%ymm10 \n\t"
                               "vmulpd %%ymm14, %%ymm11, %%ymm11 \n\t"
                               "vaddpd %%ymm6, %%ymm0, %%ymm0 \n\t"
                               "vaddpd %%ymm7, %%ymm1, %%ymm1 \n\t"
                               "vaddpd %%ymm8, %%ymm2, %%ymm2 \n\t"
                               "vaddsubpd %%ymm9, %%ymm3, %%ymm3 \n\t"
                               "vaddsubpd %%ymm10, %%ymm4, %%ymm4 \n\t"
                               "vaddsubpd %%ymm11, %%ymm5, %%ymm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5",
                               "xmm6", "xmm7", "xmm8",
                               "xmm9", "xmm10", "xmm11");

         r+=1;
      }

      __asm__ __volatile__ ("vaddpd %%ymm0, %%ymm2, %%ymm2 \n\t"
                            "vaddpd %%ymm3, %%ymm5, %%ymm5 \n\t"
                            "vaddpd %%ymm1, %%ymm2, %%ymm2 \n\t"
                            "vaddpd %%ymm4, %%ymm5, %%ymm5 \n\t"
                            "vhaddpd %%ymm5, %%ymm2, %%ymm0 \n\t"
                            "vextractf128 $0x1, %%ymm0, %%xmm1 \n\t"
                            "vaddpd %%xmm0, %%xmm1, %%xmm2 \n\t"
                            "vmovupd %%xmm2, %0 \n\t"
                            "vzeroupper"
                            :
                            "=m" (smz)
                            :
                            :
                            "xmm0", "xmm1", "xmm2",
                            "xmm5");

      acc_qflt(smz.re,qsm[0]);
      acc_qflt(smz.im,qsm[1]);
   }

   return cqsm;
}


static qflt loc_spinor_prod_re_dble(int vol,spinor_dble *s,spinor_dble *r)
{
   double smx,*qsm[1];
   qflt rqsm;
   spinor_dble *sm,*smb;

   qsm[0]=rqsm.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      __asm__ __volatile__ ("vxorpd %%ymm0, %%ymm0, %%ymm0 \n\t"
                            "vxorpd %%ymm1, %%ymm1, %%ymm1 \n\t"
                            "vxorpd %%ymm2, %%ymm2, %%ymm2 \n\t"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2");

      for (;s<smb;s++)
      {
         __asm__ __volatile__ ("vmovapd %0, %%ymm3 \n\t"
                               "vmovapd %2, %%ymm4 \n\t"
                               "vmovapd %4, %%ymm5"
                               :
                               :
                               "m" ((*s).c1.c1),
                               "m" ((*s).c1.c2),
                               "m" ((*s).c1.c3),
                               "m" ((*s).c2.c1),
                               "m" ((*s).c2.c2),
                               "m" ((*s).c2.c3)
                               :
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("vmovapd %0, %%ymm6 \n\t"
                               "vmovapd %2, %%ymm7 \n\t"
                               "vmovapd %4, %%ymm8"
                               :
                               :
                               "m" ((*r).c1.c1),
                               "m" ((*r).c1.c2),
                               "m" ((*r).c1.c3),
                               "m" ((*r).c2.c1),
                               "m" ((*r).c2.c2),
                               "m" ((*r).c2.c3)
                               :
                               "xmm6", "xmm7", "xmm8");

         __asm__ __volatile__ ("vmulpd %%ymm3, %%ymm6, %%ymm6 \n\t"
                               "vmulpd %%ymm4, %%ymm7, %%ymm7 \n\t"
                               "vmulpd %%ymm5, %%ymm8, %%ymm8 \n\t"
                               "vaddpd %%ymm6, %%ymm0, %%ymm0 \n\t"
                               "vaddpd %%ymm7, %%ymm1, %%ymm1 \n\t"
                               "vaddpd %%ymm8, %%ymm2, %%ymm2 \n\t"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm6", "xmm7", "xmm8");

         __asm__ __volatile__ ("vmovapd %0, %%ymm3 \n\t"
                               "vmovapd %2, %%ymm4 \n\t"
                               "vmovapd %4, %%ymm5"
                               :
                               :
                               "m" ((*s).c3.c1),
                               "m" ((*s).c3.c2),
                               "m" ((*s).c3.c3),
                               "m" ((*s).c4.c1),
                               "m" ((*s).c4.c2),
                               "m" ((*s).c4.c3)
                               :
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("vmovapd %0, %%ymm6 \n\t"
                               "vmovapd %2, %%ymm7 \n\t"
                               "vmovapd %4, %%ymm8"
                               :
                               :
                               "m" ((*r).c3.c1),
                               "m" ((*r).c3.c2),
                               "m" ((*r).c3.c3),
                               "m" ((*r).c4.c1),
                               "m" ((*r).c4.c2),
                               "m" ((*r).c4.c3)
                               :
                               "xmm6", "xmm7", "xmm8");

         __asm__ __volatile__ ("vmulpd %%ymm3, %%ymm6, %%ymm6 \n\t"
                               "vmulpd %%ymm4, %%ymm7, %%ymm7 \n\t"
                               "vmulpd %%ymm5, %%ymm8, %%ymm8 \n\t"
                               "vaddpd %%ymm6, %%ymm0, %%ymm0 \n\t"
                               "vaddpd %%ymm7, %%ymm1, %%ymm1 \n\t"
                               "vaddpd %%ymm8, %%ymm2, %%ymm2 \n\t"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm6", "xmm7", "xmm8");

         r+=1;
      }

      __asm__ __volatile__ ("vaddpd %%ymm0, %%ymm2, %%ymm2 \n\t"
                            "vaddpd %%ymm1, %%ymm2, %%ymm2 \n\t"
                            "vextractf128 $0x1, %%ymm2, %%xmm1 \n\t"
                            "vaddpd %%xmm1, %%xmm2, %%xmm2 \n\t"
                            "vhaddpd %%xmm2, %%xmm2, %%xmm0 \n\t"
                            "vmovsd %%xmm0, %0 \n\t"
                            "vzeroupper"
                            :
                            "=m" (smx)
                            :
                            :
                            "xmm0", "xmm1", "xmm2");

      acc_qflt(smx,qsm[0]);
   }

   return rqsm;
}


static complex_qflt loc_spinor_prod5_dble(int vol,spinor_dble *s,spinor_dble *r)
{
   double *qsm[2];
   complex_dble smz;
   complex_qflt cqsm;
   spinor_dble *sm,*smb;

   qsm[0]=cqsm.re.q;
   qsm[1]=cqsm.im.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   qsm[1][0]=0.0;
   qsm[1][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      __asm__ __volatile__ ("vxorpd %%ymm0, %%ymm0, %%ymm0 \n\t"
                            "vxorpd %%ymm1, %%ymm1, %%ymm1 \n\t"
                            "vxorpd %%ymm2, %%ymm2, %%ymm2 \n\t"
                            "vxorpd %%ymm3, %%ymm3, %%ymm3 \n\t"
                            "vxorpd %%ymm4, %%ymm4, %%ymm4 \n\t"
                            "vxorpd %%ymm5, %%ymm5, %%ymm5"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2",
                            "xmm3", "xmm4", "xmm5");

      for (;s<smb;s++)
      {
         __asm__ __volatile__ ("vmovapd %0, %%ymm6 \n\t"
                               "vmovapd %2, %%ymm7 \n\t"
                               "vmovapd %4, %%ymm8"
                               :
                               :
                               "m" ((*s).c1.c1),
                               "m" ((*s).c1.c2),
                               "m" ((*s).c1.c3),
                               "m" ((*s).c2.c1),
                               "m" ((*s).c2.c2),
                               "m" ((*s).c2.c3)
                               :
                               "xmm6", "xmm7", "xmm8");

         __asm__ __volatile__ ("vmovapd %0, %%ymm12 \n\t"
                               "vmovapd %2, %%ymm13 \n\t"
                               "vmovapd %4, %%ymm14"
                               :
                               :
                               "m" ((*r).c1.c1),
                               "m" ((*r).c1.c2),
                               "m" ((*r).c1.c3),
                               "m" ((*r).c2.c1),
                               "m" ((*r).c2.c2),
                               "m" ((*r).c2.c3)
                               :
                               "xmm12", "xmm13", "xmm14");

         __asm__ __volatile__ ("vpermilpd $0x5, %%ymm6, %%ymm9 \n\t"
                               "vpermilpd $0x5, %%ymm7, %%ymm10 \n\t"
                               "vpermilpd $0x5, %%ymm8, %%ymm11 \n\t"
                               "vmulpd %%ymm12, %%ymm6, %%ymm6 \n\t"
                               "vmulpd %%ymm13, %%ymm7, %%ymm7 \n\t"
                               "vmulpd %%ymm14, %%ymm8, %%ymm8 \n\t"
                               "vmulpd %%ymm12, %%ymm9, %%ymm9 \n\t"
                               "vmulpd %%ymm13, %%ymm10, %%ymm10 \n\t"
                               "vmulpd %%ymm14, %%ymm11, %%ymm11 \n\t"
                               "vaddpd %%ymm6, %%ymm0, %%ymm0 \n\t"
                               "vaddpd %%ymm7, %%ymm1, %%ymm1 \n\t"
                               "vaddpd %%ymm8, %%ymm2, %%ymm2 \n\t"
                               "vaddsubpd %%ymm9, %%ymm3, %%ymm3 \n\t"
                               "vaddsubpd %%ymm10, %%ymm4, %%ymm4 \n\t"
                               "vaddsubpd %%ymm11, %%ymm5, %%ymm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5",
                               "xmm6", "xmm7", "xmm8",
                               "xmm9", "xmm10", "xmm11");

         __asm__ __volatile__ ("vmovapd %0, %%ymm6 \n\t"
                               "vmovapd %2, %%ymm7 \n\t"
                               "vmovapd %4, %%ymm8"
                               :
                               :
                               "m" ((*r).c3.c1),
                               "m" ((*r).c3.c2),
                               "m" ((*r).c3.c3),
                               "m" ((*r).c4.c1),
                               "m" ((*r).c4.c2),
                               "m" ((*r).c4.c3)
                               :
                               "xmm6", "xmm7", "xmm8");

         __asm__ __volatile__ ("vmovapd %0, %%ymm12 \n\t"
                               "vmovapd %2, %%ymm13 \n\t"
                               "vmovapd %4, %%ymm14"
                               :
                               :
                               "m" ((*s).c3.c1),
                               "m" ((*s).c3.c2),
                               "m" ((*s).c3.c3),
                               "m" ((*s).c4.c1),
                               "m" ((*s).c4.c2),
                               "m" ((*s).c4.c3)
                               :
                               "xmm12", "xmm13", "xmm14");

         __asm__ __volatile__ ("vpermilpd $0x5, %%ymm6, %%ymm9 \n\t"
                               "vpermilpd $0x5, %%ymm7, %%ymm10 \n\t"
                               "vpermilpd $0x5, %%ymm8, %%ymm11 \n\t"
                               "vmulpd %%ymm12, %%ymm6, %%ymm6 \n\t"
                               "vmulpd %%ymm13, %%ymm7, %%ymm7 \n\t"
                               "vmulpd %%ymm14, %%ymm8, %%ymm8 \n\t"
                               "vmulpd %%ymm12, %%ymm9, %%ymm9 \n\t"
                               "vmulpd %%ymm13, %%ymm10, %%ymm10 \n\t"
                               "vmulpd %%ymm14, %%ymm11, %%ymm11 \n\t"
                               "vsubpd %%ymm6, %%ymm0, %%ymm0 \n\t"
                               "vsubpd %%ymm7, %%ymm1, %%ymm1 \n\t"
                               "vsubpd %%ymm8, %%ymm2, %%ymm2 \n\t"
                               "vaddsubpd %%ymm9, %%ymm3, %%ymm3 \n\t"
                               "vaddsubpd %%ymm10, %%ymm4, %%ymm4 \n\t"
                               "vaddsubpd %%ymm11, %%ymm5, %%ymm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5",
                               "xmm6", "xmm7", "xmm8",
                               "xmm9", "xmm10", "xmm11");

         r+=1;
      }

      __asm__ __volatile__ ("vaddpd %%ymm0, %%ymm2, %%ymm2 \n\t"
                            "vaddpd %%ymm3, %%ymm5, %%ymm5 \n\t"
                            "vaddpd %%ymm1, %%ymm2, %%ymm2 \n\t"
                            "vaddpd %%ymm4, %%ymm5, %%ymm5 \n\t"
                            "vhaddpd %%ymm5, %%ymm2, %%ymm0 \n\t"
                            "vextractf128 $0x1, %%ymm0, %%xmm1 \n\t"
                            "vaddpd %%xmm0, %%xmm1, %%xmm2 \n\t"
                            "vmovupd %%xmm2, %0 \n\t"
                            "vzeroupper"
                            :
                            "=m" (smz)
                            :
                            :
                            "xmm0", "xmm1", "xmm2",
                            "xmm5");

      acc_qflt(smz.re,qsm[0]);
      acc_qflt(smz.im,qsm[1]);
   }

   return cqsm;
}


static qflt loc_norm_square_dble(int vol,spinor_dble *s)
{
   double smx,*qsm[1];
   qflt rqsm;
   spinor_dble *sm,*smb;

   qsm[0]=rqsm.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      __asm__ __volatile__ ("vxorpd %%ymm6, %%ymm6, %%ymm6 \n\t"
                            "vxorpd %%ymm7, %%ymm7, %%ymm7 \n\t"
                            "vxorpd %%ymm8, %%ymm8, %%ymm8 \n\t"
                            "vxorpd %%ymm9, %%ymm9, %%ymm9 \n\t"
                            "vxorpd %%ymm10, %%ymm10, %%ymm10 \n\t"
                            "vxorpd %%ymm11, %%ymm11, %%ymm11"
                            :
                            :
                            :
                            "xmm6", "xmm7", "xmm8",
                            "xmm9", "xmm10", "xmm11");

      for (;s<smb;s++)
      {
         _avx_spinor_load_dble(*s);

         __asm__ __volatile__ ("vmulpd %%ymm0, %%ymm0, %%ymm0 \n\t"
                               "vmulpd %%ymm1, %%ymm1, %%ymm1 \n\t"
                               "vmulpd %%ymm2, %%ymm2, %%ymm2 \n\t"
                               "vmulpd %%ymm3, %%ymm3, %%ymm3 \n\t"
                               "vmulpd %%ymm4, %%ymm4, %%ymm4 \n\t"
                               "vmulpd %%ymm5, %%ymm5, %%ymm5 \n\t"
                               "vaddpd %%ymm0, %%ymm6, %%ymm6 \n\t"
                               "vaddpd %%ymm1, %%ymm7, %%ymm7 \n\t"
                               "vaddpd %%ymm2, %%ymm8, %%ymm8 \n\t"
                               "vaddpd %%ymm3, %%ymm9, %%ymm9 \n\t"
                               "vaddpd %%ymm4, %%ymm10, %%ymm10 \n\t"
                               "vaddpd %%ymm5, %%ymm11, %%ymm11"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5",
                               "xmm6", "xmm7", "xmm8",
                               "xmm9", "xmm10", "xmm11");

      }

      __asm__ __volatile__ ("vaddpd %%ymm6, %%ymm7, %%ymm7 \n\t"
                            "vaddpd %%ymm8, %%ymm9, %%ymm9 \n\t"
                            "vaddpd %%ymm10, %%ymm11, %%ymm11 \n\t"
                            "vaddpd %%ymm7, %%ymm9, %%ymm9 \n\t"
                            "vaddpd %%ymm9, %%ymm11, %%ymm11 \n\t"
                            "vextractf128 $0x1, %%ymm11, %%xmm0 \n\t"
                            "vaddpd %%xmm11, %%xmm0, %%xmm1 \n\t"
                            "vhaddpd %%xmm1, %%xmm1, %%xmm2 \n\t"
                            "vmovsd %%xmm2, %0 \n\t"
                            "vzeroupper"
                            :
                            "=m" (smx)
                            :
                            :
                            "xmm0", "xmm1", "xmm2",
                            "xmm7", "xmm9", "xmm11");

      acc_qflt(smx,qsm[0]);
   }

   return rqsm;
}

#endif

static void loc_mulc_spinor_add_dble(int vol,spinor_dble *s,spinor_dble *r,
                                     complex_dble z)
{
   spinor_dble *sm;

   _avx_load_cmplx_up_dble(z);
   sm=s+vol;

   for (;s<sm;s++)
   {
      _avx_spinor_load_dble(*s);
      _avx_mulc_spinor_add_dble(*r);
      _avx_spinor_store_dble(*s);

      r+=1;
   }

   _avx_zeroupper();
}


static void loc_mulr_spinor_add_dble(int vol,spinor_dble *s,spinor_dble *r,
                                     double c)
{
   spinor_dble *sm;

   _avx_load_real_up_dble(c);
   sm=s+vol;

   for (;s<sm;s++)
   {
      _avx_spinor_load_dble(*s);
      _avx_mulr_spinor_add_dble(*r);
      _avx_spinor_store_dble(*s);

      r+=1;
   }

   _avx_zeroupper();
}


static void loc_combine_spinor_dble(int vol,spinor_dble *s,spinor_dble *r,
                                    double cs,double cr)
{
   spinor_dble *sm;

   _avx_load_real_dble(cs);
   _avx_load_real_up_dble(cr);
   sm=s+vol;

   for (;s<sm;s++)
   {
      _avx_mulr_spinor_dble(*s);
      _avx_mulr_spinor_add_dble(*r);
      _avx_spinor_store_dble(*s);

      r+=1;
   }

   _avx_zeroupper();
}


static void loc_scale_dble(int vol,double c,spinor_dble *s)
{
   spinor_dble *sm;

   _avx_load_real_dble(c);
   sm=s+vol;

   for (;s<sm;s++)
   {
      _avx_mulr_spinor_dble(*s);
      _avx_spinor_store_dble(*s);
   }

   _avx_zeroupper();
}


static void loc_mulg5_dble(int vol,spinor_dble *s)
{
   spinor_dble *sm;

   __asm__ __volatile__ ("vxorpd %%ymm3, %%ymm3, %%ymm3 \n\t"
                         "vxorpd %%ymm4, %%ymm4, %%ymm4 \n\t"
                         "vxorpd %%ymm5, %%ymm5, %%ymm5"
                         :
                         :
                         :
                         "xmm3", "xmm4", "xmm5");

   sm=s+vol;

   for (;s<sm;s++)
   {
      __asm__ __volatile__ ("vmovapd %0, %%ymm0 \n\t"
                            "vmovapd %2, %%ymm1 \n\t"
                            "vmovapd %4, %%ymm2 \n\t"
                            "vsubpd %%ymm0, %%ymm3, %%ymm0 \n\t"
                            "vsubpd %%ymm1, %%ymm4, %%ymm1 \n\t"
                            "vsubpd %%ymm2, %%ymm5, %%ymm2"
                            :
                            :
                            "m" ((*s).c3.c1),
                            "m" ((*s).c3.c2),
                            "m" ((*s).c3.c3),
                            "m" ((*s).c4.c1),
                            "m" ((*s).c4.c2),
                            "m" ((*s).c4.c3)
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("vmovapd %%ymm0, %0 \n\t"
                            "vmovapd %%ymm1, %2 \n\t"
                            "vmovapd %%ymm2, %4"
                            :
                            "=m" ((*s).c3.c1),
                            "=m" ((*s).c3.c2),
                            "=m" ((*s).c3.c3),
                            "=m" ((*s).c4.c1),
                            "=m" ((*s).c4.c2),
                            "=m" ((*s).c4.c3));
   }

   _avx_zeroupper();
}


static void loc_mulmg5_dble(int vol,spinor_dble *s)
{
   spinor_dble *sm;

   __asm__ __volatile__ ("vxorpd %%ymm3, %%ymm3, %%ymm3 \n\t"
                         "vxorpd %%ymm4, %%ymm4, %%ymm4 \n\t"
                         "vxorpd %%ymm5, %%ymm5, %%ymm5"
                         :
                         :
                         :
                         "xmm3", "xmm4", "xmm5");

   sm=s+vol;

   for (;s<sm;s++)
   {
      __asm__ __volatile__ ("vmovapd %0, %%ymm0 \n\t"
                            "vmovapd %2, %%ymm1 \n\t"
                            "vmovapd %4, %%ymm2 \n\t"
                            "vsubpd %%ymm0, %%ymm3, %%ymm0 \n\t"
                            "vsubpd %%ymm1, %%ymm4, %%ymm1 \n\t"
                            "vsubpd %%ymm2, %%ymm5, %%ymm2"
                            :
                            :
                            "m" ((*s).c1.c1),
                            "m" ((*s).c1.c2),
                            "m" ((*s).c1.c3),
                            "m" ((*s).c2.c1),
                            "m" ((*s).c2.c2),
                            "m" ((*s).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("vmovapd %%ymm0, %0 \n\t"
                            "vmovapd %%ymm1, %2 \n\t"
                            "vmovapd %%ymm2, %4"
                            :
                            "=m" ((*s).c1.c1),
                            "=m" ((*s).c1.c2),
                            "=m" ((*s).c1.c3),
                            "=m" ((*s).c2.c1),
                            "=m" ((*s).c2.c2),
                            "=m" ((*s).c2.c3));
   }

   _avx_zeroupper();
}

#elif (defined x64)
#include "sse2.h"

static complex_qflt loc_spinor_prod_dble(int vol,spinor_dble *s,spinor_dble *r)
{
   double *qsm[2];
   complex_dble smz;
   complex_qflt cqsm;
   spinor_dble *sm,*smb;

   qsm[0]=cqsm.re.q;
   qsm[1]=cqsm.im.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   qsm[1][0]=0.0;
   qsm[1][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      __asm__ __volatile__ ("xorpd %%xmm6, %%xmm6 \n\t"
                            "xorpd %%xmm7, %%xmm7 \n\t"
                            "xorpd %%xmm8, %%xmm8 \n\t"
                            "xorpd %%xmm9, %%xmm9 \n\t"
                            "xorpd %%xmm10, %%xmm10 \n\t"
                            "xorpd %%xmm11, %%xmm11"
                            :
                            :
                            :
                            "xmm6", "xmm7", "xmm8",
                            "xmm9", "xmm10", "xmm11");

      for (;s<smb;s++)
      {
         _sse_load_dble((*s).c1);
         _sse_load_up_dble((*s).c2);

         __asm__ __volatile__ ("mulpd %0, %%xmm0 \n\t"
                               "mulpd %1, %%xmm1 \n\t"
                               "mulpd %2, %%xmm2 \n\t"
                               "mulpd %3, %%xmm3 \n\t"
                               "mulpd %4, %%xmm4 \n\t"
                               "mulpd %5, %%xmm5"
                               :
                               :
                               "m" ((*r).c1.c1),
                               "m" ((*r).c1.c2),
                               "m" ((*r).c1.c3),
                               "m" ((*r).c2.c1),
                               "m" ((*r).c2.c2),
                               "m" ((*r).c2.c3)
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("addpd %%xmm0, %%xmm1 \n\t"
                               "addpd %%xmm2, %%xmm3 \n\t"
                               "addpd %%xmm4, %%xmm5 \n\t"
                               "addpd %%xmm1, %%xmm6 \n\t"
                               "addpd %%xmm3, %%xmm7 \n\t"
                               "addpd %%xmm5, %%xmm8"
                               :
                               :
                               :
                               "xmm1", "xmm3", "xmm5",
                               "xmm6", "xmm7", "xmm8");

         _sse_load_dble((*s).c3);
         _sse_load_up_dble((*s).c4);

         __asm__ __volatile__ ("mulpd %0, %%xmm0 \n\t"
                               "mulpd %1, %%xmm1 \n\t"
                               "mulpd %2, %%xmm2 \n\t"
                               "mulpd %3, %%xmm3 \n\t"
                               "mulpd %4, %%xmm4 \n\t"
                               "mulpd %5, %%xmm5"
                               :
                               :
                               "m" ((*r).c3.c1),
                               "m" ((*r).c3.c2),
                               "m" ((*r).c3.c3),
                               "m" ((*r).c4.c1),
                               "m" ((*r).c4.c2),
                               "m" ((*r).c4.c3)
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("addpd %%xmm0, %%xmm1 \n\t"
                               "addpd %%xmm2, %%xmm3 \n\t"
                               "addpd %%xmm4, %%xmm5 \n\t"
                               "addpd %%xmm1, %%xmm6 \n\t"
                               "addpd %%xmm3, %%xmm7 \n\t"
                               "addpd %%xmm5, %%xmm8"
                               :
                               :
                               :
                               "xmm1", "xmm3", "xmm5",
                               "xmm6", "xmm7", "xmm8");

         _sse_load_dble((*s).c1);
         _sse_load_up_dble((*s).c2);

         __asm__ __volatile__ ("shufpd $0x1, %%xmm0, %%xmm0 \n\t"
                               "shufpd $0x1, %%xmm1, %%xmm1 \n\t"
                               "shufpd $0x1, %%xmm2, %%xmm2 \n\t"
                               "shufpd $0x1, %%xmm3, %%xmm3 \n\t"
                               "shufpd $0x1, %%xmm4, %%xmm4 \n\t"
                               "shufpd $0x1, %%xmm5, %%xmm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("mulpd %0, %%xmm0 \n\t"
                               "mulpd %1, %%xmm1 \n\t"
                               "mulpd %2, %%xmm2 \n\t"
                               "mulpd %3, %%xmm3 \n\t"
                               "mulpd %4, %%xmm4 \n\t"
                               "mulpd %5, %%xmm5"
                               :
                               :
                               "m" ((*r).c1.c1),
                               "m" ((*r).c1.c2),
                               "m" ((*r).c1.c3),
                               "m" ((*r).c2.c1),
                               "m" ((*r).c2.c2),
                               "m" ((*r).c2.c3)
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("addpd %%xmm0, %%xmm1 \n\t"
                               "addpd %%xmm2, %%xmm3 \n\t"
                               "addpd %%xmm4, %%xmm5 \n\t"
                               "addpd %%xmm1, %%xmm9 \n\t"
                               "addpd %%xmm3, %%xmm10 \n\t"
                               "addpd %%xmm5, %%xmm11"
                               :
                               :
                               :
                               "xmm1", "xmm3", "xmm5",
                               "xmm9", "xmm10", "xmm11");

         _sse_load_dble((*s).c3);
         _sse_load_up_dble((*s).c4);

         __asm__ __volatile__ ("shufpd $0x1, %%xmm0, %%xmm0 \n\t"
                               "shufpd $0x1, %%xmm1, %%xmm1 \n\t"
                               "shufpd $0x1, %%xmm2, %%xmm2 \n\t"
                               "shufpd $0x1, %%xmm3, %%xmm3 \n\t"
                               "shufpd $0x1, %%xmm4, %%xmm4 \n\t"
                               "shufpd $0x1, %%xmm5, %%xmm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("mulpd %0, %%xmm0 \n\t"
                               "mulpd %1, %%xmm1 \n\t"
                               "mulpd %2, %%xmm2 \n\t"
                               "mulpd %3, %%xmm3 \n\t"
                               "mulpd %4, %%xmm4 \n\t"
                               "mulpd %5, %%xmm5"
                               :
                               :
                               "m" ((*r).c3.c1),
                               "m" ((*r).c3.c2),
                               "m" ((*r).c3.c3),
                               "m" ((*r).c4.c1),
                               "m" ((*r).c4.c2),
                               "m" ((*r).c4.c3)
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("addpd %%xmm0, %%xmm1 \n\t"
                               "addpd %%xmm2, %%xmm3 \n\t"
                               "addpd %%xmm4, %%xmm5 \n\t"
                               "addpd %%xmm1, %%xmm9 \n\t"
                               "addpd %%xmm3, %%xmm10 \n\t"
                               "addpd %%xmm5, %%xmm11"
                               :
                               :
                               :
                               "xmm1", "xmm3", "xmm5",
                               "xmm9", "xmm10", "xmm11");

         r+=1;
      }

      __asm__ __volatile__ ("addpd %%xmm6, %%xmm8 \n\t"
                            "addpd %%xmm9, %%xmm11 \n\t"
                            "addpd %%xmm7, %%xmm8 \n\t"
                            "addpd %%xmm10, %%xmm11 \n\t"
                            "haddpd %%xmm8, %%xmm8 \n\t"
                            "hsubpd %%xmm11, %%xmm11 \n\t"
                            "movsd %%xmm8, %0 \n\t"
                            "movsd %%xmm11, %1"
                            :
                            "=m" (smz.re),
                            "=m" (smz.im)
                            :
                            :
                            "xmm8", "xmm11");

      acc_qflt(smz.re,qsm[0]);
      acc_qflt(smz.im,qsm[1]);
   }

   qsm[1][0]=-qsm[1][0];
   qsm[1][1]=-qsm[1][1];

   return cqsm;
}


static qflt loc_spinor_prod_re_dble(int vol,spinor_dble *s,spinor_dble *r)
{
   double smx,*qsm[1];
   qflt rqsm;
   spinor_dble *sm,*smb;

   qsm[0]=rqsm.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      __asm__ __volatile__ ("xorpd %%xmm6, %%xmm6 \n\t"
                            "xorpd %%xmm7, %%xmm7 \n\t"
                            "xorpd %%xmm8, %%xmm8"
                            :
                            :
                            :
                            "xmm6", "xmm7", "xmm8");

      for (;s<smb;s++)
      {
         _sse_load_dble((*s).c1);
         _sse_load_up_dble((*s).c2);

         __asm__ __volatile__ ("mulpd %0, %%xmm0 \n\t"
                               "mulpd %1, %%xmm1 \n\t"
                               "mulpd %2, %%xmm2 \n\t"
                               "mulpd %3, %%xmm3 \n\t"
                               "mulpd %4, %%xmm4 \n\t"
                               "mulpd %5, %%xmm5"
                               :
                               :
                               "m" ((*r).c1.c1),
                               "m" ((*r).c1.c2),
                               "m" ((*r).c1.c3),
                               "m" ((*r).c2.c1),
                               "m" ((*r).c2.c2),
                               "m" ((*r).c2.c3)
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("addpd %%xmm0, %%xmm1 \n\t"
                               "addpd %%xmm2, %%xmm3 \n\t"
                               "addpd %%xmm4, %%xmm5 \n\t"
                               "addpd %%xmm1, %%xmm6 \n\t"
                               "addpd %%xmm3, %%xmm7 \n\t"
                               "addpd %%xmm5, %%xmm8"
                               :
                               :
                               :
                               "xmm1", "xmm3", "xmm5",
                               "xmm6", "xmm7", "xmm8");

         _sse_load_dble((*s).c3);
         _sse_load_up_dble((*s).c4);

         __asm__ __volatile__ ("mulpd %0, %%xmm0 \n\t"
                               "mulpd %1, %%xmm1 \n\t"
                               "mulpd %2, %%xmm2 \n\t"
                               "mulpd %3, %%xmm3 \n\t"
                               "mulpd %4, %%xmm4 \n\t"
                               "mulpd %5, %%xmm5"
                               :
                               :
                               "m" ((*r).c3.c1),
                               "m" ((*r).c3.c2),
                               "m" ((*r).c3.c3),
                               "m" ((*r).c4.c1),
                               "m" ((*r).c4.c2),
                               "m" ((*r).c4.c3)
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("addpd %%xmm0, %%xmm1 \n\t"
                               "addpd %%xmm2, %%xmm3 \n\t"
                               "addpd %%xmm4, %%xmm5 \n\t"
                               "addpd %%xmm1, %%xmm6 \n\t"
                               "addpd %%xmm3, %%xmm7 \n\t"
                               "addpd %%xmm5, %%xmm8"
                               :
                               :
                               :
                               "xmm1", "xmm3", "xmm5",
                               "xmm6", "xmm7", "xmm8");

         r+=1;
      }

      __asm__ __volatile__ ("addpd %%xmm6, %%xmm8 \n\t"
                            "addpd %%xmm7, %%xmm8 \n\t"
                            "haddpd %%xmm8, %%xmm8 \n\t"
                            "movsd %%xmm8, %0"
                            :
                            "=m" (smx)
                            :
                            :
                            "xmm8");

      acc_qflt(smx,qsm[0]);
   }

   return rqsm;
}


static complex_qflt loc_spinor_prod5_dble(int vol,spinor_dble *s,spinor_dble *r)
{
   double *qsm[2];
   complex_dble smz;
   complex_qflt cqsm;
   spinor_dble *sm,*smb;

   qsm[0]=cqsm.re.q;
   qsm[1]=cqsm.im.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   qsm[1][0]=0.0;
   qsm[1][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      __asm__ __volatile__ ("xorpd %%xmm6, %%xmm6 \n\t"
                            "xorpd %%xmm7, %%xmm7 \n\t"
                            "xorpd %%xmm8, %%xmm8 \n\t"
                            "xorpd %%xmm9, %%xmm9 \n\t"
                            "xorpd %%xmm10, %%xmm10 \n\t"
                            "xorpd %%xmm11, %%xmm11"
                            :
                            :
                            :
                            "xmm6", "xmm7", "xmm8",
                            "xmm9", "xmm10", "xmm11");

      for (;s<smb;s++)
      {
         _sse_load_dble((*s).c1);
         _sse_load_up_dble((*s).c2);

         __asm__ __volatile__ ("mulpd %0, %%xmm0 \n\t"
                               "mulpd %1, %%xmm1 \n\t"
                               "mulpd %2, %%xmm2 \n\t"
                               "mulpd %3, %%xmm3 \n\t"
                               "mulpd %4, %%xmm4 \n\t"
                               "mulpd %5, %%xmm5"
                               :
                               :
                               "m" ((*r).c1.c1),
                               "m" ((*r).c1.c2),
                               "m" ((*r).c1.c3),
                               "m" ((*r).c2.c1),
                               "m" ((*r).c2.c2),
                               "m" ((*r).c2.c3)
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("addpd %%xmm0, %%xmm1 \n\t"
                               "addpd %%xmm2, %%xmm3 \n\t"
                               "addpd %%xmm4, %%xmm5 \n\t"
                               "addpd %%xmm1, %%xmm6 \n\t"
                               "addpd %%xmm3, %%xmm7 \n\t"
                               "addpd %%xmm5, %%xmm8"
                               :
                               :
                               :
                               "xmm1", "xmm3", "xmm5",
                               "xmm6", "xmm7", "xmm8");

         _sse_load_dble((*s).c3);
         _sse_load_up_dble((*s).c4);

         __asm__ __volatile__ ("mulpd %0, %%xmm0 \n\t"
                               "mulpd %1, %%xmm1 \n\t"
                               "mulpd %2, %%xmm2 \n\t"
                               "mulpd %3, %%xmm3 \n\t"
                               "mulpd %4, %%xmm4 \n\t"
                               "mulpd %5, %%xmm5"
                               :
                               :
                               "m" ((*r).c3.c1),
                               "m" ((*r).c3.c2),
                               "m" ((*r).c3.c3),
                               "m" ((*r).c4.c1),
                               "m" ((*r).c4.c2),
                               "m" ((*r).c4.c3)
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("addpd %%xmm0, %%xmm1 \n\t"
                               "addpd %%xmm2, %%xmm3 \n\t"
                               "addpd %%xmm4, %%xmm5 \n\t"
                               "subpd %%xmm1, %%xmm6 \n\t"
                               "subpd %%xmm3, %%xmm7 \n\t"
                               "subpd %%xmm5, %%xmm8"
                               :
                               :
                               :
                               "xmm1", "xmm3", "xmm5",
                               "xmm6", "xmm7", "xmm8");

         _sse_load_dble((*s).c1);
         _sse_load_up_dble((*s).c2);

         __asm__ __volatile__ ("shufpd $0x1, %%xmm0, %%xmm0 \n\t"
                               "shufpd $0x1, %%xmm1, %%xmm1 \n\t"
                               "shufpd $0x1, %%xmm2, %%xmm2 \n\t"
                               "shufpd $0x1, %%xmm3, %%xmm3 \n\t"
                               "shufpd $0x1, %%xmm4, %%xmm4 \n\t"
                               "shufpd $0x1, %%xmm5, %%xmm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("mulpd %0, %%xmm0 \n\t"
                               "mulpd %1, %%xmm1 \n\t"
                               "mulpd %2, %%xmm2 \n\t"
                               "mulpd %3, %%xmm3 \n\t"
                               "mulpd %4, %%xmm4 \n\t"
                               "mulpd %5, %%xmm5"
                               :
                               :
                               "m" ((*r).c1.c1),
                               "m" ((*r).c1.c2),
                               "m" ((*r).c1.c3),
                               "m" ((*r).c2.c1),
                               "m" ((*r).c2.c2),
                               "m" ((*r).c2.c3)
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("addpd %%xmm0, %%xmm1 \n\t"
                               "addpd %%xmm2, %%xmm3 \n\t"
                               "addpd %%xmm4, %%xmm5 \n\t"
                               "addpd %%xmm1, %%xmm9 \n\t"
                               "addpd %%xmm3, %%xmm10 \n\t"
                               "addpd %%xmm5, %%xmm11"
                               :
                               :
                               :
                               "xmm1", "xmm3", "xmm5",
                               "xmm9", "xmm10", "xmm11");

         _sse_load_dble((*s).c3);
         _sse_load_up_dble((*s).c4);

         __asm__ __volatile__ ("shufpd $0x1, %%xmm0, %%xmm0 \n\t"
                               "shufpd $0x1, %%xmm1, %%xmm1 \n\t"
                               "shufpd $0x1, %%xmm2, %%xmm2 \n\t"
                               "shufpd $0x1, %%xmm3, %%xmm3 \n\t"
                               "shufpd $0x1, %%xmm4, %%xmm4 \n\t"
                               "shufpd $0x1, %%xmm5, %%xmm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("mulpd %0, %%xmm0 \n\t"
                               "mulpd %1, %%xmm1 \n\t"
                               "mulpd %2, %%xmm2 \n\t"
                               "mulpd %3, %%xmm3 \n\t"
                               "mulpd %4, %%xmm4 \n\t"
                               "mulpd %5, %%xmm5"
                               :
                               :
                               "m" ((*r).c3.c1),
                               "m" ((*r).c3.c2),
                               "m" ((*r).c3.c3),
                               "m" ((*r).c4.c1),
                               "m" ((*r).c4.c2),
                               "m" ((*r).c4.c3)
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("addpd %%xmm0, %%xmm1 \n\t"
                               "addpd %%xmm2, %%xmm3 \n\t"
                               "addpd %%xmm4, %%xmm5 \n\t"
                               "subpd %%xmm1, %%xmm9 \n\t"
                               "subpd %%xmm3, %%xmm10 \n\t"
                               "subpd %%xmm5, %%xmm11"
                               :
                               :
                               :
                               "xmm1", "xmm3", "xmm5",
                               "xmm9", "xmm10", "xmm11");

         r+=1;
      }

      __asm__ __volatile__ ("addpd %%xmm6, %%xmm8 \n\t"
                            "addpd %%xmm9, %%xmm11 \n\t"
                            "addpd %%xmm7, %%xmm8 \n\t"
                            "addpd %%xmm10, %%xmm11 \n\t"
                            "haddpd %%xmm8, %%xmm8 \n\t"
                            "hsubpd %%xmm11, %%xmm11 \n\t"
                            "movsd %%xmm8, %0 \n\t"
                            "movsd %%xmm11, %1"
                            :
                            "=m" (smz.re),
                            "=m" (smz.im)
                            :
                            :
                            "xmm8", "xmm11");

      acc_qflt(smz.re,qsm[0]);
      acc_qflt(smz.im,qsm[1]);
   }

   qsm[1][0]=-qsm[1][0];
   qsm[1][1]=-qsm[1][1];

   return cqsm;
}


static qflt loc_norm_square_dble(int vol,spinor_dble *s)
{
   double smx,*qsm[1];
   qflt rqsm;
   spinor_dble *sm,*smb;

   qsm[0]=rqsm.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      __asm__ __volatile__ ("xorpd %%xmm6, %%xmm6 \n\t"
                            "xorpd %%xmm7, %%xmm7 \n\t"
                            "xorpd %%xmm8, %%xmm8"
                            :
                            :
                            :
                            "xmm6", "xmm7", "xmm8");

      for (;s<smb;s++)
      {
         _sse_load_dble((*s).c1);
         _sse_load_up_dble((*s).c2);

         __asm__ __volatile__ ("mulpd %%xmm0, %%xmm0 \n\t"
                               "mulpd %%xmm1, %%xmm1 \n\t"
                               "mulpd %%xmm2, %%xmm2 \n\t"
                               "mulpd %%xmm3, %%xmm3 \n\t"
                               "mulpd %%xmm4, %%xmm4 \n\t"
                               "mulpd %%xmm5, %%xmm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("addpd %%xmm0, %%xmm1 \n\t"
                               "addpd %%xmm2, %%xmm3 \n\t"
                               "addpd %%xmm4, %%xmm5 \n\t"
                               "addpd %%xmm1, %%xmm6 \n\t"
                               "addpd %%xmm3, %%xmm7 \n\t"
                               "addpd %%xmm5, %%xmm8"
                               :
                               :
                               :
                               "xmm1", "xmm3", "xmm5",
                               "xmm6", "xmm7", "xmm8");

         _sse_load_dble((*s).c3);
         _sse_load_up_dble((*s).c4);

         __asm__ __volatile__ ("mulpd %%xmm0, %%xmm0 \n\t"
                               "mulpd %%xmm1, %%xmm1 \n\t"
                               "mulpd %%xmm2, %%xmm2 \n\t"
                               "mulpd %%xmm3, %%xmm3 \n\t"
                               "mulpd %%xmm4, %%xmm4 \n\t"
                               "mulpd %%xmm5, %%xmm5"
                               :
                               :
                               :
                               "xmm0", "xmm1", "xmm2",
                               "xmm3", "xmm4", "xmm5");

         __asm__ __volatile__ ("addpd %%xmm0, %%xmm1 \n\t"
                               "addpd %%xmm2, %%xmm3 \n\t"
                               "addpd %%xmm4, %%xmm5 \n\t"
                               "addpd %%xmm1, %%xmm6 \n\t"
                               "addpd %%xmm3, %%xmm7 \n\t"
                               "addpd %%xmm5, %%xmm8"
                               :
                               :
                               :
                               "xmm1", "xmm3", "xmm5",
                               "xmm6", "xmm7", "xmm8");
      }

      __asm__ __volatile__ ("addpd %%xmm6, %%xmm8 \n\t"
                            "addpd %%xmm7, %%xmm8 \n\t"
                            "haddpd %%xmm8, %%xmm8 \n\t"
                            "movsd %%xmm8, %0"
                            :
                            "=m" (smx)
                            :
                            :
                            "xmm8");

      acc_qflt(smx,qsm[0]);
   }

   return rqsm;
}


static void loc_mulc_spinor_add_dble(int vol,spinor_dble *s,spinor_dble *r,
                                     complex_dble z)
{
   spinor_dble *sm;

   _sse_load_cmplx_dble(z);
   sm=s+vol;

   for (;s<sm;s++)
   {
      _sse_load_dble((*s).c1);
      _sse_load_up_dble((*s).c2);
      _sse_mulc_vector_add_dble((*r).c1);
      _sse_mulc_vector_add_up_dble((*r).c2);
      _sse_store_dble((*s).c1);
      _sse_store_up_dble((*s).c2);

      _sse_load_dble((*s).c3);
      _sse_load_up_dble((*s).c4);
      _sse_mulc_vector_add_dble((*r).c3);
      _sse_mulc_vector_add_up_dble((*r).c4);
      _sse_store_dble((*s).c3);
      _sse_store_up_dble((*s).c4);

      r+=1;
   }
}


static void loc_mulr_spinor_add_dble(int vol,spinor_dble *s,spinor_dble *r,
                                     double c)
{
   spinor_dble *sm;

   _sse_load_real_dble(c);

   sm=s+vol;

   for (;s<sm;s++)
   {
      _sse_load_dble((*r).c1);
      _sse_mulr_vector_add_dble((*s).c1);
      _sse_load_up_dble((*r).c2);
      _sse_mulr_vector_add_up_dble((*s).c2);
      _sse_store_dble((*s).c1);
      _sse_store_up_dble((*s).c2);

      _sse_load_dble((*r).c3);
      _sse_mulr_vector_add_dble((*s).c3);
      _sse_load_up_dble((*r).c4);
      _sse_mulr_vector_add_up_dble((*s).c4);
      _sse_store_dble((*s).c3);
      _sse_store_up_dble((*s).c4);

      r+=1;
   }
}


static void loc_combine_spinor_dble(int vol,spinor_dble *s,spinor_dble *r,
                                    double cs,double cr)
{
   spinor_dble *sm;

   __asm__ __volatile__ ("movddup %0, %%xmm12 \n\t"
                         "movddup %1, %%xmm14 \n\t"
                         "movapd %%xmm12, %%xmm13 \n\t"
                         "movapd %%xmm14, %%xmm15 \n\t"
                         :
                         :
                         "m" (cs),
                         "m" (cr)
                         :
                         "xmm12", "xmm13", "xmm14", "xmm15");

   sm=s+vol;

   for (;s<sm;s++)
   {
      __asm__ __volatile__ ("movapd %0, %%xmm0 \n\t"
                            "movapd %1, %%xmm1 \n\t"
                            "movapd %2, %%xmm2 \n\t"
                            "movapd %3, %%xmm3 \n\t"
                            "movapd %4, %%xmm4 \n\t"
                            "movapd %5, %%xmm5 \n\t"
                            "mulpd %%xmm12, %%xmm0 \n\t"
                            "mulpd %%xmm13, %%xmm1 \n\t"
                            "mulpd %%xmm12, %%xmm2 \n\t"
                            "mulpd %%xmm13, %%xmm3 \n\t"
                            "mulpd %%xmm12, %%xmm4 \n\t"
                            "mulpd %%xmm13, %%xmm5"
                            :
                            :
                            "m" ((*s).c1.c1),
                            "m" ((*s).c1.c2),
                            "m" ((*s).c1.c3),
                            "m" ((*s).c2.c1),
                            "m" ((*s).c2.c2),
                            "m" ((*s).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      __asm__ __volatile__ ("movapd %0, %%xmm6 \n\t"
                            "movapd %1, %%xmm7 \n\t"
                            "movapd %2, %%xmm8 \n\t"
                            "movapd %3, %%xmm9 \n\t"
                            "movapd %4, %%xmm10 \n\t"
                            "movapd %5, %%xmm11 \n\t"
                            "mulpd %%xmm14, %%xmm6 \n\t"
                            "mulpd %%xmm15, %%xmm7 \n\t"
                            "mulpd %%xmm14, %%xmm8 \n\t"
                            "mulpd %%xmm15, %%xmm9 \n\t"
                            "mulpd %%xmm14, %%xmm10 \n\t"
                            "mulpd %%xmm15, %%xmm11"
                            :
                            :
                            "m" ((*r).c1.c1),
                            "m" ((*r).c1.c2),
                            "m" ((*r).c1.c3),
                            "m" ((*r).c2.c1),
                            "m" ((*r).c2.c2),
                            "m" ((*r).c2.c3)
                            :
                            "xmm6", "xmm7", "xmm8", "xmm9",
                            "xmm10", "xmm11");

      __asm__ __volatile__ ("addpd %%xmm6, %%xmm0 \n\t"
                            "addpd %%xmm7, %%xmm1 \n\t"
                            "addpd %%xmm8, %%xmm2 \n\t"
                            "addpd %%xmm9, %%xmm3 \n\t"
                            "addpd %%xmm10, %%xmm4 \n\t"
                            "addpd %%xmm11, %%xmm5 \n\t"
                            "movapd %%xmm0, %0 \n\t"
                            "movapd %%xmm1, %1 \n\t"
                            "movapd %%xmm2, %2 \n\t"
                            "movapd %%xmm3, %3 \n\t"
                            "movapd %%xmm4, %4 \n\t"
                            "movapd %%xmm5, %5"
                            :
                            "=m" ((*s).c1.c1),
                            "=m" ((*s).c1.c2),
                            "=m" ((*s).c1.c3),
                            "=m" ((*s).c2.c1),
                            "=m" ((*s).c2.c2),
                            "=m" ((*s).c2.c3)
                            :
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      __asm__ __volatile__ ("movapd %0, %%xmm0 \n\t"
                            "movapd %1, %%xmm1 \n\t"
                            "movapd %2, %%xmm2 \n\t"
                            "movapd %3, %%xmm3 \n\t"
                            "movapd %4, %%xmm4 \n\t"
                            "movapd %5, %%xmm5 \n\t"
                            "mulpd %%xmm12, %%xmm0 \n\t"
                            "mulpd %%xmm13, %%xmm1 \n\t"
                            "mulpd %%xmm12, %%xmm2 \n\t"
                            "mulpd %%xmm13, %%xmm3 \n\t"
                            "mulpd %%xmm12, %%xmm4 \n\t"
                            "mulpd %%xmm13, %%xmm5"
                            :
                            :
                            "m" ((*s).c3.c1),
                            "m" ((*s).c3.c2),
                            "m" ((*s).c3.c3),
                            "m" ((*s).c4.c1),
                            "m" ((*s).c4.c2),
                            "m" ((*s).c4.c3)
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      __asm__ __volatile__ ("movapd %0, %%xmm6 \n\t"
                            "movapd %1, %%xmm7 \n\t"
                            "movapd %2, %%xmm8 \n\t"
                            "movapd %3, %%xmm9 \n\t"
                            "movapd %4, %%xmm10 \n\t"
                            "movapd %5, %%xmm11 \n\t"
                            "mulpd %%xmm14, %%xmm6 \n\t"
                            "mulpd %%xmm15, %%xmm7 \n\t"
                            "mulpd %%xmm14, %%xmm8 \n\t"
                            "mulpd %%xmm15, %%xmm9 \n\t"
                            "mulpd %%xmm14, %%xmm10 \n\t"
                            "mulpd %%xmm15, %%xmm11"
                            :
                            :
                            "m" ((*r).c3.c1),
                            "m" ((*r).c3.c2),
                            "m" ((*r).c3.c3),
                            "m" ((*r).c4.c1),
                            "m" ((*r).c4.c2),
                            "m" ((*r).c4.c3)
                            :
                            "xmm6", "xmm7", "xmm8", "xmm9",
                            "xmm10", "xmm11");

      __asm__ __volatile__ ("addpd %%xmm6, %%xmm0 \n\t"
                            "addpd %%xmm7, %%xmm1 \n\t"
                            "addpd %%xmm8, %%xmm2 \n\t"
                            "addpd %%xmm9, %%xmm3 \n\t"
                            "addpd %%xmm10, %%xmm4 \n\t"
                            "addpd %%xmm11, %%xmm5 \n\t"
                            "movapd %%xmm0, %0 \n\t"
                            "movapd %%xmm1, %1 \n\t"
                            "movapd %%xmm2, %2 \n\t"
                            "movapd %%xmm3, %3 \n\t"
                            "movapd %%xmm4, %4 \n\t"
                            "movapd %%xmm5, %5"
                            :
                            "=m" ((*s).c3.c1),
                            "=m" ((*s).c3.c2),
                            "=m" ((*s).c3.c3),
                            "=m" ((*s).c4.c1),
                            "=m" ((*s).c4.c2),
                            "=m" ((*s).c4.c3)
                            :
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      r+=1;
   }
}


static void loc_scale_dble(int vol,double c,spinor_dble *s)
{
   spinor_dble *sm;

   _sse_load_real_dble(c);

   sm=s+vol;

   for (;s<sm;s++)
   {
      _sse_load_dble((*s).c1);
      _sse_mulr_vector_dble();
      _sse_load_up_dble((*s).c2);
      _sse_mulr_vector_up_dble();
      _sse_store_dble((*s).c1);
      _sse_store_up_dble((*s).c2);

      _sse_load_dble((*s).c3);
      _sse_mulr_vector_dble();
      _sse_load_up_dble((*s).c4);
      _sse_mulr_vector_up_dble();
      _sse_store_dble((*s).c3);
      _sse_store_up_dble((*s).c4);
   }
}


static void loc_mulg5_dble(int vol,spinor_dble *s)
{
   spinor_dble *sm;


   __asm__ __volatile__ ("movapd %0, %%xmm6 \n\t"
                         "movapd %%xmm6, %%xmm7"
                         :
                         :
                         "m" (_sse_sgn_dble)
                         :
                         "xmm6", "xmm7");

   sm=s+vol;

   for (;s<sm;s++)
   {
      _sse_load_dble((*s).c3);
      _sse_mulr_vector_dble();
      _sse_load_up_dble((*s).c4);
      _sse_mulr_vector_up_dble();
      _sse_store_dble((*s).c3);
      _sse_store_up_dble((*s).c4);
   }
}


static void loc_mulmg5_dble(int vol,spinor_dble *s)
{
   spinor_dble *sm;

   __asm__ __volatile__ ("movapd %0, %%xmm6 \n\t"
                         "movapd %%xmm6, %%xmm7"
                         :
                         :
                         "m" (_sse_sgn_dble)
                         :
                         "xmm6", "xmm7");

   sm=s+vol;

   for (;s<sm;s++)
   {
      _sse_load_dble((*s).c1);
      _sse_mulr_vector_dble();
      _sse_load_up_dble((*s).c2);
      _sse_mulr_vector_up_dble();
      _sse_store_dble((*s).c1);
      _sse_store_up_dble((*s).c2);
   }
}

#else

static complex_qflt loc_spinor_prod_dble(int vol,spinor_dble *s,spinor_dble *r)
{
   double *qsm[2];
   complex_dble smz;
   complex_qflt cqsm;
   spinor_dble *sm,*smb;

   qsm[0]=cqsm.re.q;
   qsm[1]=cqsm.im.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   qsm[1][0]=0.0;
   qsm[1][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      smz.re=0.0;
      smz.im=0.0;

      for (;s<smb;s++)
      {
         smz.re+=(_vector_prod_re((*s).c1,(*r).c1)+
                  _vector_prod_re((*s).c2,(*r).c2)+
                  _vector_prod_re((*s).c3,(*r).c3)+
                  _vector_prod_re((*s).c4,(*r).c4));

         smz.im+=(_vector_prod_im((*s).c1,(*r).c1)+
                  _vector_prod_im((*s).c2,(*r).c2)+
                  _vector_prod_im((*s).c3,(*r).c3)+
                  _vector_prod_im((*s).c4,(*r).c4));

         r+=1;
      }

      acc_qflt(smz.re,qsm[0]);
      acc_qflt(smz.im,qsm[1]);
   }

   return cqsm;
}


static qflt loc_spinor_prod_re_dble(int vol,spinor_dble *s,spinor_dble *r)
{
   double smx,*qsm[1];
   qflt rqsm;
   spinor_dble *sm,*smb;

   qsm[0]=rqsm.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      smx=0.0;

      for (;s<smb;s++)
      {
         smx+=(_vector_prod_re((*s).c1,(*r).c1)+
               _vector_prod_re((*s).c2,(*r).c2)+
               _vector_prod_re((*s).c3,(*r).c3)+
               _vector_prod_re((*s).c4,(*r).c4));

         r+=1;
      }

      acc_qflt(smx,qsm[0]);
   }

   return rqsm;
}


static complex_qflt loc_spinor_prod5_dble(int vol,spinor_dble *s,spinor_dble *r)
{
   double *qsm[2];
   complex_dble smz;
   complex_qflt cqsm;
   spinor_dble *sm,*smb;

   qsm[0]=cqsm.re.q;
   qsm[1]=cqsm.im.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   qsm[1][0]=0.0;
   qsm[1][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      smz.re=0.0;
      smz.im=0.0;

      for (;s<smb;s++)
      {
         smz.re+=(_vector_prod_re((*s).c1,(*r).c1)+
                  _vector_prod_re((*s).c2,(*r).c2));

         smz.re-=(_vector_prod_re((*s).c3,(*r).c3)+
                  _vector_prod_re((*s).c4,(*r).c4));

         smz.im+=(_vector_prod_im((*s).c1,(*r).c1)+
                  _vector_prod_im((*s).c2,(*r).c2));

         smz.im-=(_vector_prod_im((*s).c3,(*r).c3)+
                  _vector_prod_im((*s).c4,(*r).c4));

         r+=1;
      }

      acc_qflt(smz.re,qsm[0]);
      acc_qflt(smz.im,qsm[1]);
   }

   return cqsm;
}


static qflt loc_norm_square_dble(int vol,spinor_dble *s)
{
   double smx,*qsm[1];
   qflt rqsm;
   spinor_dble *sm,*smb;

   qsm[0]=rqsm.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   sm=s+vol;

   while (s<sm)
   {
      smb=s+8;
      if (smb>sm)
         smb=sm;

      smx=0.0;

      for (;s<smb;s++)
      {
         smx+=(_vector_prod_re((*s).c1,(*s).c1)+
               _vector_prod_re((*s).c2,(*s).c2)+
               _vector_prod_re((*s).c3,(*s).c3)+
               _vector_prod_re((*s).c4,(*s).c4));
      }

      acc_qflt(smx,qsm[0]);
   }

   return rqsm;
}


static void loc_mulc_spinor_add_dble(int vol,spinor_dble *s,spinor_dble *r,
                                     complex_dble z)
{
   spinor_dble *sm;

   sm=s+vol;

   for (;s<sm;s++)
   {
      _vector_mulc_assign((*s).c1,z,(*r).c1);
      _vector_mulc_assign((*s).c2,z,(*r).c2);
      _vector_mulc_assign((*s).c3,z,(*r).c3);
      _vector_mulc_assign((*s).c4,z,(*r).c4);

      r+=1;
   }
}


static void loc_mulr_spinor_add_dble(int vol,spinor_dble *s,spinor_dble *r,
                                     double c)
{
   spinor_dble *sm;

   sm=s+vol;

   for (;s<sm;s++)
   {
      _vector_mulr_assign((*s).c1,c,(*r).c1);
      _vector_mulr_assign((*s).c2,c,(*r).c2);
      _vector_mulr_assign((*s).c3,c,(*r).c3);
      _vector_mulr_assign((*s).c4,c,(*r).c4);

      r+=1;
   }
}


static void loc_combine_spinor_dble(int vol,spinor_dble *s,spinor_dble *r,
                                    double cs,double cr)
{
   spinor_dble *sm;

   sm=s+vol;

   for (;s<sm;s++)
   {
      _vector_combine((*s).c1,(*r).c1,cs,cr);
      _vector_combine((*s).c2,(*r).c2,cs,cr);
      _vector_combine((*s).c3,(*r).c3,cs,cr);
      _vector_combine((*s).c4,(*r).c4,cs,cr);

      r+=1;
   }
}


static void loc_scale_dble(int vol,double c,spinor_dble *s)
{
   spinor_dble *sm;

   sm=s+vol;

   for (;s<sm;s++)
   {
      _vector_mul((*s).c1,c,(*s).c1);
      _vector_mul((*s).c2,c,(*s).c2);
      _vector_mul((*s).c3,c,(*s).c3);
      _vector_mul((*s).c4,c,(*s).c4);
   }
}


static void loc_mulg5_dble(int vol,spinor_dble *s)
{
   spinor_dble *sm;

   sm=s+vol;

   for (;s<sm;s++)
   {
      (*s).c3.c1.re=-(*s).c3.c1.re;
      (*s).c3.c1.im=-(*s).c3.c1.im;
      (*s).c3.c2.re=-(*s).c3.c2.re;
      (*s).c3.c2.im=-(*s).c3.c2.im;
      (*s).c3.c3.re=-(*s).c3.c3.re;
      (*s).c3.c3.im=-(*s).c3.c3.im;
      (*s).c4.c1.re=-(*s).c4.c1.re;
      (*s).c4.c1.im=-(*s).c4.c1.im;
      (*s).c4.c2.re=-(*s).c4.c2.re;
      (*s).c4.c2.im=-(*s).c4.c2.im;
      (*s).c4.c3.re=-(*s).c4.c3.re;
      (*s).c4.c3.im=-(*s).c4.c3.im;
   }
}


static void loc_mulmg5_dble(int vol,spinor_dble *s)
{
   spinor_dble *sm;

   sm=s+vol;

   for (;s<sm;s++)
   {
      (*s).c1.c1.re=-(*s).c1.c1.re;
      (*s).c1.c1.im=-(*s).c1.c1.im;
      (*s).c1.c2.re=-(*s).c1.c2.re;
      (*s).c1.c2.im=-(*s).c1.c2.im;
      (*s).c1.c3.re=-(*s).c1.c3.re;
      (*s).c1.c3.im=-(*s).c1.c3.im;
      (*s).c2.c1.re=-(*s).c2.c1.re;
      (*s).c2.c1.im=-(*s).c2.c1.im;
      (*s).c2.c2.re=-(*s).c2.c2.re;
      (*s).c2.c2.im=-(*s).c2.c2.im;
      (*s).c2.c3.re=-(*s).c2.c3.re;
      (*s).c2.c3.im=-(*s).c2.c3.im;
   }
}

#endif

complex_qflt spinor_prod_dble(int vol,int icom,spinor_dble *s,spinor_dble *r)
{
   int k;
   double *qsm[2];
   qflt reqsm,imqsm;
   complex_qflt cqsm;

   if ((icom&0x2)==0)
      cqsm=loc_spinor_prod_dble(vol,s,r);
   else
   {
      reqsm.q[0]=0.0;
      reqsm.q[1]=0.0;
      imqsm.q[0]=0.0;
      imqsm.q[1]=0.0;

#pragma omp parallel private(k,cqsm) reduction(sum_qflt : reqsm,imqsm)
      {
         k=omp_get_thread_num();
         cqsm=loc_spinor_prod_dble(vol,s+k*vol,r+k*vol);

         reqsm.q[0]=cqsm.re.q[0];
         reqsm.q[1]=cqsm.re.q[1];
         imqsm.q[0]=cqsm.im.q[0];
         imqsm.q[1]=cqsm.im.q[1];
      }

      cqsm.re.q[0]=reqsm.q[0];
      cqsm.re.q[1]=reqsm.q[1];
      cqsm.im.q[0]=imqsm.q[0];
      cqsm.im.q[1]=imqsm.q[1];
   }

   if ((NPROC>1)&&(icom&0x1))
   {
      qsm[0]=cqsm.re.q;
      qsm[1]=cqsm.im.q;
      global_qsum(2,qsm,qsm);
   }

   return cqsm;
}


qflt spinor_prod_re_dble(int vol,int icom,spinor_dble *s,spinor_dble *r)
{
   int k;
   double *qsm[1];
   qflt rqsm;

   if ((icom&0x2)==0)
      rqsm=loc_spinor_prod_re_dble(vol,s,r);
   else
   {
      rqsm.q[0]=0.0;
      rqsm.q[1]=0.0;

#pragma omp parallel private(k) reduction(sum_qflt : rqsm)
      {
         k=omp_get_thread_num();
         rqsm=loc_spinor_prod_re_dble(vol,s+k*vol,r+k*vol);
      }
   }

   if ((NPROC>1)&&(icom&0x1))
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   return rqsm;
}


complex_qflt spinor_prod5_dble(int vol,int icom,spinor_dble *s,spinor_dble *r)
{
   int k;
   double *qsm[2];
   qflt reqsm,imqsm;
   complex_qflt cqsm;

   if ((icom&0x2)==0)
      cqsm=loc_spinor_prod5_dble(vol,s,r);
   else
   {
      reqsm.q[0]=0.0;
      reqsm.q[1]=0.0;
      imqsm.q[0]=0.0;
      imqsm.q[1]=0.0;

#pragma omp parallel private(k,cqsm) reduction(sum_qflt : reqsm,imqsm)
      {
         k=omp_get_thread_num();
         cqsm=loc_spinor_prod5_dble(vol,s+k*vol,r+k*vol);

         reqsm.q[0]=cqsm.re.q[0];
         reqsm.q[1]=cqsm.re.q[1];
         imqsm.q[0]=cqsm.im.q[0];
         imqsm.q[1]=cqsm.im.q[1];
      }

      cqsm.re.q[0]=reqsm.q[0];
      cqsm.re.q[1]=reqsm.q[1];
      cqsm.im.q[0]=imqsm.q[0];
      cqsm.im.q[1]=imqsm.q[1];
   }

   if ((NPROC>1)&&(icom&0x1))
   {
      qsm[0]=cqsm.re.q;
      qsm[1]=cqsm.im.q;
      global_qsum(2,qsm,qsm);
   }

   return cqsm;
}


qflt norm_square_dble(int vol,int icom,spinor_dble *s)
{
   int k;
   double *qsm[1];
   qflt rqsm;

   if ((icom&0x2)==0)
      rqsm=loc_norm_square_dble(vol,s);
   else
   {
      rqsm.q[0]=0.0;
      rqsm.q[1]=0.0;

#pragma omp parallel private(k) reduction(sum_qflt : rqsm)
      {
         k=omp_get_thread_num();
         rqsm=loc_norm_square_dble(vol,s+k*vol);
      }
   }

   if ((NPROC>1)&&(icom&0x1))
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   return rqsm;
}


void mulc_spinor_add_dble(int vol,int icom,spinor_dble *s,spinor_dble *r,
                          complex_dble z)
{
   int k;

   if ((icom&0x2)==0)
      loc_mulc_spinor_add_dble(vol,s,r,z);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_mulc_spinor_add_dble(vol,s+k*vol,r+k*vol,z);
      }
   }
}


void mulr_spinor_add_dble(int vol,int icom,spinor_dble *s,spinor_dble *r,
                          double c)
{
   int k;

   if ((icom&0x2)==0)
      loc_mulr_spinor_add_dble(vol,s,r,c);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_mulr_spinor_add_dble(vol,s+k*vol,r+k*vol,c);
      }
   }
}


void combine_spinor_dble(int vol,int icom,spinor_dble *s,spinor_dble *r,
                         double cs,double cr)
{
   int k;

   if ((icom&0x2)==0)
      loc_combine_spinor_dble(vol,s,r,cs,cr);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_combine_spinor_dble(vol,s+k*vol,r+k*vol,cs,cr);
      }
   }
}


void scale_dble(int vol,int icom,double c,spinor_dble *s)
{
   int k;

   if ((icom&0x2)==0)
      loc_scale_dble(vol,c,s);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_scale_dble(vol,c,s+k*vol);
      }
   }
}


void mulg5_dble(int vol,int icom,spinor_dble *s)
{
   int k;

   if ((icom&0x2)==0)
      loc_mulg5_dble(vol,s);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_mulg5_dble(vol,s+k*vol);
      }
   }
}


void mulmg5_dble(int vol,int icom,spinor_dble *s)
{
   int k;

   if ((icom&0x2)==0)
      loc_mulmg5_dble(vol,s);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_mulmg5_dble(vol,s+k*vol);
      }
   }
}


void project_dble(int vol,int icom,spinor_dble *s,spinor_dble *r)
{
   complex_dble z;
   complex_qflt qz;

   qz=spinor_prod_dble(vol,icom,r,s);
   z.re=-qz.re.q[0];
   z.im=-qz.im.q[0];
   mulc_spinor_add_dble(vol,icom,s,r,z);
}


double normalize_dble(int vol,int icom,spinor_dble *s)
{
   double r;
   qflt qr;

   qr=norm_square_dble(vol,icom,s);
   r=sqrt(qr.q[0]);

   if (r!=0.0)
      scale_dble(vol,icom,1.0/r,s);
   else if (icom!=0)
      error_loc(1,1,"normalize_dble [salg_dble.c]",
                "Vector has vanishing norm");

   return r;
}
