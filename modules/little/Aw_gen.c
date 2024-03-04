
/*******************************************************************************
*
* File Aw_gen.c
*
* Copyright (C) 2007-2011, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Basic programs needed for the computation of the hopping terms of the
* little Dirac operator.
*
*   void gather_ud(int ifc,su3_dble *udb,block_t *b)
*     Gathers the link variables from the global gauge field udb on the
*     links that "stick out" of the block b in direction ifc and assigns
*     them to the field (*b).bb[ifc].ud. Only the variables on the links
*     connecting an odd block point with an even boundary point are copied.
*     The ordering of the copied link variables coincides with the one of
*     the boundary points.
*
*   void gather_se(int n,int ifc,block_t *b,spinor *sbuf)
*     Copies the spinors on the even points of the interior boundary of the
*     block b in direction ifc from the fields (*b).s[i], i=1,..,n, to the
*     buffer sbuf. The ordering of the spinors in the buffer coincides with
*     the one of the even boundary points on the opposite face of the block.
*
*   void gather_so(int n,int ifc,block_t *b,spinor_dble *sdbuf)
*     Fetches the spinors on the odd points of the interior boundary of the
*     block b in direction ifc from the fields (*b).s[i], i=1,..,n, converts
*     them to double precision, applies the link variables in direction ifc
*     and stores them in the buffer sdbuf. The ordering of the spinors in
*     the buffer coincides with the one of the even boundary points on that
*     face of the block.
*
* The following is an array of functions indexed by the direction mu=0,..,3:
*
*   void (*spinor_prod_gamma[mu])(int vol,spinor_dble *sd,spinor_dble *rd,
*                                 complex_dble *sp)
*      Computes the scalar products (sd,rd) and (sd,gamma_mu*rd), where
*      gamma_mu denotes the Dirac matrix with index mu=0,..,3 and the spinor
*      fields are assumed to have vol elements. On exit the products are
*      assigned to sp[0] and sp[1], respectively.
*
* The representation of the Dirac matrices is the one specified in the notes
* doc/dirac.pdf.
*
* All these programs are thread-safe. If SSE inline-assembly is used, the
* spinors must be aligned to a 16 byte boundary.
*
*******************************************************************************/

#define AW_GEN_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "utils.h"
#include "sflds.h"
#include "su3fcts.h"
#include "block.h"
#include "little.h"
#include "global.h"


void gather_ud(int ifc,su3_dble *udb,block_t *b)
{
   int i,volh,ix,*ipp,*imb;
   su3_dble *ud;

   volh=(*b).bb[ifc].vol/2;
   ipp=(*b).bb[ifc].ipp;
   imb=(*b).imb;
   ud=(*b).bb[ifc].ud;

   for (i=0;i<volh;i++)
   {
      ix=imb[ipp[i]];
      ud[0]=udb[8*(ix-(VOLUME/2))+(ifc^0x1)];
      ud+=1;
   }
}


void gather_se(int n,int ifc,block_t *b,spinor *sbuf)
{
   int i,j,volh,*imb;
   spinor **s,*r;

   volh=(*b).bb[ifc^0x1].vol/2;
   imb=(*b).bb[ifc^0x1].map;
   s=(*b).s;

   for (i=1;i<=n;i++)
   {
      r=s[i];

      for (j=0;j<volh;j++)
      {
         sbuf[0]=r[imb[j]];
         sbuf+=1;
      }
   }
}

#if (defined x64)
#include "sse2.h"

#define _sse_load_cvt(v) \
__asm__ __volatile__ ("cvtps2pd %0, %%xmm0 \n\t" \
                      "cvtps2pd %1, %%xmm1 \n\t" \
                      "cvtps2pd %2, %%xmm2" \
                      : \
                      : "m" ((v).c1), \
                        "m" ((v).c2), \
                        "m" ((v).c3)  \
                      : \
                      "xmm0", "xmm1", "xmm2")

#define _start_sm() \
__asm__ __volatile__ ("xorpd %%xmm12, %%xmm12 \n\t" \
                      "xorpd %%xmm13, %%xmm13 \n\t" \
                      "xorpd %%xmm14, %%xmm14 \n\t" \
                      "xorpd %%xmm15, %%xmm15" \
                      : \
                      : \
                      : \
                      "xmm12", "xmm13", "xmm14", "xmm15")

#define _store_sm() \
__asm__ __volatile__ ("shufpd $0x1, %%xmm12, %%xmm12 \n\t" \
                      "shufpd $0x1, %%xmm14, %%xmm14 \n\t" \
                      "addsubpd %%xmm12, %%xmm13 \n\t" \
                      "addsubpd %%xmm14, %%xmm15 \n\t" \
                      "shufpd $0x1, %%xmm13, %%xmm13 \n\t" \
                      "shufpd $0x1, %%xmm15, %%xmm15 \n\t" \
                      "movapd %%xmm13, %0 \n\t" \
                      "movapd %%xmm15, %1 \n\t" \
                      : \
                      "=m" (sm[0]), \
                      "=m" (sm[1]) \
                      : \
                      : \
                      "xmm12", "xmm13", "xmm14", "xmm15")

#define _load_chi(s) \
__asm__ __volatile__ ("movddup %0, %%xmm0 \n\t" \
                      "movddup %1, %%xmm1 \n\t" \
                      "movddup %2, %%xmm2 \n\t" \
                      "movddup %3, %%xmm3 \n\t" \
                      "movddup %4, %%xmm4 \n\t" \
                      "movddup %5, %%xmm5" \
                      : \
                      : \
                      "m" ((s).c1.re), \
                      "m" ((s).c2.re), \
                      "m" ((s).c3.re), \
                      "m" ((s).c1.im), \
                      "m" ((s).c2.im), \
                      "m" ((s).c3.im) \
                      : \
                      "xmm0", "xmm1", "xmm2", "xmm3", \
                      "xmm4", "xmm5")

#define _load_psi0(s) \
__asm__ __volatile__ ("movapd %0, %%xmm6 \n\t" \
                      "movapd %1, %%xmm7 \n\t" \
                      "movapd %2, %%xmm8 \n\t" \
                      "movapd %0, %%xmm9 \n\t" \
                      "movapd %1, %%xmm10 \n\t" \
                      "movapd %2, %%xmm11 \n\t" \
                      "mulpd %%xmm0, %%xmm6 \n\t" \
                      "mulpd %%xmm1, %%xmm7 \n\t" \
                      "mulpd %%xmm2, %%xmm8 \n\t" \
                      "mulpd %%xmm3, %%xmm9 \n\t" \
                      "mulpd %%xmm4, %%xmm10 \n\t" \
                      "mulpd %%xmm5, %%xmm11" \
                      : \
                      : \
                      "m" ((s).c1), \
                      "m" ((s).c2), \
                      "m" ((s).c3) \
                      : \
                      "xmm6", "xmm7", "xmm8", "xmm9", \
                      "xmm10", "xmm11")

#define _load_psi1_add(s) \
__asm__ __volatile__ ("mulpd %0, %%xmm0 \n\t" \
                      "mulpd %1, %%xmm1 \n\t" \
                      "mulpd %2, %%xmm2 \n\t" \
                      "mulpd %0, %%xmm3 \n\t" \
                      "mulpd %1, %%xmm4 \n\t" \
                      "mulpd %2, %%xmm5 \n\t" \
                      "addpd %%xmm6, %%xmm12 \n\t" \
                      "addpd %%xmm9, %%xmm13 \n\t" \
                      "addpd %%xmm0, %%xmm14 \n\t" \
                      "addpd %%xmm3, %%xmm15 \n\t" \
                      "addpd %%xmm7, %%xmm12 \n\t" \
                      "addpd %%xmm10, %%xmm13 \n\t" \
                      "addpd %%xmm1, %%xmm14 \n\t" \
                      "addpd %%xmm4, %%xmm15 \n\t" \
                      "addpd %%xmm8, %%xmm12 \n\t" \
                      "addpd %%xmm11, %%xmm13 \n\t" \
                      "addpd %%xmm2, %%xmm14 \n\t" \
                      "addpd %%xmm5, %%xmm15" \
                      : \
                      : \
                      "m" ((s).c1), \
                      "m" ((s).c2), \
                      "m" ((s).c3) \
                      : \
                      "xmm0", "xmm1", "xmm2", "xmm3", \
                      "xmm4", "xmm5", "xmm12", "xmm13", \
                      "xmm14", "xmm15")

#define _load_psi1_sub(s) \
__asm__ __volatile__ ("mulpd %0, %%xmm0 \n\t" \
                      "mulpd %1, %%xmm1 \n\t" \
                      "mulpd %2, %%xmm2 \n\t" \
                      "mulpd %0, %%xmm3 \n\t" \
                      "mulpd %1, %%xmm4 \n\t" \
                      "mulpd %2, %%xmm5 \n\t" \
                      "addpd %%xmm6, %%xmm12 \n\t" \
                      "addpd %%xmm9, %%xmm13 \n\t" \
                      "subpd %%xmm0, %%xmm14 \n\t" \
                      "subpd %%xmm3, %%xmm15 \n\t" \
                      "addpd %%xmm7, %%xmm12 \n\t" \
                      "addpd %%xmm10, %%xmm13 \n\t" \
                      "subpd %%xmm1, %%xmm14 \n\t" \
                      "subpd %%xmm4, %%xmm15 \n\t" \
                      "addpd %%xmm8, %%xmm12 \n\t" \
                      "addpd %%xmm11, %%xmm13 \n\t" \
                      "subpd %%xmm2, %%xmm14 \n\t" \
                      "subpd %%xmm5, %%xmm15" \
                      : \
                      : \
                      "m" ((s).c1), \
                      "m" ((s).c2), \
                      "m" ((s).c3) \
                      : \
                      "xmm0", "xmm1", "xmm2", "xmm3", \
                      "xmm4", "xmm5", "xmm12", "xmm13", \
                      "xmm14", "xmm15")


static void apply_u2s(int vol,int *imb,su3_dble *ud,spinor *s,
                      spinor_dble *rd)
{
   int *imm;
   spinor *si;

   imm=imb+vol;

   while (imb<imm)
   {
      si=s+(*imb);
      imb+=1;

      if (imb<imm)
         _prefetch_spinor(s+(*imb));

      _sse_load_cvt((*si).c1);
      _sse_su3_multiply_dble(*ud);
      _sse_store_up_dble((*rd).c1);

      _sse_load_cvt((*si).c2);
      _sse_su3_multiply_dble(*ud);
      _sse_store_up_dble((*rd).c2);

      _sse_load_cvt((*si).c3);
      _sse_su3_multiply_dble(*ud);
      _sse_store_up_dble((*rd).c3);

      _sse_load_cvt((*si).c4);
      _sse_su3_multiply_dble(*ud);
      _sse_store_up_dble((*rd).c4);

      ud+=1;
      rd+=1;
   }
}


static void apply_udag2s(int vol,int *imb,su3_dble *ud,spinor *s,
                         spinor_dble *rd)
{
   int *imm;
   spinor *si;

   imm=imb+vol;

   while (imb<imm)
   {
      si=s+(*imb);
      imb+=1;

      if (imb<imm)
         _prefetch_spinor(s+(*imb));

      _sse_load_cvt((*si).c1);
      _sse_su3_inverse_multiply_dble(*ud);
      _sse_store_up_dble((*rd).c1);

      _sse_load_cvt((*si).c2);
      _sse_su3_inverse_multiply_dble(*ud);
      _sse_store_up_dble((*rd).c2);

      _sse_load_cvt((*si).c3);
      _sse_su3_inverse_multiply_dble(*ud);
      _sse_store_up_dble((*rd).c3);

      _sse_load_cvt((*si).c4);
      _sse_su3_inverse_multiply_dble(*ud);
      _sse_store_up_dble((*rd).c4);

      ud+=1;
      rd+=1;
   }
}


static void spinor_prod_gamma0(int vol,spinor_dble *sd,spinor_dble *rd,
                               complex_dble *sp)
{
   complex_dble sm[2] ALIGNED16;
   complex_qflt qsm0,qsm1;
   spinor_dble *rt,*rm;

   qsm0.re.q[0]=0.0;
   qsm0.re.q[1]=0.0;
   qsm0.im.q[0]=0.0;
   qsm0.im.q[1]=0.0;

   qsm1.re.q[0]=0.0;
   qsm1.re.q[1]=0.0;
   qsm1.im.q[0]=0.0;
   qsm1.im.q[1]=0.0;

   rt=rd+vol;
   rm=rd;

   while (rm<rt)
   {
      rm+=8;
      if (rm>rt)
         rm=rt;

      _start_sm();

      for (;rd<rm;rd++)
      {
         _load_chi((*rd).c1);
         _load_psi0((*sd).c1);
         _load_psi1_add((*sd).c3);

         _load_chi((*rd).c2);
         _load_psi0((*sd).c2);
         _load_psi1_add((*sd).c4);

         _load_chi((*rd).c3);
         _load_psi0((*sd).c3);
         _load_psi1_add((*sd).c1);

         _load_chi((*rd).c4);
         _load_psi0((*sd).c4);
         _load_psi1_add((*sd).c2);

         sd+=1;
      }

      _store_sm();

      acc_qflt(sm[0].re,qsm0.re.q);
      acc_qflt(sm[0].im,qsm0.im.q);

      acc_qflt(sm[1].re,qsm1.re.q);
      acc_qflt(sm[1].im,qsm1.im.q);
   }

   sp[0].re=qsm0.re.q[0];
   sp[0].im=qsm0.im.q[0];

   sp[1].re=-qsm1.re.q[0];
   sp[1].im=-qsm1.im.q[0];
}


static void spinor_prod_gamma1(int vol,spinor_dble *sd,spinor_dble *rd,
                               complex_dble *sp)
{
   complex_dble sm[2] ALIGNED16;
   complex_qflt qsm0,qsm1;
   spinor_dble *rt,*rm;

   qsm0.re.q[0]=0.0;
   qsm0.re.q[1]=0.0;
   qsm0.im.q[0]=0.0;
   qsm0.im.q[1]=0.0;

   qsm1.re.q[0]=0.0;
   qsm1.re.q[1]=0.0;
   qsm1.im.q[0]=0.0;
   qsm1.im.q[1]=0.0;

   rt=rd+vol;
   rm=rd;

   while (rm<rt)
   {
      rm+=8;
      if (rm>rt)
         rm=rt;

      _start_sm();

      for (;rd<rm;rd++)
      {
         _load_chi((*rd).c1);
         _load_psi0((*sd).c1);
         _load_psi1_sub((*sd).c4);

         _load_chi((*rd).c2);
         _load_psi0((*sd).c2);
         _load_psi1_sub((*sd).c3);

         _load_chi((*rd).c3);
         _load_psi0((*sd).c3);
         _load_psi1_add((*sd).c2);

         _load_chi((*rd).c4);
         _load_psi0((*sd).c4);
         _load_psi1_add((*sd).c1);

         sd+=1;
      }

      _store_sm();

      acc_qflt(sm[0].re,qsm0.re.q);
      acc_qflt(sm[0].im,qsm0.im.q);

      acc_qflt(sm[1].re,qsm1.re.q);
      acc_qflt(sm[1].im,qsm1.im.q);
   }

   sp[0].re=qsm0.re.q[0];
   sp[0].im=qsm0.im.q[0];

   sp[1].re=qsm1.im.q[0];
   sp[1].im=-qsm1.re.q[0];
}


static void spinor_prod_gamma2(int vol,spinor_dble *sd,spinor_dble *rd,
                               complex_dble *sp)
{
   complex_dble sm[2] ALIGNED16;
   complex_qflt qsm0,qsm1;
   spinor_dble *rt,*rm;

   qsm0.re.q[0]=0.0;
   qsm0.re.q[1]=0.0;
   qsm0.im.q[0]=0.0;
   qsm0.im.q[1]=0.0;

   qsm1.re.q[0]=0.0;
   qsm1.re.q[1]=0.0;
   qsm1.im.q[0]=0.0;
   qsm1.im.q[1]=0.0;

   rt=rd+vol;
   rm=rd;

   while (rm<rt)
   {
      rm+=8;
      if (rm>rt)
         rm=rt;

      _start_sm();

      for (;rd<rm;rd++)
      {
         _load_chi((*rd).c1);
         _load_psi0((*sd).c1);
         _load_psi1_add((*sd).c4);

         _load_chi((*rd).c2);
         _load_psi0((*sd).c2);
         _load_psi1_sub((*sd).c3);

         _load_chi((*rd).c3);
         _load_psi0((*sd).c3);
         _load_psi1_sub((*sd).c2);

         _load_chi((*rd).c4);
         _load_psi0((*sd).c4);
         _load_psi1_add((*sd).c1);

         sd+=1;
      }

      _store_sm();

      acc_qflt(sm[0].re,qsm0.re.q);
      acc_qflt(sm[0].im,qsm0.im.q);

      acc_qflt(sm[1].re,qsm1.re.q);
      acc_qflt(sm[1].im,qsm1.im.q);
   }

   sp[0].re=qsm0.re.q[0];
   sp[0].im=qsm0.im.q[0];

   sp[1].re=-qsm1.re.q[0];
   sp[1].im=-qsm1.im.q[0];
}


static void spinor_prod_gamma3(int vol,spinor_dble *sd,spinor_dble *rd,
                               complex_dble *sp)
{
   complex_dble sm[2] ALIGNED16;
   complex_qflt qsm0,qsm1;
   spinor_dble *rt,*rm;

   qsm0.re.q[0]=0.0;
   qsm0.re.q[1]=0.0;
   qsm0.im.q[0]=0.0;
   qsm0.im.q[1]=0.0;

   qsm1.re.q[0]=0.0;
   qsm1.re.q[1]=0.0;
   qsm1.im.q[0]=0.0;
   qsm1.im.q[1]=0.0;

   rt=rd+vol;
   rm=rd;

   while (rm<rt)
   {
      rm+=8;
      if (rm>rt)
         rm=rt;

      _start_sm();

      for (;rd<rm;rd++)
      {
         _load_chi((*rd).c1);
         _load_psi0((*sd).c1);
         _load_psi1_sub((*sd).c3);

         _load_chi((*rd).c2);
         _load_psi0((*sd).c2);
         _load_psi1_add((*sd).c4);

         _load_chi((*rd).c3);
         _load_psi0((*sd).c3);
         _load_psi1_add((*sd).c1);

         _load_chi((*rd).c4);
         _load_psi0((*sd).c4);
         _load_psi1_sub((*sd).c2);

         sd+=1;
      }

      _store_sm();

      acc_qflt(sm[0].re,qsm0.re.q);
      acc_qflt(sm[0].im,qsm0.im.q);

      acc_qflt(sm[1].re,qsm1.re.q);
      acc_qflt(sm[1].im,qsm1.im.q);
   }

   sp[0].re=qsm0.re.q[0];
   sp[0].im=qsm0.im.q[0];

   sp[1].re=qsm1.im.q[0];
   sp[1].im=-qsm1.re.q[0];
}

#else

static void apply_u2s(int vol,int *imb,su3_dble *ud,spinor *s,
                      spinor_dble *rd)
{
   int *imm;
   spinor *si;

   imm=imb+vol;

   for (;imb<imm;imb++)
   {
      si=s+(*imb);

      _su3_multiply((*rd).c1,(*ud),(*si).c1);
      _su3_multiply((*rd).c2,(*ud),(*si).c2);
      _su3_multiply((*rd).c3,(*ud),(*si).c3);
      _su3_multiply((*rd).c4,(*ud),(*si).c4);

      ud+=1;
      rd+=1;
   }
}


static void apply_udag2s(int vol,int *imb,su3_dble *ud,spinor *s,
                         spinor_dble *rd)
{
   int *imm;
   spinor *si;

   imm=imb+vol;

   for (;imb<imm;imb++)
   {
      si=s+(*imb);

      _su3_inverse_multiply((*rd).c1,(*ud),(*si).c1);
      _su3_inverse_multiply((*rd).c2,(*ud),(*si).c2);
      _su3_inverse_multiply((*rd).c3,(*ud),(*si).c3);
      _su3_inverse_multiply((*rd).c4,(*ud),(*si).c4);

      ud+=1;
      rd+=1;
   }
}


static void spinor_prod_gamma0(int vol,spinor_dble *sd,spinor_dble *rd,
                               complex_dble *sp)
{
   complex_dble z0,z1;
   complex_qflt qsm0,qsm1;
   spinor_dble *rt,*rm;

   qsm0.re.q[0]=0.0;
   qsm0.re.q[1]=0.0;
   qsm0.im.q[0]=0.0;
   qsm0.im.q[1]=0.0;

   qsm1.re.q[0]=0.0;
   qsm1.re.q[1]=0.0;
   qsm1.im.q[0]=0.0;
   qsm1.im.q[1]=0.0;

   rt=rd+vol;
   rm=rd;

   while (rm<rt)
   {
      rm+=8;
      if (rm>rt)
         rm=rt;

      z0.re=0.0;
      z0.im=0.0;
      z1.re=0.0;
      z1.im=0.0;

      for (;rd<rm;rd++)
      {
         z0.re+=(_vector_prod_re((*sd).c1,(*rd).c1)+
                 _vector_prod_re((*sd).c2,(*rd).c2)+
                 _vector_prod_re((*sd).c3,(*rd).c3)+
                 _vector_prod_re((*sd).c4,(*rd).c4));

         z0.im+=(_vector_prod_im((*sd).c1,(*rd).c1)+
                 _vector_prod_im((*sd).c2,(*rd).c2)+
                 _vector_prod_im((*sd).c3,(*rd).c3)+
                 _vector_prod_im((*sd).c4,(*rd).c4));

         z1.re+=(_vector_prod_re((*sd).c1,(*rd).c3)+
                 _vector_prod_re((*sd).c2,(*rd).c4)+
                 _vector_prod_re((*sd).c3,(*rd).c1)+
                 _vector_prod_re((*sd).c4,(*rd).c2));

         z1.im+=(_vector_prod_im((*sd).c1,(*rd).c3)+
                 _vector_prod_im((*sd).c2,(*rd).c4)+
                 _vector_prod_im((*sd).c3,(*rd).c1)+
                 _vector_prod_im((*sd).c4,(*rd).c2));

         sd+=1;
      }

      acc_qflt(z0.re,qsm0.re.q);
      acc_qflt(z0.im,qsm0.im.q);

      acc_qflt(z1.re,qsm1.re.q);
      acc_qflt(z1.im,qsm1.im.q);
   }

   sp[0].re=qsm0.re.q[0];
   sp[0].im=qsm0.im.q[0];

   sp[1].re=-qsm1.re.q[0];
   sp[1].im=-qsm1.im.q[0];
}


static void spinor_prod_gamma1(int vol,spinor_dble *sd,spinor_dble *rd,
                               complex_dble *sp)
{
   complex_dble z0,z1;
   complex_qflt qsm0,qsm1;
   spinor_dble *rt,*rm;

   qsm0.re.q[0]=0.0;
   qsm0.re.q[1]=0.0;
   qsm0.im.q[0]=0.0;
   qsm0.im.q[1]=0.0;

   qsm1.re.q[0]=0.0;
   qsm1.re.q[1]=0.0;
   qsm1.im.q[0]=0.0;
   qsm1.im.q[1]=0.0;

   rt=rd+vol;
   rm=rd;

   while (rm<rt)
   {
      rm+=8;
      if (rm>rt)
         rm=rt;

      z0.re=0.0;
      z0.im=0.0;
      z1.re=0.0;
      z1.im=0.0;

      for (;rd<rm;rd++)
      {
         z0.re+=(_vector_prod_re((*sd).c1,(*rd).c1)+
                 _vector_prod_re((*sd).c2,(*rd).c2)+
                 _vector_prod_re((*sd).c3,(*rd).c3)+
                 _vector_prod_re((*sd).c4,(*rd).c4));

         z0.im+=(_vector_prod_im((*sd).c1,(*rd).c1)+
                 _vector_prod_im((*sd).c2,(*rd).c2)+
                 _vector_prod_im((*sd).c3,(*rd).c3)+
                 _vector_prod_im((*sd).c4,(*rd).c4));

         z1.re+=(_vector_prod_re((*sd).c1,(*rd).c4)+
                 _vector_prod_re((*sd).c2,(*rd).c3));

         z1.re-=(_vector_prod_re((*sd).c3,(*rd).c2)+
                 _vector_prod_re((*sd).c4,(*rd).c1));

         z1.im+=(_vector_prod_im((*sd).c1,(*rd).c4)+
                 _vector_prod_im((*sd).c2,(*rd).c3));

         z1.im-=(_vector_prod_im((*sd).c3,(*rd).c2)+
                 _vector_prod_im((*sd).c4,(*rd).c1));

         sd+=1;
      }

      acc_qflt(z0.re,qsm0.re.q);
      acc_qflt(z0.im,qsm0.im.q);

      acc_qflt(z1.re,qsm1.re.q);
      acc_qflt(z1.im,qsm1.im.q);
   }

   sp[0].re=qsm0.re.q[0];
   sp[0].im=qsm0.im.q[0];

   sp[1].re=qsm1.im.q[0];
   sp[1].im=-qsm1.re.q[0];
}


static void spinor_prod_gamma2(int vol,spinor_dble *sd,spinor_dble *rd,
                               complex_dble *sp)
{
   complex_dble z0,z1;
   complex_qflt qsm0,qsm1;
   spinor_dble *rt,*rm;

   qsm0.re.q[0]=0.0;
   qsm0.re.q[1]=0.0;
   qsm0.im.q[0]=0.0;
   qsm0.im.q[1]=0.0;

   qsm1.re.q[0]=0.0;
   qsm1.re.q[1]=0.0;
   qsm1.im.q[0]=0.0;
   qsm1.im.q[1]=0.0;

   rt=rd+vol;
   rm=rd;

   while (rm<rt)
   {
      rm+=8;
      if (rm>rt)
         rm=rt;

      z0.re=0.0;
      z0.im=0.0;
      z1.re=0.0;
      z1.im=0.0;

      for (;rd<rm;rd++)
      {
         z0.re+=(_vector_prod_re((*sd).c1,(*rd).c1)+
                 _vector_prod_re((*sd).c2,(*rd).c2)+
                 _vector_prod_re((*sd).c3,(*rd).c3)+
                 _vector_prod_re((*sd).c4,(*rd).c4));

         z0.im+=(_vector_prod_im((*sd).c1,(*rd).c1)+
                 _vector_prod_im((*sd).c2,(*rd).c2)+
                 _vector_prod_im((*sd).c3,(*rd).c3)+
                 _vector_prod_im((*sd).c4,(*rd).c4));

         z1.re+=(_vector_prod_re((*sd).c1,(*rd).c4)+
                 _vector_prod_re((*sd).c4,(*rd).c1));

         z1.re-=(_vector_prod_re((*sd).c2,(*rd).c3)+
                 _vector_prod_re((*sd).c3,(*rd).c2));

         z1.im+=(_vector_prod_im((*sd).c1,(*rd).c4)+
                 _vector_prod_im((*sd).c4,(*rd).c1));

         z1.im-=(_vector_prod_im((*sd).c2,(*rd).c3)+
                 _vector_prod_im((*sd).c3,(*rd).c2));

         sd+=1;
      }

      acc_qflt(z0.re,qsm0.re.q);
      acc_qflt(z0.im,qsm0.im.q);

      acc_qflt(z1.re,qsm1.re.q);
      acc_qflt(z1.im,qsm1.im.q);
   }

   sp[0].re=qsm0.re.q[0];
   sp[0].im=qsm0.im.q[0];

   sp[1].re=-qsm1.re.q[0];
   sp[1].im=-qsm1.im.q[0];
}


static void spinor_prod_gamma3(int vol,spinor_dble *sd,spinor_dble *rd,
                               complex_dble *sp)
{
   complex_dble z0,z1;
   complex_qflt qsm0,qsm1;
   spinor_dble *rt,*rm;

   qsm0.re.q[0]=0.0;
   qsm0.re.q[1]=0.0;
   qsm0.im.q[0]=0.0;
   qsm0.im.q[1]=0.0;

   qsm1.re.q[0]=0.0;
   qsm1.re.q[1]=0.0;
   qsm1.im.q[0]=0.0;
   qsm1.im.q[1]=0.0;

   rt=rd+vol;
   rm=rd;

   while (rm<rt)
   {
      rm+=8;
      if (rm>rt)
         rm=rt;

      z0.re=0.0;
      z0.im=0.0;
      z1.re=0.0;
      z1.im=0.0;

      for (;rd<rm;rd++)
      {
         z0.re+=(_vector_prod_re((*sd).c1,(*rd).c1)+
                 _vector_prod_re((*sd).c2,(*rd).c2)+
                 _vector_prod_re((*sd).c3,(*rd).c3)+
                 _vector_prod_re((*sd).c4,(*rd).c4));

         z0.im+=(_vector_prod_im((*sd).c1,(*rd).c1)+
                 _vector_prod_im((*sd).c2,(*rd).c2)+
                 _vector_prod_im((*sd).c3,(*rd).c3)+
                 _vector_prod_im((*sd).c4,(*rd).c4));

         z1.re+=(_vector_prod_re((*sd).c1,(*rd).c3)+
                 _vector_prod_re((*sd).c4,(*rd).c2));

         z1.re-=(_vector_prod_re((*sd).c2,(*rd).c4)+
                 _vector_prod_re((*sd).c3,(*rd).c1));

         z1.im+=(_vector_prod_im((*sd).c1,(*rd).c3)+
                 _vector_prod_im((*sd).c4,(*rd).c2));

         z1.im-=(_vector_prod_im((*sd).c2,(*rd).c4)+
                 _vector_prod_im((*sd).c3,(*rd).c1));

         sd+=1;
      }

      acc_qflt(z0.re,qsm0.re.q);
      acc_qflt(z0.im,qsm0.im.q);

      acc_qflt(z1.re,qsm1.re.q);
      acc_qflt(z1.im,qsm1.im.q);
   }

   sp[0].re=qsm0.re.q[0];
   sp[0].im=qsm0.im.q[0];

   sp[1].re=qsm1.im.q[0];
   sp[1].im=-qsm1.re.q[0];
}

#endif

void gather_so(int n,int ifc,block_t *b,spinor_dble *sbuf)
{
   int i,volh,*ipp;
   spinor **s;
   su3_dble *ud;

   volh=(*b).bb[ifc].vol/2;
   ipp=(*b).bb[ifc].ipp;
   s=(*b).s;
   ud=(*b).bb[ifc].ud;

   if (ifc&0x1)
   {
      for (i=1;i<=n;i++)
      {
         apply_udag2s(volh,ipp,ud,s[i],sbuf);
         sbuf+=volh;
      }
   }
   else
   {
      for (i=1;i<=n;i++)
      {
         apply_u2s(volh,ipp,ud,s[i],sbuf);
         sbuf+=volh;
      }
   }
}


void (*spinor_prod_gamma[4])
(int vol,spinor_dble *sd,spinor_dble *rd,complex_dble *sp)=
{spinor_prod_gamma0,spinor_prod_gamma1,spinor_prod_gamma2,spinor_prod_gamma3};
