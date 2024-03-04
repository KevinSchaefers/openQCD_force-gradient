
/*******************************************************************************
*
* File frcfcts.c
*
* Copyright (C) 2005, 2011-2016, 2018, 2021 Martin Luescher, Stefan Schaefer
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic functions used for the force calculation.
*
*   void cmb6weyl(double *c,weyl_dble *s,weyl_dble *r)
*     Assigns the real linear combination c[0]*s[0]+..+c[5]*s[5] of 6
*     Weyl spinors to the Weyl spinor r[0]. The latter is assumed to
*     be different from s[0],..,s[5].
*
*   void det2xt(pauli_dble *m,u3_alg_dble *X)
*     Computes the matrices X[0],..,X[5] associated with the SW term on a
*     given lattice point (see the notes). The program expects that m[0]
*     and m[1] contain the Hermitian part of the inverse of the SW term
*     at the chosen point.
*
*   void prod2xt(spinor_dble *r,spinor_dble *s,u3_alg_dble *X)
*     Computes the matrices X[0],..,X[5] associated to a pair of spinors
*     r and s at a given lattice point (see the notes).
*
* The following is an array of functions indexed by the direction mu=0,..,3:
*
*   void (*prod2xv[])(spinor_dble *rx,spinor_dble *ry,
*                     spinor_dble *sx,spinor_dble *sy,su3_dble *u)
*     Computes the complex 3x3 matrix
*
*       u=tr{gamma_5*(1-gamma_mu)*[(sy x rx^dag)+(ry x sx^dag)]}
*
*     where ..x.. denotes the tensor product in spinor space and the trace
*     is taken over the Dirac indices.
*
* The programs in this module serve to compute the spin part of the quark
* forces. See the notes "Molecular-dynamics quark forces" [doc/forces.pdf]
* for detailed explanations. The data type u3_alg_dble is described at the
* top of the module su3fcts/su3prod.c.
*
* The matrices computed by the program det2xt() are
*
*  X[n]=i*tr{sigma_{mu,nu}*diag(m[0],m[1])}
*
* where (mu,nu)=(0,1),(0,2),(0,3),(2,3),(3,1),(1,2) for n=0,..,5. Similarly,
* the program prod2xt() computes
*
*  X[n]=i*tr{(gamma_5*sigma_{mu,nu}*s) x (r^dag)+(s<->r)}
*
* where ..x.. denotes the tensor product in spinor space. In both cases,
* the trace is taken over the Dirac indices only.
*
* The programs in this module are thread-safe. If SSE (AVX) instructions are
* used, the Weyl spinors must be aligned to a 16 byte boundary.
*
*******************************************************************************/

#define FRCFCTS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "su3.h"
#include "forces.h"

#define _re(z,w) ((z).re*(w).re+(z).im*(w).im)
#define _im(z,w) ((z).im*(w).re-(z).re*(w).im)

typedef union
{
   spinor_dble s;
   weyl_dble w[2];
} spin_dble;

typedef union
{
   weyl_dble w;
   complex_dble c[6];
} hspin_dble;

#if (defined x64)

void cmb6weyl(double *c,weyl_dble *s,weyl_dble *r)
{
   hspin_dble *hs,*hr,*hm;

   hs=(hspin_dble*)(s);
   hr=(hspin_dble*)(r);

   __asm__ __volatile__ ("movddup %0, %%xmm0\n\t"
                         "movapd %1, %%xmm3 \n\t"
                         "movapd %2, %%xmm4 \n\t"
                         "movapd %%xmm0, %%xmm1 \n\t"
                         "movapd %3, %%xmm5 \n\t"
                         "movapd %4, %%xmm6 \n\t"
                         "movapd %%xmm0, %%xmm2 \n\t"
                         "movapd %5, %%xmm7 \n\t"
                         "movapd %6, %%xmm8"
                         :
                         :
                         "m" (c[0]),
                         "m" ((*hs).c[0]),
                         "m" ((*hs).c[1]),
                         "m" ((*hs).c[2]),
                         "m" ((*hs).c[3]),
                         "m" ((*hs).c[4]),
                         "m" ((*hs).c[5])
                         :
                         "xmm0", "xmm1", "xmm2",
                         "xmm3", "xmm4", "xmm5",
                         "xmm6", "xmm7", "xmm8");

   __asm__ __volatile__ ("mulpd %%xmm0, %%xmm3 \n\t"
                         "mulpd %%xmm1, %%xmm4 \n\t"
                         "mulpd %%xmm2, %%xmm5 \n\t"
                         "mulpd %%xmm0, %%xmm6 \n\t"
                         "mulpd %%xmm1, %%xmm7 \n\t"
                         "mulpd %%xmm2, %%xmm8"
                         :
                         :
                         :
                         "xmm3", "xmm4", "xmm5",
                         "xmm6", "xmm7", "xmm8");

   hm=hs+6;
   hs+=1;
   c+=1;

   for (;hs<hm;hs++)
   {
      __asm__ __volatile__ ("movddup %0, %%xmm0 \n\t"
                            "movapd %1, %%xmm9 \n\t"
                            "movapd %2, %%xmm10 \n\t"
                            "movapd %%xmm0, %%xmm1 \n\t"
                            "movapd %3, %%xmm11 \n\t"
                            "movapd %4, %%xmm12 \n\t"
                            "movapd %%xmm0, %%xmm2 \n\t"
                            "movapd %5, %%xmm13 \n\t"
                            "movapd %6, %%xmm14"
                            :
                            :
                            "m" (c[0]),
                            "m" ((*hs).c[0]),
                            "m" ((*hs).c[1]),
                            "m" ((*hs).c[2]),
                            "m" ((*hs).c[3]),
                            "m" ((*hs).c[4]),
                            "m" ((*hs).c[5])
                            :
                            "xmm0", "xmm1", "xmm2",
                            "xmm9", "xmm10", "xmm11",
                            "xmm12", "xmm13", "xmm14");

      __asm__ __volatile__ ("mulpd %%xmm0, %%xmm9 \n\t"
                            "mulpd %%xmm1, %%xmm10 \n\t"
                            "mulpd %%xmm2, %%xmm11 \n\t"
                            "mulpd %%xmm0, %%xmm12 \n\t"
                            "mulpd %%xmm1, %%xmm13 \n\t"
                            "mulpd %%xmm2, %%xmm14 \n\t"
                            "addpd %%xmm9, %%xmm3 \n\t"
                            "addpd %%xmm10, %%xmm4 \n\t"
                            "addpd %%xmm11, %%xmm5 \n\t"
                            "addpd %%xmm12, %%xmm6 \n\t"
                            "addpd %%xmm13, %%xmm7 \n\t"
                            "addpd %%xmm14, %%xmm8"
                            :
                            :
                            :
                            "xmm3", "xmm4", "xmm5",
                            "xmm6", "xmm7", "xmm8",
                            "xmm9", "xmm10", "xmm11",
                            "xmm12", "xmm13", "xmm14");

      c+=1;
   }

   __asm__ __volatile__ ("movapd %%xmm3, %0 \n\t"
                         "movapd %%xmm4, %1 \n\t"
                         "movapd %%xmm5, %2 \n\t"
                         "movapd %%xmm6, %3 \n\t"
                         "movapd %%xmm7, %4 \n\t"
                         "movapd %%xmm8, %5"
                         :
                         "=m" ((*hr).c[0]),
                         "=m" ((*hr).c[1]),
                         "=m" ((*hr).c[2]),
                         "=m" ((*hr).c[3]),
                         "=m" ((*hr).c[4]),
                         "=m" ((*hr).c[5])
                         :
                         :);
}


static void vec2pauli(weyl_dble *r,weyl_dble *s,pauli_dble *m)
{
   double *u;
   hspin_dble *hr,*hs;
   complex_dble *cr,*cs,*crp,*csp,*crm,*csm;

   u=(*m).u;
   hr=(hspin_dble*)(r);
   hs=(hspin_dble*)(s);
   cr=(*hr).c;
   cs=(*hs).c;

   __asm__ __volatile__ ("movapd %0, %%xmm0 \n\t"
                         "movapd %1, %%xmm1 \n\t"
                         "movapd %2, %%xmm2 \n\t"
                         "movapd %3, %%xmm3 \n\t"
                         "movapd %4, %%xmm4 \n\t"
                         "movapd %5, %%xmm5"
                         :
                         :
                         "m" (cr[0]),
                         "m" (cr[1]),
                         "m" (cr[2]),
                         "m" (cr[3]),
                         "m" (cr[4]),
                         "m" (cr[5])
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
                         "m" (cs[0]),
                         "m" (cs[1]),
                         "m" (cs[2]),
                         "m" (cs[3]),
                         "m" (cs[4]),
                         "m" (cs[5])
                         :
                         "xmm0", "xmm1", "xmm2",
                         "xmm3", "xmm4", "xmm5");

   __asm__ __volatile__ ("haddpd %%xmm1, %%xmm0 \n\t"
                         "haddpd %%xmm3, %%xmm2 \n\t"
                         "haddpd %%xmm5, %%xmm4 \n\t"
                         "addpd %%xmm0, %%xmm0 \n\t"
                         "addpd %%xmm2, %%xmm2 \n\t"
                         "addpd %%xmm4, %%xmm4 \n\t"
                         "movapd %%xmm0, %0 \n\t"
                         "movapd %%xmm2, %2 \n\t"
                         "movapd %%xmm4, %4"
                         :
                         "=m" (u[0]),
                         "=m" (u[1]),
                         "=m" (u[2]),
                         "=m" (u[3]),
                         "=m" (u[4]),
                         "=m" (u[5])
                         :
                         :
                         "xmm0", "xmm2", "xmm4");

   u+=6;
   crm=cr+5;
   csm=cs+6;

   for (;cr<crm;cr++)
   {
      __asm__ __volatile__ ("movapd %0, %%xmm10 \n\t"
                            "movapd %1, %%xmm11 \n\t"
                            "movapd %0, %%xmm12 \n\t"
                            "movapd %1, %%xmm13 \n\t"
                            "shufpd $0x1, %%xmm10, %%xmm10 \n\t"
                            "shufpd $0x1, %%xmm11, %%xmm11"
                            :
                            :
                            "m" (cr[0]),
                            "m" (cs[0])
                            :
                            "xmm10", "xmm11", "xmm12", "xmm13");

      crp=cr+1;

      for (csp=cs+1;csp<csm;csp++)
      {
         __asm__ __volatile__ ("movddup %0, %%xmm6 \n\t"
                               "movddup %1, %%xmm7 \n\t"
                               "movddup %2, %%xmm8 \n\t"
                               "movddup %3, %%xmm9"
                               :
                               :
                               "m" (csp[0].re),
                               "m" (crp[0].re),
                               "m" (csp[0].im),
                               "m" (crp[0].im)
                               :
                               "xmm6", "xmm7", "xmm8", "xmm9");

         __asm__ __volatile__ ("mulpd %%xmm10, %%xmm6 \n\t"
                               "mulpd %%xmm11, %%xmm7 \n\t"
                               "mulpd %%xmm12, %%xmm8 \n\t"
                               "mulpd %%xmm13, %%xmm9 \n\t"
                               "addsubpd %%xmm8, %%xmm6 \n\t"
                               "addsubpd %%xmm9, %%xmm7 \n\t"
                               "addpd %%xmm7, %%xmm6 \n\t"
                               "shufpd $0x1, %%xmm6, %%xmm6 \n\t"
                               "movapd %%xmm6, %0"
                               :
                               "=m" (u[0]),
                               "=m" (u[1])
                               :
                               :
                               "xmm6", "xmm7", "xmm8", "xmm9");

         crp+=1;
         u+=2;
      }

      cs+=1;
   }
}

#else

void cmb6weyl(double *c,weyl_dble *s,weyl_dble *r)
{
   weyl_dble *sm;

   sm=s+6;

   _vector_mul((*r).c1,c[0],(*s).c1);
   _vector_mul((*r).c2,c[0],(*s).c2);
   c+=1;
   s+=1;

   for (;s<sm;s++)
   {
      _vector_mulr_assign((*r).c1,c[0],(*s).c1);
      _vector_mulr_assign((*r).c2,c[0],(*s).c2);
      c+=1;
   }
}


static void vec2pauli(weyl_dble *r,weyl_dble *s,pauli_dble *m)
{
   double *u;
   su3_vector_dble *r1,*r2,*s1,*s2;

   u=(*m).u;
   r1=&((*r).c1);
   r2=&((*r).c2);
   s1=&((*s).c1);
   s2=&((*s).c2);

   u[ 0]=_re((*s1).c1,(*r1).c1)+_re((*s1).c1,(*r1).c1);
   u[ 1]=_re((*s1).c2,(*r1).c2)+_re((*s1).c2,(*r1).c2);
   u[ 2]=_re((*s1).c3,(*r1).c3)+_re((*s1).c3,(*r1).c3);

   u[ 3]=_re((*s2).c1,(*r2).c1)+_re((*s2).c1,(*r2).c1);
   u[ 4]=_re((*s2).c2,(*r2).c2)+_re((*s2).c2,(*r2).c2);
   u[ 5]=_re((*s2).c3,(*r2).c3)+_re((*s2).c3,(*r2).c3);

   u[ 6]=_re((*s1).c1,(*r1).c2)+_re((*r1).c1,(*s1).c2);
   u[ 7]=_im((*s1).c1,(*r1).c2)+_im((*r1).c1,(*s1).c2);
   u[ 8]=_re((*s1).c1,(*r1).c3)+_re((*r1).c1,(*s1).c3);
   u[ 9]=_im((*s1).c1,(*r1).c3)+_im((*r1).c1,(*s1).c3);

   u[10]=_re((*s1).c1,(*r2).c1)+_re((*r1).c1,(*s2).c1);
   u[11]=_im((*s1).c1,(*r2).c1)+_im((*r1).c1,(*s2).c1);
   u[12]=_re((*s1).c1,(*r2).c2)+_re((*r1).c1,(*s2).c2);
   u[13]=_im((*s1).c1,(*r2).c2)+_im((*r1).c1,(*s2).c2);
   u[14]=_re((*s1).c1,(*r2).c3)+_re((*r1).c1,(*s2).c3);
   u[15]=_im((*s1).c1,(*r2).c3)+_im((*r1).c1,(*s2).c3);

   u[16]=_re((*s1).c2,(*r1).c3)+_re((*r1).c2,(*s1).c3);
   u[17]=_im((*s1).c2,(*r1).c3)+_im((*r1).c2,(*s1).c3);

   u[18]=_re((*s1).c2,(*r2).c1)+_re((*r1).c2,(*s2).c1);
   u[19]=_im((*s1).c2,(*r2).c1)+_im((*r1).c2,(*s2).c1);
   u[20]=_re((*s1).c2,(*r2).c2)+_re((*r1).c2,(*s2).c2);
   u[21]=_im((*s1).c2,(*r2).c2)+_im((*r1).c2,(*s2).c2);
   u[22]=_re((*s1).c2,(*r2).c3)+_re((*r1).c2,(*s2).c3);
   u[23]=_im((*s1).c2,(*r2).c3)+_im((*r1).c2,(*s2).c3);

   u[24]=_re((*s1).c3,(*r2).c1)+_re((*r1).c3,(*s2).c1);
   u[25]=_im((*s1).c3,(*r2).c1)+_im((*r1).c3,(*s2).c1);
   u[26]=_re((*s1).c3,(*r2).c2)+_re((*r1).c3,(*s2).c2);
   u[27]=_im((*s1).c3,(*r2).c2)+_im((*r1).c3,(*s2).c2);
   u[28]=_re((*s1).c3,(*r2).c3)+_re((*r1).c3,(*s2).c3);
   u[29]=_im((*s1).c3,(*r2).c3)+_im((*r1).c3,(*s2).c3);

   u[30]=_re((*s2).c1,(*r2).c2)+_re((*r2).c1,(*s2).c2);
   u[31]=_im((*s2).c1,(*r2).c2)+_im((*r2).c1,(*s2).c2);
   u[32]=_re((*s2).c1,(*r2).c3)+_re((*r2).c1,(*s2).c3);
   u[33]=_im((*s2).c1,(*r2).c3)+_im((*r2).c1,(*s2).c3);
   u[34]=_re((*s2).c2,(*r2).c3)+_re((*r2).c2,(*s2).c3);
   u[35]=_im((*s2).c2,(*r2).c3)+_im((*r2).c2,(*s2).c3);
}

#endif

void det2xt(pauli_dble *m,u3_alg_dble *X)
{
   double x,*up,*um;
   u3_alg_dble *X0,*X1;

   up=m[0].u;
   um=m[1].u;

   X0=X;
   X1=X+3;

   x=up[10]+up[10];
   (*X0).c1=x;
   (*X1).c1=-x;
   x=um[10]+um[10];
   (*X0).c1-=x;
   (*X1).c1-=x;

   x=up[20]+up[20];
   (*X0).c2=x;
   (*X1).c2=-x;
   x=um[20]+um[20];
   (*X0).c2-=x;
   (*X1).c2-=x;

   x=up[28]+up[28];
   (*X0).c3=x;
   (*X1).c3=-x;
   x=um[28]+um[28];
   (*X0).c3-=x;
   (*X1).c3-=x;

   x=up[19]-up[13];
   (*X0).c4=x;
   (*X1).c4=-x;
   x=um[19]-um[13];
   (*X0).c4-=x;
   (*X1).c4-=x;

   x=up[12]+up[18];
   (*X0).c5=x;
   (*X1).c5=-x;
   x=um[12]+um[18];
   (*X0).c5-=x;
   (*X1).c5-=x;

   x=up[25]-up[15];
   (*X0).c6=x;
   (*X1).c6=-x;
   x=um[25]-um[15];
   (*X0).c6-=x;
   (*X1).c6-=x;

   x=up[14]+up[24];
   (*X0).c7=x;
   (*X1).c7=-x;
   x=um[14]+um[24];
   (*X0).c7-=x;
   (*X1).c7-=x;

   x=up[27]-up[23];
   (*X0).c8=x;
   (*X1).c8=-x;
   x=um[27]-um[23];
   (*X0).c8-=x;
   (*X1).c8-=x;

   x=up[22]+up[26];
   (*X0).c9=x;
   (*X1).c9=-x;
   x=um[22]+um[26];
   (*X0).c9-=x;
   (*X1).c9-=x;

   X0=X+1;
   X1=X+4;

   x=up[11]+up[11];
   (*X0).c1=-x;
   (*X1).c1=x;
   x=um[11]+um[11];
   (*X0).c1+=x;
   (*X1).c1+=x;

   x=up[21]+up[21];
   (*X0).c2=-x;
   (*X1).c2=x;
   x=um[21]+um[21];
   (*X0).c2+=x;
   (*X1).c2+=x;

   x=up[29]+up[29];
   (*X0).c3=-x;
   (*X1).c3=x;
   x=um[29]+um[29];
   (*X0).c3+=x;
   (*X1).c3+=x;

   x=up[18]-up[12];
   (*X0).c4=x;
   (*X1).c4=-x;
   x=um[18]-um[12];
   (*X0).c4-=x;
   (*X1).c4-=x;

   x=up[13]+up[19];
   (*X0).c5=-x;
   (*X1).c5=x;
   x=um[13]+um[19];
   (*X0).c5+=x;
   (*X1).c5+=x;

   x=up[24]-up[14];
   (*X0).c6=x;
   (*X1).c6=-x;
   x=um[24]-um[14];
   (*X0).c6-=x;
   (*X1).c6-=x;

   x=up[25]+up[15];
   (*X0).c7=-x;
   (*X1).c7=x;
   x=um[25]+um[15];
   (*X0).c7+=x;
   (*X1).c7+=x;

   x=up[26]-up[22];
   (*X0).c8=x;
   (*X1).c8=-x;
   x=um[26]-um[22];
   (*X0).c8-=x;
   (*X1).c8-=x;

   x=up[27]+up[23];
   (*X0).c9=-x;
   (*X1).c9=x;
   x=um[27]+um[23];
   (*X0).c9+=x;
   (*X1).c9+=x;

   X0=X+2;
   X1=X+5;

   x=up[0]-up[3];
   (*X0).c1=x;
   (*X1).c1=-x;
   x=um[0]-um[3];
   (*X0).c1-=x;
   (*X1).c1-=x;

   x=up[1]-up[4];
   (*X0).c2=x;
   (*X1).c2=-x;
   x=um[1]-um[4];
   (*X0).c2-=x;
   (*X1).c2-=x;

   x=up[2]-up[5];
   (*X0).c3=x;
   (*X1).c3=-x;
   x=um[2]-um[5];
   (*X0).c3-=x;
   (*X1).c3-=x;

   x=up[31]-up[7];
   (*X0).c4=x;
   (*X1).c4=-x;
   x=um[31]-um[7];
   (*X0).c4-=x;
   (*X1).c4-=x;

   x=up[6]-up[30];
   (*X0).c5=x;
   (*X1).c5=-x;
   x=um[6]-um[30];
   (*X0).c5-=x;
   (*X1).c5-=x;

   x=up[33]-up[9];
   (*X0).c6=x;
   (*X1).c6=-x;
   x=um[33]-um[9];
   (*X0).c6-=x;
   (*X1).c6-=x;

   x=up[8]-up[32];
   (*X0).c7=x;
   (*X1).c7=-x;
   x=um[8]-um[32];
   (*X0).c7-=x;
   (*X1).c7-=x;

   x=up[35]-up[17];
   (*X0).c8=x;
   (*X1).c8=-x;
   x=um[35]-um[17];
   (*X0).c8-=x;
   (*X1).c8-=x;

   x=up[16]-up[34];
   (*X0).c9=x;
   (*X1).c9=-x;
   x=um[16]-um[34];
   (*X0).c9-=x;
   (*X1).c9-=x;
}


static void det2xt5(pauli_dble *m,u3_alg_dble *X)
{
   double x,*up,*um;
   u3_alg_dble *X0,*X1;

   up=m[0].u;
   um=m[1].u;

   X0=X;
   X1=X+3;

   x=up[10]+up[10];
   (*X0).c1=x;
   (*X1).c1=-x;
   x=um[10]+um[10];
   (*X0).c1+=x;
   (*X1).c1+=x;

   x=up[20]+up[20];
   (*X0).c2=x;
   (*X1).c2=-x;
   x=um[20]+um[20];
   (*X0).c2+=x;
   (*X1).c2+=x;

   x=up[28]+up[28];
   (*X0).c3=x;
   (*X1).c3=-x;
   x=um[28]+um[28];
   (*X0).c3+=x;
   (*X1).c3+=x;

   x=up[19]-up[13];
   (*X0).c4=x;
   (*X1).c4=-x;
   x=um[19]-um[13];
   (*X0).c4+=x;
   (*X1).c4+=x;

   x=up[12]+up[18];
   (*X0).c5=x;
   (*X1).c5=-x;
   x=um[12]+um[18];
   (*X0).c5+=x;
   (*X1).c5+=x;

   x=up[25]-up[15];
   (*X0).c6=x;
   (*X1).c6=-x;
   x=um[25]-um[15];
   (*X0).c6+=x;
   (*X1).c6+=x;

   x=up[14]+up[24];
   (*X0).c7=x;
   (*X1).c7=-x;
   x=um[14]+um[24];
   (*X0).c7+=x;
   (*X1).c7+=x;

   x=up[27]-up[23];
   (*X0).c8=x;
   (*X1).c8=-x;
   x=um[27]-um[23];
   (*X0).c8+=x;
   (*X1).c8+=x;

   x=up[22]+up[26];
   (*X0).c9=x;
   (*X1).c9=-x;
   x=um[22]+um[26];
   (*X0).c9+=x;
   (*X1).c9+=x;

   X0=X+1;
   X1=X+4;

   x=up[11]+up[11];
   (*X0).c1=-x;
   (*X1).c1=x;
   x=um[11]+um[11];
   (*X0).c1-=x;
   (*X1).c1-=x;

   x=up[21]+up[21];
   (*X0).c2=-x;
   (*X1).c2=x;
   x=um[21]+um[21];
   (*X0).c2-=x;
   (*X1).c2-=x;

   x=up[29]+up[29];
   (*X0).c3=-x;
   (*X1).c3=x;
   x=um[29]+um[29];
   (*X0).c3-=x;
   (*X1).c3-=x;

   x=up[18]-up[12];
   (*X0).c4=x;
   (*X1).c4=-x;
   x=um[18]-um[12];
   (*X0).c4+=x;
   (*X1).c4+=x;

   x=up[13]+up[19];
   (*X0).c5=-x;
   (*X1).c5=x;
   x=um[13]+um[19];
   (*X0).c5-=x;
   (*X1).c5-=x;

   x=up[24]-up[14];
   (*X0).c6=x;
   (*X1).c6=-x;
   x=um[24]-um[14];
   (*X0).c6+=x;
   (*X1).c6+=x;

   x=up[25]+up[15];
   (*X0).c7=-x;
   (*X1).c7=x;
   x=um[25]+um[15];
   (*X0).c7-=x;
   (*X1).c7-=x;

   x=up[26]-up[22];
   (*X0).c8=x;
   (*X1).c8=-x;
   x=um[26]-um[22];
   (*X0).c8+=x;
   (*X1).c8+=x;

   x=up[27]+up[23];
   (*X0).c9=-x;
   (*X1).c9=x;
   x=um[27]+um[23];
   (*X0).c9-=x;
   (*X1).c9-=x;

   X0=X+2;
   X1=X+5;

   x=up[0]-up[3];
   (*X0).c1=x;
   (*X1).c1=-x;
   x=um[0]-um[3];
   (*X0).c1+=x;
   (*X1).c1+=x;

   x=up[1]-up[4];
   (*X0).c2=x;
   (*X1).c2=-x;
   x=um[1]-um[4];
   (*X0).c2+=x;
   (*X1).c2+=x;

   x=up[2]-up[5];
   (*X0).c3=x;
   (*X1).c3=-x;
   x=um[2]-um[5];
   (*X0).c3+=x;
   (*X1).c3+=x;

   x=up[31]-up[7];
   (*X0).c4=x;
   (*X1).c4=-x;
   x=um[31]-um[7];
   (*X0).c4+=x;
   (*X1).c4+=x;

   x=up[6]-up[30];
   (*X0).c5=x;
   (*X1).c5=-x;
   x=um[6]-um[30];
   (*X0).c5+=x;
   (*X1).c5+=x;

   x=up[33]-up[9];
   (*X0).c6=x;
   (*X1).c6=-x;
   x=um[33]-um[9];
   (*X0).c6+=x;
   (*X1).c6+=x;

   x=up[8]-up[32];
   (*X0).c7=x;
   (*X1).c7=-x;
   x=um[8]-um[32];
   (*X0).c7+=x;
   (*X1).c7+=x;

   x=up[35]-up[17];
   (*X0).c8=x;
   (*X1).c8=-x;
   x=um[35]-um[17];
   (*X0).c8+=x;
   (*X1).c8+=x;

   x=up[16]-up[34];
   (*X0).c9=x;
   (*X1).c9=-x;
   x=um[16]-um[34];
   (*X0).c9+=x;
   (*X1).c9+=x;
}


void prod2xt(spinor_dble *r,spinor_dble *s,u3_alg_dble *X)
{
   pauli_dble m[2] ALIGNED16;
   spin_dble *spr,*sps;

   spr=(spin_dble*)(r);
   sps=(spin_dble*)(s);

   vec2pauli((*spr).w,(*sps).w,m);
   vec2pauli((*spr).w+1,(*sps).w+1,m+1);

   det2xt5(m,X);
}


static void set2mat(su3_vector_dble *psi,su3_dble *u)
{
   (*u).c11.re=_re(psi[0].c1,psi[2].c1)+_re(psi[1].c1,psi[3].c1);
   (*u).c11.im=_im(psi[0].c1,psi[2].c1)+_im(psi[1].c1,psi[3].c1);
   (*u).c12.re=_re(psi[0].c1,psi[2].c2)+_re(psi[1].c1,psi[3].c2);
   (*u).c12.im=_im(psi[0].c1,psi[2].c2)+_im(psi[1].c1,psi[3].c2);
   (*u).c13.re=_re(psi[0].c1,psi[2].c3)+_re(psi[1].c1,psi[3].c3);
   (*u).c13.im=_im(psi[0].c1,psi[2].c3)+_im(psi[1].c1,psi[3].c3);

   (*u).c21.re=_re(psi[0].c2,psi[2].c1)+_re(psi[1].c2,psi[3].c1);
   (*u).c21.im=_im(psi[0].c2,psi[2].c1)+_im(psi[1].c2,psi[3].c1);
   (*u).c22.re=_re(psi[0].c2,psi[2].c2)+_re(psi[1].c2,psi[3].c2);
   (*u).c22.im=_im(psi[0].c2,psi[2].c2)+_im(psi[1].c2,psi[3].c2);
   (*u).c23.re=_re(psi[0].c2,psi[2].c3)+_re(psi[1].c2,psi[3].c3);
   (*u).c23.im=_im(psi[0].c2,psi[2].c3)+_im(psi[1].c2,psi[3].c3);

   (*u).c31.re=_re(psi[0].c3,psi[2].c1)+_re(psi[1].c3,psi[3].c1);
   (*u).c31.im=_im(psi[0].c3,psi[2].c1)+_im(psi[1].c3,psi[3].c1);
   (*u).c32.re=_re(psi[0].c3,psi[2].c2)+_re(psi[1].c3,psi[3].c2);
   (*u).c32.im=_im(psi[0].c3,psi[2].c2)+_im(psi[1].c3,psi[3].c2);
   (*u).c33.re=_re(psi[0].c3,psi[2].c3)+_re(psi[1].c3,psi[3].c3);
   (*u).c33.im=_im(psi[0].c3,psi[2].c3)+_im(psi[1].c3,psi[3].c3);
}


static void add2mat(su3_vector_dble *psi,su3_dble *u)
{
   (*u).c11.re+=_re(psi[0].c1,psi[2].c1)+_re(psi[1].c1,psi[3].c1);
   (*u).c11.im+=_im(psi[0].c1,psi[2].c1)+_im(psi[1].c1,psi[3].c1);
   (*u).c12.re+=_re(psi[0].c1,psi[2].c2)+_re(psi[1].c1,psi[3].c2);
   (*u).c12.im+=_im(psi[0].c1,psi[2].c2)+_im(psi[1].c1,psi[3].c2);
   (*u).c13.re+=_re(psi[0].c1,psi[2].c3)+_re(psi[1].c1,psi[3].c3);
   (*u).c13.im+=_im(psi[0].c1,psi[2].c3)+_im(psi[1].c1,psi[3].c3);

   (*u).c21.re+=_re(psi[0].c2,psi[2].c1)+_re(psi[1].c2,psi[3].c1);
   (*u).c21.im+=_im(psi[0].c2,psi[2].c1)+_im(psi[1].c2,psi[3].c1);
   (*u).c22.re+=_re(psi[0].c2,psi[2].c2)+_re(psi[1].c2,psi[3].c2);
   (*u).c22.im+=_im(psi[0].c2,psi[2].c2)+_im(psi[1].c2,psi[3].c2);
   (*u).c23.re+=_re(psi[0].c2,psi[2].c3)+_re(psi[1].c2,psi[3].c3);
   (*u).c23.im+=_im(psi[0].c2,psi[2].c3)+_im(psi[1].c2,psi[3].c3);

   (*u).c31.re+=_re(psi[0].c3,psi[2].c1)+_re(psi[1].c3,psi[3].c1);
   (*u).c31.im+=_im(psi[0].c3,psi[2].c1)+_im(psi[1].c3,psi[3].c1);
   (*u).c32.re+=_re(psi[0].c3,psi[2].c2)+_re(psi[1].c3,psi[3].c2);
   (*u).c32.im+=_im(psi[0].c3,psi[2].c2)+_im(psi[1].c3,psi[3].c2);
   (*u).c33.re+=_re(psi[0].c3,psi[2].c3)+_re(psi[1].c3,psi[3].c3);
   (*u).c33.im+=_im(psi[0].c3,psi[2].c3)+_im(psi[1].c3,psi[3].c3);
}


static void prod2xv0(spinor_dble *rx,spinor_dble *ry,
                     spinor_dble *sx,spinor_dble *sy,su3_dble *u)
{
   su3_vector_dble psi[4];

   _vector_add(psi[0],(*ry).c1,(*ry).c3);
   _vector_add(psi[1],(*ry).c2,(*ry).c4);
   _vector_sub(psi[2],(*sx).c1,(*sx).c3);
   _vector_sub(psi[3],(*sx).c2,(*sx).c4);
   set2mat(psi,u);

   _vector_add(psi[0],(*sy).c1,(*sy).c3);
   _vector_add(psi[1],(*sy).c2,(*sy).c4);
   _vector_sub(psi[2],(*rx).c1,(*rx).c3);
   _vector_sub(psi[3],(*rx).c2,(*rx).c4);
   add2mat(psi,u);
}


static void prod2xv1(spinor_dble *rx,spinor_dble *ry,
                     spinor_dble *sx,spinor_dble *sy,su3_dble *u)
{
   su3_vector_dble psi[4];

   _vector_i_add(psi[0],(*ry).c1,(*ry).c4);
   _vector_i_add(psi[1],(*ry).c2,(*ry).c3);
   _vector_i_sub(psi[2],(*sx).c1,(*sx).c4);
   _vector_i_sub(psi[3],(*sx).c2,(*sx).c3);
   set2mat(psi,u);

   _vector_i_add(psi[0],(*sy).c1,(*sy).c4);
   _vector_i_add(psi[1],(*sy).c2,(*sy).c3);
   _vector_i_sub(psi[2],(*rx).c1,(*rx).c4);
   _vector_i_sub(psi[3],(*rx).c2,(*rx).c3);
   add2mat(psi,u);
}


static void prod2xv2(spinor_dble *rx,spinor_dble *ry,
                     spinor_dble *sx,spinor_dble *sy,su3_dble *u)
{
   su3_vector_dble psi[4];

   _vector_add(psi[0],(*ry).c1,(*ry).c4);
   _vector_sub(psi[1],(*ry).c2,(*ry).c3);
   _vector_sub(psi[2],(*sx).c1,(*sx).c4);
   _vector_add(psi[3],(*sx).c2,(*sx).c3);
   set2mat(psi,u);

   _vector_add(psi[0],(*sy).c1,(*sy).c4);
   _vector_sub(psi[1],(*sy).c2,(*sy).c3);
   _vector_sub(psi[2],(*rx).c1,(*rx).c4);
   _vector_add(psi[3],(*rx).c2,(*rx).c3);
   add2mat(psi,u);
}


static void prod2xv3(spinor_dble *rx,spinor_dble *ry,
                     spinor_dble *sx,spinor_dble *sy,su3_dble *u)
{
   su3_vector_dble psi[4];

   _vector_i_add(psi[0],(*ry).c1,(*ry).c3);
   _vector_i_sub(psi[1],(*ry).c2,(*ry).c4);
   _vector_i_sub(psi[2],(*sx).c1,(*sx).c3);
   _vector_i_add(psi[3],(*sx).c2,(*sx).c4);
   set2mat(psi,u);

   _vector_i_add(psi[0],(*sy).c1,(*sy).c3);
   _vector_i_sub(psi[1],(*sy).c2,(*sy).c4);
   _vector_i_sub(psi[2],(*rx).c1,(*rx).c3);
   _vector_i_add(psi[3],(*rx).c2,(*rx).c4);
   add2mat(psi,u);
}


void (*prod2xv[4])(spinor_dble *rx,spinor_dble *ry,
                   spinor_dble *sx,spinor_dble *sy,su3_dble *u)=
{prod2xv0,prod2xv1,prod2xv2,prod2xv3};
