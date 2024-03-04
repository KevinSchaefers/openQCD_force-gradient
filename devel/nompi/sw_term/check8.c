
/*******************************************************************************
*
* File check8.c
*
* Copyright (C) 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of sw_exp() and sw_dexp().
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "random.h"
#include "linalg.h"
#include "sw_term.h"

typedef union
{
   complex_dble c[36];
   double r[72];
} mat_t;

#if (defined AVX)
static pauli_dble Aw[4] ALIGNED32;
static mat_t mw[3] ALIGNED32;

static weyl_dble wsv[6] ALIGNED32;
static pauli_dble pde[1] ALIGNED32;
static pauli_dble pex[1] ALIGNED32;
static mat_t mcm[2] ALIGNED32;
static mat_t mde[4] ALIGNED32;

#else
static pauli_dble Aw[4] ALIGNED16;
static mat_t mw[3] ALIGNED16;

static weyl_dble wsv[6] ALIGNED16;
static pauli_dble pde[1] ALIGNED16;
static pauli_dble pex[1] ALIGNED16;
static mat_t mcm[2] ALIGNED16;
static mat_t mde[4] ALIGNED16;

#endif


static void random_diagonal_pauli(pauli_dble *A)
{
   int k;
   double tr,nrm,*u;

   u=(*A).u;

   for (k=6;k<36;k++)
      u[k]=0.0;

   gauss_dble(u,6);
   tr=tr0_pauli_mat(A)/6.0;

   for (k=0;k<6;k++)
      u[k]-=tr;

   nrm=tr1_pauli_mat(A,A)/2.0;
   nrm=1.0/sqrt(nrm);

   for (k=0;k<6;k++)
      u[k]*=nrm;
}


static void random_pauli(pauli_dble *A)
{
   int k;
   double tr,nrm,*u;

   u=(*A).u;
   gauss_dble(u,36);
   tr=tr0_pauli_mat(A)/6.0;

   for (k=0;k<6;k++)
      u[k]-=tr;

   nrm=tr1_pauli_mat(A,A)/2.0;
   nrm=1.0/sqrt(nrm);

   for (k=0;k<36;k++)
      u[k]*=nrm;
}


static void random_mat(mat_t *m)
{
   int k;
   double nrm,*r;

   r=(*m).r;
   gauss_dble(r,72);

   nrm=0.0;

   for (k=0;k<72;k++)
      nrm+=r[k]*r[k];

   nrm=1.0/sqrt(nrm/2.0);

   for (k=0;k<72;k++)
      r[k]*=nrm;
}


static void pauli2mat(pauli_dble *A,mat_t *m)
{
   int i,j,k;
   double *u;
   complex_dble *c;

   u=(*A).u;
   c=(*m).c;
   k=6;

   for (i=0;i<6;i++)
   {
      c[6*i+i].re=u[i];
      c[6*i+i].im=0.0;

      for (j=i+1;j<6;j++)
      {
         c[6*i+j].re=u[k];
         c[6*j+i].re=c[6*i+j].re;
         k+=1;
         c[6*i+j].im=u[k];
         c[6*j+i].im=-c[6*i+j].im;
         k+=1;
      }
   }
}


static void copy_pauli(pauli_dble *A,pauli_dble *B)
{
   int k;
   double *uA,*uB;

   uA=(*A).u;
   uB=(*B).u;

   for (k=0;k<36;k++)
      uB[k]=uA[k];
}


static void mulr_pauli(double r,pauli_dble *A,pauli_dble *B)
{
   int k;
   double *uA,*uB;

   uA=(*A).u;
   uB=(*B).u;

   for (k=0;k<36;k++)
      uB[k]=r*uA[k];
}


static void mulr_add_pauli(double r,pauli_dble *A,pauli_dble *B)
{
   int k;
   double *uA,*uB;

   uA=(*A).u;
   uB=(*B).u;

   for (k=0;k<36;k++)
      uA[k]+=r*uB[k];
}


static void prod_pauli(pauli_dble *A,pauli_dble *B,pauli_dble *C)
{
   pauli2weyl(B,wsv);
   prod_pauli_mat(A,wsv,wsv);
   weyl2pauli(wsv,C);
}


static void copy_mat(mat_t *a,mat_t *b)
{
   int k;
   double *ra,*rb;

   ra=(*a).r;
   rb=(*b).r;

   for (k=0;k<72;k++)
      rb[k]=ra[k];
}


static void mulr_add_mat(double r,mat_t *a,mat_t *b)
{
   int k;
   double *ra,*rb;

   ra=(*a).r;
   rb=(*b).r;

   for (k=0;k<72;k++)
      ra[k]+=r*rb[k];
}


static void commutator(mat_t *a,mat_t *b,mat_t *c)
{
   cmat_mul_dble(6,(*a).c,(*b).c,mcm[0].c);
   cmat_mul_dble(6,(*b).c,(*a).c,mcm[1].c);
   cmat_sub_dble(6,mcm[0].c,mcm[1].c,(*c).c);
}


static void exp_pauli(int N,pauli_dble *A,pauli_dble *B)
{
   int k;
   double c;

   for (k=0;k<36;k++)
   {
      if (k<6)
         pex[0].u[k]=1.0;
      else
         pex[0].u[k]=0.0;
   }

   copy_pauli(pex,B);
   c=1.0;

   for (k=1;k<=N;k++)
   {
      c/=(double)(k);
      prod_pauli(A,pex,pex);
      mulr_add_pauli(c,B,pex);
   }
}


static void dexp1_pauli(int N,pauli_dble *A,mat_t *dA,mat_t *B)
{
   int k;
   double c;

   pauli2mat(A,mde);
   copy_mat(dA,mde+1);
   copy_mat(dA,B);
   c=1.0;

   for (k=1;k<=N;k++)
   {
      c/=(double)(k+1);
      commutator(mde,mde+1,mde+2);
      copy_mat(mde+2,mde+1);
      mulr_add_mat(c,B,mde+1);
   }

   copy_mat(B,mde);
   exp_pauli(N,A,pde);
   pauli2mat(pde,mde+1);
   cmat_mul_dble(6,mde[0].c,mde[1].c,(*B).c);
}


static void dexp2_pauli(int N,pauli_dble *A,mat_t *dA,mat_t *B)
{
   int k,l;
   double q[36];
   swexp_wsp_t *swsp;

   swsp=alloc_swexp_wsp(N);
   sw_dexp(A,1.0,swsp,q);
   pauli2mat(A,mde);
   copy_mat(dA,mde+1);

   for (k=0;k<72;k++)
      (*B).r[k]=0.0;

   for (k=0;k<6;k++)
   {
      if (k>0)
      {
         cmat_mul_dble(6,mde[0].c,mde[1].c,mde[2].c);
         copy_mat(mde+2,mde+1);
      }
      else
         copy_mat(mde+1,mde+2);

      for (l=0;l<6;l++)
      {
         if (l>0)
         {
            cmat_mul_dble(6,mde[2].c,mde[0].c,mde[3].c);
            copy_mat(mde+3,mde+2);
         }

         mulr_add_mat(q[6*k+l],B,mde+2);
      }
   }

   free(swsp);
}


static double cmp_pauli(pauli_dble *A,pauli_dble *B)
{
   int k;
   double d,dmx;
   double *uA,*uB;

   dmx=0.0;
   uA=(*A).u;
   uB=(*B).u;

   for (k=0;k<36;k++)
   {
      d=fabs(uA[k]-uB[k]);
      if (d>dmx)
         dmx=d;
   }

   return dmx;
}


static double cmp_mat(mat_t *a,mat_t *b)
{
   int k;
   double d,dmx;
   double *ra,*rb;

   dmx=0.0;
   ra=(*a).r;
   rb=(*b).r;

   for (k=0;k<72;k++)
   {
      d=fabs(ra[k]-rb[k]);
      if (d>dmx)
         dmx=d;
   }

   return dmx;
}


int main(void)
{
   int k,j,ifail;
   double r,d,dmx;
   pauli_wsp_t *pwsp;
   swexp_wsp_t *swsp;

   printf("\n");
   printf("Check of sw_exp() and sw_dexp()\n");
   printf("-------------------------------\n\n");

   rlxd_init(1,1,1234,1);
   pwsp=alloc_pauli_wsp();
   swsp=alloc_swexp_wsp(18);

   dmx=0.0;

   for (k=0;k<10;k++)
   {
      r=0.91287;
      random_diagonal_pauli(Aw);
      sw_exp(0,Aw,r,swsp,Aw+1);

      for (j=0;j<6;j++)
         (*Aw).u[j]=r*exp((*Aw).u[j]);

      d=cmp_pauli(Aw,Aw+1);
      if (d>dmx)
         dmx=d;
   }

   printf("Check exp(A) in the case of a diagonal matrix:\n");
   printf("Maximal absolute deviation = %.1e\n\n",dmx);

   dmx=0.0;

   for (k=0;k<10;k++)
   {
      r=0.91287;
      random_pauli(Aw);
      copy_pauli(Aw,Aw+2);
      sw_exp(0,Aw,r,swsp,Aw+1);

      d=cmp_pauli(Aw,Aw+2);
      if (d>dmx)
         dmx=d;

      pauli2weyl(Aw,wsv);
      weyl2pauli(wsv,Aw+1);

      d=cmp_pauli(Aw,Aw+1);
      if (d>dmx)
         dmx=d;
   }

   printf("Check that exp(A) leaves A unchanged:\n");
   printf("Maximal absolute deviation = %.1e (should be 0.0)\n\n",dmx);

   dmx=0.0;

   for (k=0;k<10;k++)
   {
      random_pauli(Aw);
      r=0.123+0.01*(double)(k);
      mulr_pauli(r,Aw,Aw+1);
      mulr_pauli(1.0-r,Aw,Aw+2);

      r=1.1097;
      sw_exp(0,Aw,r*r,swsp,Aw);
      sw_exp(0,Aw+1,r,swsp,Aw+1);
      sw_exp(0,Aw+2,r,swsp,Aw+2);
      prod_pauli(Aw+1,Aw+2,Aw+3);

      d=cmp_pauli(Aw,Aw+3);
      if (d>dmx)
         dmx=d;
   }

   printf("Comparison of exp(r*A)*exp((1-r)*A) with exp(A):\n");
   printf("Maximal absolute deviation = %.1e\n\n",dmx);

   dmx=0.0;

   for (k=0;k<10;k++)
   {
      r=0.0125674;
      random_pauli(Aw);
      sw_exp(0,Aw,1.0,swsp,Aw+1);
      sw_exp(0,Aw,r,swsp,Aw);
      mulr_pauli(1.0/r,Aw,Aw);

      d=cmp_pauli(Aw+1,Aw);
      if (d>dmx)
         dmx=d;
   }

   printf("Check of the scaling of exp(A) by a constant:\n");
   printf("Maximal absolute deviation = %.1e\n\n",dmx);

   dmx=0.0;

   for (k=0;k<10;k++)
   {
      r=0.0125674;
      random_pauli(Aw);
      sw_exp(0,Aw,r,swsp,Aw+1);
      sw_exp(1,Aw,1.0/r,swsp,Aw+2);
      prod_pauli(Aw+1,Aw+2,Aw+3);

      for (j=0;j<36;j++)
      {
         if (j<6)
            (*Aw).u[j]=1.0;
         else
            (*Aw).u[j]=0.0;
      }

      d=cmp_pauli(Aw,Aw+3);
      if (d>dmx)
         dmx=d;
   }

   printf("Check that (r*exp(A))*((1/r)*exp(-A))=1:\n");
   printf("Maximal absolute deviation = %.1e\n\n",dmx);

   dmx=0.0;
   ifail=0;

   for (k=0;k<10;k++)
   {
      random_pauli(Aw);
      sw_exp(0,Aw,1.0,swsp,Aw+1);
      sw_exp(1,Aw,1.0,swsp,Aw+2);
      ifail|=inv_pauli_dble(0.0,Aw+1,pwsp,Aw+3);

      if (ifail==0)
      {
         d=cmp_pauli(Aw+2,Aw+3);
         if (d>dmx)
            dmx=d;
      }
      else
         break;
   }

   printf("Comparison with inv_pauli_dble():\n");
   printf("Maximal absolute deviation = %.1e (ifail=%d)\n\n",dmx,ifail);

   dmx=0.0;

   for (k=0;k<10;k++)
   {
      random_pauli(Aw);
      sw_exp(0,Aw,1.0,swsp,Aw+1);
      exp_pauli(18,Aw,Aw+2);

      d=cmp_pauli(Aw+1,Aw+2);
      if (d>dmx)
         dmx=d;
   }

   printf("Comparision of exp(A) with power series:\n");
   printf("Maximal absolute deviation = %.1e\n\n",dmx);

   dmx=0.0;

   for (k=0;k<10;k++)
   {
      random_pauli(Aw);
      random_mat(mw);

      dexp1_pauli(20,Aw,mw,mw+1);
      dexp2_pauli(18,Aw,mw,mw+2);

      d=cmp_mat(mw+1,mw+2);
      if (d>dmx)
         dmx=d;
   }

   printf("Check of sw_dexp():\n");
   printf("Maximal absolute deviation = %.1e\n\n",dmx);

   exit(0);
}
