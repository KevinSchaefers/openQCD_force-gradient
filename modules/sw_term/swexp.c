
/*******************************************************************************
*
* File swexp.c
*
* Copyright (C) 2018, 2021 Antonio Rago, Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Exponential of traceless 6x6 Hermitian matrices and related functions.
*
*   swexp_wsp_t *alloc_swexp_wsp(int N)
*     Returns the address of a new instance of an swexp_wsp_t workspace.
*     The parameter N specifies the order of the Taylor expansion of the
*     exponential function to be used in the programs sw_exp() and sw_dexp().
*     A NULL pointer is returned if N<1 or if the allocation fails.
*
*   void free_swexp_wsp(swexp_wsp_t *swsp)
*     Frees the swexp_wsp_t workspace swsp previously allocated by
*     alloc_swexp_wsp().
*
*   void sw_exp(int s,pauli_dble *A,double r,swexp_wsp_t *swsp,
*               pauli_dble *B)
*     Assigns r*exp(A) (if s=0) or r*exp(-A) (if s!=0) to B. The program
*     assumes A is traceless and calculates the exponential approximately
*     by evaluating the Taylor expansion of the exponential up to order
*     (*swsp).N. It is permissible to set B=A.
*
*   void sw_dexp(pauli_dble *A,double r,swexp_wsp_t *swsp,double *q)
*     Computes the coeffients q[6*k+l], k,l=0,..,5, required to calculate
*     the derivatives of r*exp(A) with respect to A (see the notes). The
*     program assumes A is traceless and approximates the exponential by
*     evaluating its Taylor expansion up to order (*swsp).N.
*
* The derivative of exp(A) with respect to a parameter t of A is given by
*
*  sum_{k,l} Q_{kl}*A^k*(dA/dt)*A^l,  Q_{kl}=q[6*k+l],
*
* where q[0],..,q[35] are the coefficients calculated by sw_dexp() (with
* r=1.0) and the indices k,l run from 0 to 5. The matrix Q_{kl} is exactly
* symmetric.
*
* The programs in this module are thread-safe. If SSE (AVX) instructions are
* used, the Pauli matrices must be aligned to a 16 (32) byte boundary.
*
*******************************************************************************/

#define SWEXP_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "sw_term.h"


swexp_wsp_t *alloc_swexp_wsp(int N)
{
   int k,l;
   double **csv,**ksv,**qsv,*p;
   weyl_dble *wsv;
   pauli_dble *Asv;
   swexp_wsp_t *swsp;

   if (N>=1)
   {
      swsp=malloc(sizeof(*swsp));
      csv=malloc((N+9)*sizeof(*ksv));
      p=malloc(((N*(N+19))/2-1)*sizeof(*p));
      wsv=amalloc(6*sizeof(*wsv),5);
      Asv=amalloc(3*sizeof(*Asv),5);

      if ((swsp==NULL)||(csv==NULL)||(p==NULL)||(wsv==NULL)||(Asv==NULL))
         return NULL;

      ksv=csv+2;
      qsv=ksv+N+1;

      p[0]=1.0;

      for (k=1;k<=(N+1);k++)
         p[k]=p[k-1]/(double)(k);

      ksv[0]=p+2*(N+1);

      for (k=0;k<=N;k++)
      {
         if (k>0)
            ksv[k]=ksv[k-1]+N-k+2;

         for (l=0;l<=(N-k);l++)
            ksv[k][l]=p[k+l+1];
      }

      csv[0]=p;
      csv[1]=p+N+1;

      for (k=0;k<=N;k++)
      {
         if (k&0x1)
            csv[1][k]=-csv[0][k];
         else
            csv[1][k]=csv[0][k];
      }

      p+=((N+1)*(N+6))/2;

      for (k=0;k<6;k++)
      {
         qsv[k]=p;
         p+=(N-k+1);
      }

      (*swsp).N=N;
      (*swsp).csv=csv;
      (*swsp).ksv=ksv;
      (*swsp).qsv=qsv;
      (*swsp).psv=p;
      (*swsp).wsv=wsv;
      (*swsp).Asv=Asv;

      return swsp;
   }
   else
      return NULL;
}


void free_swexp_wsp(swexp_wsp_t *swsp)
{
   afree((*swsp).Asv);
   afree((*swsp).wsv);
   free((*swsp).csv[0]);
   free((*swsp).csv);
   free(swsp);
}


static void sw_cpoly(pauli_dble *A,swexp_wsp_t *swsp)
{
   double tr[5];
   double *psv;
   weyl_dble *wsv;
   pauli_dble *Asv;

   psv=(*swsp).psv;
   wsv=(*swsp).wsv;
   Asv=(*swsp).Asv;

   pauli2weyl(A,wsv);
   prod_pauli_mat(A,wsv,wsv);
   weyl2pauli(wsv,Asv);
   prod_pauli_mat(A,wsv,wsv);
   weyl2pauli(wsv,Asv+1);

   tr[0]=tr0_pauli_mat(Asv);
   tr[1]=tr0_pauli_mat(Asv+1);
   tr[2]=tr1_pauli_mat(Asv,Asv);
   tr[3]=tr1_pauli_mat(Asv,Asv+1);
   tr[4]=tr1_pauli_mat(Asv+1,Asv+1);

   psv[0]=(1.0/144.0)*(8.0*tr[1]*tr[1]-24.0*tr[4]
                       +tr[0]*(18.0*tr[2]-3.0*tr[0]*tr[0]));
   psv[1]=(1.0/30.0)*(5.0*tr[0]*tr[1]-6.0*tr[3]);
   psv[2]=(1.0/8.0)*(tr[0]*tr[0]-2.0*tr[2]);
   psv[3]=(-1.0/3.0)*tr[1];
   psv[4]=-0.5*tr[0];
}


static void sw_cayley(int n,double *c,swexp_wsp_t *swsp,double *q)
{
   int k;
   double q5,*cm,*psv;

   psv=(*swsp).psv;

   if (n>5)
   {
      cm=c;
      c=c+n-6;

      q[0]=c[1];
      q[1]=c[2];
      q[2]=c[3];
      q[3]=c[4];
      q[4]=c[5];
      q[5]=c[6];

      for (;c>=cm;c--)
      {
         q5=q[5];
         q[5]=q[4];
         q[4]=q[3]-q5*psv[4];
         q[3]=q[2]-q5*psv[3];
         q[2]=q[1]-q5*psv[2];
         q[1]=q[0]-q5*psv[1];
         q[0]=c[0]-q5*psv[0];
      }
   }
   else
   {
      for (k=0;k<=5;k++)
      {
         if (k<=n)
            q[k]=c[k];
         else
            q[k]=0.0;
      }
   }
}


static void sw_multi_cayley(int n,swexp_wsp_t *swsp,double *q)
{
   int k,l,m;
   double **ksv,**qsv;

   ksv=(*swsp).ksv;
   qsv=(*swsp).qsv;

   for (k=0;k<=n;k++)
   {
      m=n-k;
      sw_cayley(m,ksv[k],swsp,q);

      if (m>5)
         m=5;

      for (l=0;l<=m;l++)
         qsv[l][k]=q[l];
   }

   for (l=0;l<6;l++)
   {
      sw_cayley(n-l,qsv[l],swsp,q);
      q+=6;
   }
}


void sw_exp(int s,pauli_dble *A,double r,swexp_wsp_t *swsp,pauli_dble *B)
{
   int N,k;
   double q[6],**csv;
   weyl_dble *wsv;
   pauli_dble *Asv;

   N=(*swsp).N;
   csv=(*swsp).csv;
   wsv=(*swsp).wsv;
   Asv=(*swsp).Asv;

   s=(s!=0);
   sw_cpoly(A,swsp);
   sw_cayley(N,csv[s],swsp,q);

   for (k=0;k<6;k++)
      q[k]*=r;

   lc3_pauli_mat(q+3,A,Asv,Asv+2);
   prod_pauli_mat(Asv+2,wsv,wsv);
   weyl2pauli(wsv,Asv+1);

   lc3_pauli_mat(q,A,Asv,Asv+2);
   add_pauli_mat(Asv+1,Asv+2,B);
}


void sw_dexp(pauli_dble *A,double r,swexp_wsp_t *swsp,double *q)
{
   int N,k,l;

   N=(*swsp).N;
   sw_cpoly(A,swsp);
   sw_multi_cayley(N,swsp,q);

   for (k=0;k<6;k++)
   {
      q[6*k+k]*=r;

      for (l=0;l<k;l++)
      {
         q[6*k+l]*=r;
         q[6*l+k]=q[6*k+l];
      }
   }
}
