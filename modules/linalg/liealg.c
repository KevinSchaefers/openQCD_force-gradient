
/*******************************************************************************
*
* File liealg.c
*
* Copyright (C) 2005-2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Basic functions for fields with values in the Lie algebra of SU(3).
*
*   void random_alg(int vol,int icom,su3_alg_dble *X)
*     Initializes the Lie algebra elements X to random values
*     with distribution proportional to exp{tr[X^2]}.
*
*   qflt norm_square_alg(int vol,int icom,su3_alg_dble *X)
*     Returns the square of the square-norm of the field X.
*
*   qflt scalar_prod_alg(int vol,int icom,su3_alg_dble *X,su3_alg_dble *Y)
*     Returns scalar product of the fields X and Y.
*
*   double unorm_alg(int vol,int icom,su3_alg_dble *X)
*     Returns the uniform norm of the field X.
*
*   void set_alg2zero(int vol,int icom,su3_alg_dble *X)
*     Sets the array elements X to zero.
*
*   void set_ualg2zero(int vol,int icom,u3_alg_dble *X)
*     Sets the array elements X to zero.
*
*   void assign_alg2alg(int vol,int icom,su3_alg_dble *X,su3_alg_dble *Y)
*     Assigns the field X to the field Y.
*
*   void flip_assign_alg2alg(int vol,int icom,su3_alg_dble *X,su3_alg_dble *Y)
*     Assigns the field -X to the field Y.
*
*   void muladd_assign_alg(int vol,int icom,double r,su3_alg_dble *X,
*                          su3_alg_dble *Y)
*     Adds r*X to Y.
*
* The quadruple-precision type qflt is defined in su3.h. See doc/qsum.pdf for
* further explanations.
*
* Lie algebra elements X are traceless antihermitian 3x3 matrices that
* are represented by structures with real elements x1,...,x8 through
*
*  X_11=i*(x1+x2), X_22=i*(x2-2*x1), X_33=i*(x1-2*x2),
*
*  X_12=x3+i*x4, X_13=x5+i*x6, X_23=x7+i*x8
*
* The scalar product (X,Y) of any two elements of the Lie algebra is
*
*  (X,Y)=-2*tr{XY}
*
* and the norm of X is (X,X)^(1/2). The uniform norm of a field of matrices
* is the maximum of the norm of its elements.
*
* All programs in this module operate on arrays of Lie algebra elements whose
* base address is passed through the arguments. The length of the array is
* specified by the parameter vol and the meaning of the parameter icom is
* explained in the file sflds/README.icom.
*
*******************************************************************************/

#define LIEALG_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "random.h"
#include "linalg.h"
#include "global.h"


static void loc_random_alg(int vol,su3_alg_dble *X)
{
   double c1,c2,c3,rb[8];
   su3_alg_dble *Xm;

   c1=(sqrt(3.0)+1.0)/6.0;
   c2=(sqrt(3.0)-1.0)/6.0;
   c3=0.5*sqrt(2.0);

   Xm=X+vol;

   for (;X<Xm;X++)
   {
      gauss_dble(rb,8);

      (*X).c1=c1*rb[0]+c2*rb[1];
      (*X).c2=c1*rb[1]+c2*rb[0];
      (*X).c3=c3*rb[2];
      (*X).c4=c3*rb[3];
      (*X).c5=c3*rb[4];
      (*X).c6=c3*rb[5];
      (*X).c7=c3*rb[6];
      (*X).c8=c3*rb[7];
   }
}


static qflt loc_norm_square_alg(int vol,su3_alg_dble *X)
{
   double sm,*qsm[1];
   qflt rqsm;
   su3_alg_dble *Xm;

   qsm[0]=rqsm.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   Xm=X+vol;

   for (;X<Xm;X++)
   {
      sm=3.0*((*X).c1*(*X).c1+(*X).c2*(*X).c2-(*X).c1*(*X).c2)+
         (*X).c3*(*X).c3+(*X).c4*(*X).c4+(*X).c5*(*X).c5+
         (*X).c6*(*X).c6+(*X).c7*(*X).c7+(*X).c8*(*X).c8;

      acc_qflt(sm,qsm[0]);
   }

   return rqsm;
}


static qflt loc_scalar_prod_alg(int vol,su3_alg_dble *X,su3_alg_dble *Y)
{
   double sm,*qsm[1];
   qflt rqsm;
   su3_alg_dble *Xm;

   qsm[0]=rqsm.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   Xm=X+vol;

   for (;X<Xm;X++)
   {
      sm=12.0*((*X).c1*(*Y).c1+(*X).c2*(*Y).c2)
         -6.0*((*X).c1*(*Y).c2+(*X).c2*(*Y).c1)
         +4.0*((*X).c3*(*Y).c3+(*X).c4*(*Y).c4+(*X).c5*(*Y).c5+
               (*X).c6*(*Y).c6+(*X).c7*(*Y).c7+(*X).c8*(*Y).c8);

      Y+=1;
      acc_qflt(sm,qsm[0]);
   }

   return rqsm;
}


static double loc_unorm_alg(int vol,su3_alg_dble *X)
{
   double mxn,nrm;
   su3_alg_dble *Xm;

   mxn=0.0;
   Xm=X+vol;

   for (;X<Xm;X++)
   {
      nrm=3.0*((*X).c1*(*X).c1+(*X).c2*(*X).c2-(*X).c1*(*X).c2)+
         (*X).c3*(*X).c3+(*X).c4*(*X).c4+(*X).c5*(*X).c5+
         (*X).c6*(*X).c6+(*X).c7*(*X).c7+(*X).c8*(*X).c8;

      if (nrm>mxn)
         mxn=nrm;
   }

   return mxn;
}


static void loc_set_alg2zero(int vol,su3_alg_dble *X)
{
   su3_alg_dble *Xm;

   Xm=X+vol;

   for (;X<Xm;X++)
   {
      (*X).c1=0.0;
      (*X).c2=0.0;
      (*X).c3=0.0;
      (*X).c4=0.0;
      (*X).c5=0.0;
      (*X).c6=0.0;
      (*X).c7=0.0;
      (*X).c8=0.0;
   }
}


static void loc_set_ualg2zero(int vol,u3_alg_dble *X)
{
   u3_alg_dble *Xm;

   Xm=X+vol;

   for (;X<Xm;X++)
   {
      (*X).c1=0.0;
      (*X).c2=0.0;
      (*X).c3=0.0;
      (*X).c4=0.0;
      (*X).c5=0.0;
      (*X).c6=0.0;
      (*X).c7=0.0;
      (*X).c8=0.0;
      (*X).c9=0.0;
   }
}


static void loc_assign_alg2alg(int vol,su3_alg_dble *X,su3_alg_dble *Y)
{
   su3_alg_dble *Xm;

   Xm=X+vol;

   for (;X<Xm;X++)
   {
      (*Y).c1=(*X).c1;
      (*Y).c2=(*X).c2;
      (*Y).c3=(*X).c3;
      (*Y).c4=(*X).c4;
      (*Y).c5=(*X).c5;
      (*Y).c6=(*X).c6;
      (*Y).c7=(*X).c7;
      (*Y).c8=(*X).c8;

      Y+=1;
   }
}


static void loc_flip_assign_alg2alg(int vol,su3_alg_dble *X,su3_alg_dble *Y)
{
   su3_alg_dble *Xm;

   Xm=X+vol;

   for (;X<Xm;X++)
   {
      (*Y).c1=-(*X).c1;
      (*Y).c2=-(*X).c2;
      (*Y).c3=-(*X).c3;
      (*Y).c4=-(*X).c4;
      (*Y).c5=-(*X).c5;
      (*Y).c6=-(*X).c6;
      (*Y).c7=-(*X).c7;
      (*Y).c8=-(*X).c8;

      Y+=1;
   }
}


static void loc_muladd_assign_alg(int vol,double r,su3_alg_dble *X,
                                  su3_alg_dble *Y)
{
   su3_alg_dble *Xm;

   Xm=X+vol;

   for (;X<Xm;X++)
   {
      (*Y).c1+=r*(*X).c1;
      (*Y).c2+=r*(*X).c2;
      (*Y).c3+=r*(*X).c3;
      (*Y).c4+=r*(*X).c4;
      (*Y).c5+=r*(*X).c5;
      (*Y).c6+=r*(*X).c6;
      (*Y).c7+=r*(*X).c7;
      (*Y).c8+=r*(*X).c8;

      Y+=1;
   }
}


void random_alg(int vol,int icom,su3_alg_dble *X)
{
   int k;

   if (icom&0x2)
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_random_alg(vol,X+k*vol);
      }
   }
   else
      loc_random_alg(vol,X);
}


qflt norm_square_alg(int vol,int icom,su3_alg_dble *X)
{
   int k;
   double *qsm[1];
   qflt rqsm;

   if ((icom&0x2)==0)
      rqsm=loc_norm_square_alg(vol,X);
   else
   {
      rqsm.q[0]=0.0;
      rqsm.q[1]=0.0;

#pragma omp parallel private(k) reduction(sum_qflt : rqsm)
      {
         k=omp_get_thread_num();
         rqsm=loc_norm_square_alg(vol,X+k*vol);
      }
   }

   if ((NPROC>1)&&(icom&0x1))
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   rqsm.q[0]*=4.0;
   rqsm.q[1]*=4.0;

   return rqsm;
}


qflt scalar_prod_alg(int vol,int icom,su3_alg_dble *X,su3_alg_dble *Y)
{
   int k;
   double *qsm[1];
   qflt rqsm;

   if ((icom&0x2)==0)
      rqsm=loc_scalar_prod_alg(vol,X,Y);
   else
   {
      rqsm.q[0]=0.0;
      rqsm.q[1]=0.0;

#pragma omp parallel private(k) reduction(sum_qflt : rqsm)
      {
         k=omp_get_thread_num();
         rqsm=loc_scalar_prod_alg(vol,X+k*vol,Y+k*vol);
      }
   }

   if ((NPROC>1)&&(icom&0x1))
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   return rqsm;
}


double unorm_alg(int vol,int icom,su3_alg_dble *X)
{
   int k;
   double mxn,nrm;

   if ((icom&0x2)==0)
      mxn=loc_unorm_alg(vol,X);
   else
   {
      mxn=0.0;

#pragma omp parallel private(k) reduction(max : mxn)
      {
         k=omp_get_thread_num();
         mxn=loc_unorm_alg(vol,X+k*vol);
      }
   }

   mxn=2.0*sqrt(mxn);

   if ((NPROC>1)&&(icom&0x1))
   {
      nrm=mxn;
      MPI_Allreduce(&nrm,&mxn,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
   }

   return mxn;
}


void set_alg2zero(int vol,int icom,su3_alg_dble *X)
{
   int k;

   if ((icom&0x2)==0)
      loc_set_alg2zero(vol,X);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_set_alg2zero(vol,X+k*vol);
      }
   }
}


void set_ualg2zero(int vol,int icom,u3_alg_dble *X)
{
   int k;

   if ((icom&0x2)==0)
      loc_set_ualg2zero(vol,X);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_set_ualg2zero(vol,X+k*vol);
      }
   }
}


void assign_alg2alg(int vol,int icom,su3_alg_dble *X,su3_alg_dble *Y)
{
   int k;

   if ((icom&0x2)==0)
      loc_assign_alg2alg(vol,X,Y);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_assign_alg2alg(vol,X+k*vol,Y+k*vol);
      }
   }
}


void flip_assign_alg2alg(int vol,int icom,su3_alg_dble *X,su3_alg_dble *Y)
{
   int k;

   if ((icom&0x2)==0)
      loc_flip_assign_alg2alg(vol,X,Y);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_flip_assign_alg2alg(vol,X+k*vol,Y+k*vol);
      }
   }
}


void muladd_assign_alg(int vol,int icom,double r,su3_alg_dble *X,
                       su3_alg_dble *Y)
{
   int k;

   if ((icom&0x2)==0)
      loc_muladd_assign_alg(vol,r,X,Y);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_muladd_assign_alg(vol,r,X+k*vol,Y+k*vol);
      }
   }
}
