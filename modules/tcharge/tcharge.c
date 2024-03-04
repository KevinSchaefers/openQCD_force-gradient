
/*******************************************************************************
*
* File tcharge.c
*
* Copyright (C) 2010-2013, 2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the topological charge using the symmetric field tensor.
*
*   double tcharge(void)
*     Returns the "field-theoretic" topological charge Q of the global
*     double-precision gauge field, using a symmetric expression for the
*     gauge-field tensor.
*
*   double tcharge_slices(double *qsl)
*     Computes the sum qsl[x0] of the "field-theoretic" topological charge
*     density of the double-precision gauge field at time x0=0,1,...,N0-1
*     (where N0=NPROC0*L0). The program returns the total charge.
*
*   void tcharge_fld(double *f)
*     Assigns the density field q to the observable field f such that
*     f[ix]=q(x), where 0<=ix<VOLUME is the index of the point x.
*
* The topological charge density q(x) is defined by
*
*  q(x)=(8*Pi^2)^(-1)*{F_{01}^a(x)*F_{23}^a(x)+
*                      F_{02}^a(x)*F_{31}^a(x)+
*                      F_{03}^a(x)*F_{12}^a(x)},
*
* where
*
*  F_{mu,nu}^a(x)=-2*tr{F_{mu,nu}(x)*T^a}, a=1,..,8,
*
* are the SU(3) components of the symmetric field tensor returned by the
* program ftensor() [ftensor.c].
*
* At the boundaries of the lattice (if any), the density q(x) is set to zero.
* The total charge Q is the sums of q(x) over all points x with time component
* in the range
*
*  0<x0<NPROC0*L0-1        (open bc),
*
*  0<x0<NPROC0*L0          (SF and open-SF bc),
*
*  0<=x0<NPROC0*L0         (periodic bc).
*
* The programs in this module act globally and must be called simultaneously
* by the OpenMP master thread on all MPI processes.
*
*******************************************************************************/

#define TCHARGE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "su3fcts.h"
#include "tcharge.h"
#include "global.h"

#define N0 (NPROC0*L0)

static double *qcsl[N0];
static qflt rqcsl[N0];
static u3_alg_dble **ft;


static double prodXY(u3_alg_dble *X,u3_alg_dble *Y)
{
   double sm;

   sm=(-2.0/3.0)*((*X).c1+(*X).c2+(*X).c3)*((*Y).c1+(*Y).c2+(*Y).c3)+
      2.0*((*X).c1*(*Y).c1+(*X).c2*(*Y).c2+(*X).c3*(*Y).c3)+
      4.0*((*X).c4*(*Y).c4+(*X).c5*(*Y).c5+(*X).c6*(*Y).c6+
           (*X).c7*(*Y).c7+(*X).c8*(*Y).c8+(*X).c9*(*Y).c9);

   return sm;
}


static double density(int ix)
{
   double sm;

   sm=prodXY(ft[0]+ix,ft[3]+ix)+
      prodXY(ft[1]+ix,ft[4]+ix)+
      prodXY(ft[2]+ix,ft[5]+ix);

   return sm;
}


double tcharge(void)
{
   int bc,tmx;
   int k,ix,t;
   double Q,pi,fact,*qsm[1];
   qflt rqsm;

   ft=ftensor();
   bc=bc_type();
   if (bc==0)
      tmx=N0-1;
   else
      tmx=N0;
   rqsm.q[0]=0.0;
   rqsm.q[1]=0.0;

#pragma omp parallel private(k,ix,t,Q) reduction(sum_qflt : rqsm)
   {
      k=omp_get_thread_num();

      for (ix=(k*VOLUME_TRD);ix<((k+1)*VOLUME_TRD);ix++)
      {
         t=global_time(ix);

         if (((t>0)&&(t<tmx))||(bc==3))
         {
            Q=density(ix);
            acc_qflt(Q,rqsm.q);
         }
      }
   }

   if (NPROC>1)
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   pi=4.0*atan(1.0);
   fact=1.0/(8.0*pi*pi);

   return fact*rqsm.q[0];
}


double tcharge_slices(double *qsl)
{
   int bc,tmx;
   int k,ix,t;
   double Q,pi,fact;

   ft=ftensor();
   bc=bc_type();
   if (bc==0)
      tmx=N0-1;
   else
      tmx=N0;

   for (t=0;t<N0;t++)
   {
      qcsl[t]=rqcsl[t].q;
      rqcsl[t].q[0]=0.0;
      rqcsl[t].q[1]=0.0;
   }

#pragma omp parallel private(k,ix,t,Q) reduction(sum_qflt : rqcsl[cpr[0]*L0:L0])
   {
      k=omp_get_thread_num();

      for (ix=(k*VOLUME_TRD);ix<((k+1)*VOLUME_TRD);ix++)
      {
         t=global_time(ix);

         if (((t>0)&&(t<tmx))||(bc==3))
         {
            Q=density(ix);
            acc_qflt(Q,rqcsl[t].q);
         }
      }
   }

   if (NPROC>1)
      global_qsum(N0,qcsl,qcsl);

   pi=4.0*atan(1.0);
   fact=1.0/(8.0*pi*pi);

   for (t=0;t<N0;t++)
   {
      qsl[t]=fact*rqcsl[t].q[0];

      if (t>0)
         add_qflt(qcsl[t],qcsl[0],qcsl[0]);
   }

   return fact*rqcsl[0].q[0];
}


void tcharge_fld(double *f)
{
   int bc,tmx;
   int k,ix,t;
   double pi,fact,*fr;

   ft=ftensor();
   bc=bc_type();
   if (bc==0)
      tmx=N0-1;
   else
      tmx=N0;

   pi=4.0*atan(1.0);
   fact=1.0/(8.0*pi*pi);

#pragma omp parallel private(k,ix,t,fr)
   {
      k=omp_get_thread_num();
      fr=f+k*VOLUME_TRD;

      for (ix=(k*VOLUME_TRD);ix<((k+1)*VOLUME_TRD);ix++)
      {
         t=global_time(ix);

         if (((t>0)&&(t<tmx))||(bc==3))
            fr[0]=fact*density(ix);
         else
            fr[0]=0.0;

         fr+=1;
      }
   }
}
