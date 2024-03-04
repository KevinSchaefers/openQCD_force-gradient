
/*******************************************************************************
*
* File swflds.c
*
* Copyright (C) 2006, 2011, 2013, 2016, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Allocation and initialization of the global SW fields.
*
*   pauli *swfld(void)
*     Returns the base address of the single-precision SW field. If it
*     is not already allocated, the field is allocated and initialized
*     to unity.
*
*   pauli_dble *swdfld(void)
*     Returns the base address of the double-precision SW field. If it
*     is not already allocated, the field is allocated and initialized
*     to unity.
*
*   void assign_swd2sw(void)
*     Assigns the double-precision to the single-precision SW field.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define SWFLDS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "sw_term.h"
#include "global.h"

static pauli *swb=NULL;
static pauli_dble *swdb=NULL;


static void alloc_sw(void)
{
   int k,l;
   pauli *sw,*sm;

   error_root(sizeof(pauli)!=(36*sizeof(float)),1,"alloc_sw [swflds.c]",
              "The pauli structures are not properly packed");

   swb=amalloc(2*VOLUME*sizeof(*swb),ALIGN);
   error(swb==NULL,1,"alloc_sw [swflds.c]",
         "Unable to allocate the global single-precision SW field");

#pragma omp parallel private(k,l,sw,sm)
   {
      k=omp_get_thread_num();
      sw=swb+2*k*VOLUME_TRD;
      sm=sw+2*VOLUME_TRD;

      for (;sw<sm;sw++)
      {
         for (l=0;l<36;l++)
         {
            if (l<6)
               (*sw).u[l]=1.0f;
            else
               (*sw).u[l]=0.0f;
         }
      }
   }
}


pauli *swfld(void)
{
   if (swb==NULL)
      alloc_sw();

   return swb;
}


static void alloc_swd(void)
{
   int k,l;
   pauli_dble *sw,*sm;

   error_root(sizeof(pauli_dble)!=(36*sizeof(double)),1,"alloc_swd [swflds.c]",
              "The pauli_dble structures are not properly packed");

   swdb=amalloc(2*VOLUME*sizeof(*swdb),ALIGN);
   error(swdb==NULL,1,"alloc_swd [swflds.c]",
         "Unable to allocate the global double-precision SW field");

#pragma omp parallel private(k,l,sw,sm)
   {
      k=omp_get_thread_num();
      sw=swdb+2*k*VOLUME_TRD;
      sm=sw+2*VOLUME_TRD;

      for (;sw<sm;sw++)
      {
         for (l=0;l<36;l++)
         {
            if (l<6)
               (*sw).u[l]=1.0;
            else
               (*sw).u[l]=0.0;
         }
      }
   }
}


pauli_dble *swdfld(void)
{
   if (swdb==NULL)
      alloc_swd();

   return swdb;
}


void assign_swd2sw(void)
{
   int k;

   if (swb==NULL)
      alloc_sw();
   if (swdb==NULL)
      error_loc(1,1,"assign_swd2sw [swflds.c]",
                "Double-precision SW field is not allocated");

#pragma omp parallel private(k)
   {
      k=omp_get_thread_num();
      assign_pauli(2*VOLUME_TRD,swdb+2*k*VOLUME_TRD,swb+2*k*VOLUME_TRD);
   }

   set_flags(ASSIGNED_SWD2SW);
}
