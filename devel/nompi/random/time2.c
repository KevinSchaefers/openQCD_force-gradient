
/*******************************************************************************
*
* File time2.c
*
* Copyright (C) 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Measurement of the processor time required to produce Gaussian random
* numbers using gauss() and gauss_dble().
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#if (defined _OPENMP)
#include <omp.h>
#endif
#include "random.h"

#define N 100
#define NLOOPS 100000
#define NTHREAD 2


int main(void)
{
   int k,level;
   float t1,t2,dt;
#if (defined _OPENMP)
   int l;
   float rs[(N+16)*NTHREAD];
   double rd[(N+8)*NTHREAD];
#else
   float rs[N];
   double rd[N];
#endif

   printf("\n");
   printf("Timing of gauss() and gauss_dble()\n");
   printf("----------------------------------\n\n");

#if (defined AVX)
   printf("Using AVX2 inline assembly.\n\n");
#elif (defined SSE)
   printf("Using SSE3 inline assembly.\n\n");
#endif

   printf("Using %d OpenMP threads.\n\n",NTHREAD);
   printf("Time per random number in nanoseconds:\n\n");
   printf("  Program           level 0      level 1      level 2\n");
   printf("-----------------------------------------------------\n");
   printf("  gauss     ");

#if (defined _OPENMP)
   omp_set_num_threads(NTHREAD);
#endif

   for (level=0;level<=2;level++)
   {
      rlxs_init(NTHREAD,level,1,1);

#if (defined _OPENMP)
      t1=(float)(omp_get_wtime());

#pragma omp parallel private(k,l)
      {
         l=omp_get_thread_num();
         for (k=1;k<=NLOOPS;k++)
            gauss(rs+l*(N+16),N);
      }

      t2=(float)(omp_get_wtime());
      dt=1.0e9f*(t2-t1)/((float)(N)*(float)(NLOOPS));

      printf("       %6.2f",dt);
#else
      t1=(float)(clock());
      for (k=1;k<=NLOOPS;k++)
         gauss(rs,N);
      t2=(float)(clock());

      dt=(t2-t1)/(float)(CLOCKS_PER_SEC);
      dt*=1.0e9f/((float)(N)*(float)(NLOOPS));

      printf("       %6.2f",dt);
#endif
   }

   printf("\n");
   printf("  gauss_dble          -  ");

   for (level=1;level<=2;level++)
   {
      rlxd_init(NTHREAD,level,1,1);

#if (defined _OPENMP)
      t1=(float)(omp_get_wtime());

#pragma omp parallel private(k,l)
      {
         l=omp_get_thread_num();
         for (k=1;k<=NLOOPS;k++)
            gauss_dble(rd+l*(N+8),N);
      }

      t2=(float)(omp_get_wtime());
      dt=1.0e9f*(t2-t1)/((float)(N)*(float)(NLOOPS));

      printf("       %6.2f",dt);
#else
      t1=(float)(clock());
      for (k=1;k<=NLOOPS;k++)
         gauss_dble(rd,N);
      t2=(float)(clock());

      dt=(t2-t1)/(float)(CLOCKS_PER_SEC);
      dt*=1.0e9f/((float)(N)*(float)(NLOOPS));

      printf("       %6.2f",dt);
#endif
   }

   printf("\n");
   printf("---------------------------------------------------\n\n");
   printf("The RANLUX p-values used at level 0, 1 and 2 are \n");
   printf("218, 404 and 794, respectively.\n\n");
   exit(0);
}
