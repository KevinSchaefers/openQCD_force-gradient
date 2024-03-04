
/*******************************************************************************
*
* File time3.c
*
* Copyright (C) 2005-2011, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of inv_pauli_dble() and det_pauli_dble().
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "random.h"
#include "utils.h"
#include "random.h"
#include "linalg.h"
#include "sw_term.h"

#if (defined AVX)
static pauli_dble m[2] ALIGNED32;
#else
static pauli_dble m[2] ALIGNED16;
#endif

int main(void)
{
   int n,count,itest;
   double t1,t2,dt,mu;
   pauli_wsp_t *pwsp;

   printf("\n");
   printf("Timing of inv_pauli_dble() and det_pauli_dble()\n");
   printf("-----------------------------------------------\n\n");

#if (defined AVX)
   printf("Using AVX instructions\n\n");
#elif (defined x64)
   printf("Using SSE3 instructions and up to 16 xmm registers\n\n");
#endif

   printf("Measurement made with all data in cache\n\n");

   rlxd_init(1,1,23456,1);
   pwsp=alloc_pauli_wsp();

   ranlxd((*m).u,36);
   mu=0.1234;

   for (n=0;n<6;n++)
      (*m).u[n]=1.0;

   for (n=6;n<36;n++)
      (*m).u[n]=0.01*((*m).u[n]-0.5);

   n=(int)(1.0e5);
   dt=0.0;
   itest=0;

   while (dt<2.0)
   {
      t1=(double)clock();
      for (count=0;count<n;count++)
         itest=inv_pauli_dble(mu,m,pwsp,m+1);
      t2=(double)clock();
      dt=(t2-t1)/(double)(CLOCKS_PER_SEC);
      n*=2;
   }

   dt*=2.0e6/(double)(n);
   error(itest!=0,1,"main [time3.c]","Inversion was not safe");

   printf("Time per call of inv_pauli_dble():\n");
   printf("%.3f micro sec\n\n",dt);

   n=(int)(1.0e5);
   dt=0.0;
   itest=0;

   while (dt<2.0)
   {
      t1=(double)clock();
      for (count=0;count<n;count++)
         det_pauli_dble(mu,m,pwsp);
      t2=(double)clock();
      dt=(t2-t1)/(double)(CLOCKS_PER_SEC);
      n*=2;
   }

   dt*=2.0e6/(double)(n);

   printf("Time per call of det_pauli_dble():\n");
   printf("%.3f micro sec\n\n",dt);

   exit(0);
}
