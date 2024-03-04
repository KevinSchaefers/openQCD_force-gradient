
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2017, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the programs gather_msfld() and scatter_msfld().
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "random.h"
#include "lattice.h"
#include "msfcts.h"
#include "global.h"

static double zero[3]={0.0,0.0,0.0};
static double **f;
static complex_dble **rf;


static void alloc_msfld(int nf,int nrf)
{
   int i;
   double *f1,**f2;
   complex_dble *rf1,**rf2;

   f1=malloc(nf*VOLUME*sizeof(*f1));
   f2=malloc(nf*sizeof(*f2));

   rf1=malloc(nrf*VOLUME*sizeof(*rf1));
   rf2=malloc(nrf*sizeof(*rf2));

   error((f1==NULL)||(f2==NULL)||(rf1==NULL)||(rf2==NULL),1,
         "alloc_msfld [check1.c]","Unable to allocate fields");

   f=f2;
   rf=rf2;

   for (i=0;i<nf;i++)
   {
      f2[0]=f1;
      f1+=VOLUME;
      f2+=1;
   }

   for (i=0;i<nrf;i++)
   {
      rf2[0]=rf1;
      rf1+=VOLUME;
      rf2+=1;
   }
}


static void check_msfld(void)
{
   int ix;
   int ie1,ie2,ie3;

   ie1=0;
   ie2=0;
   ie3=0;

   for (ix=0;ix<VOLUME;ix++)
   {
      ie1|=(f[0][ix]!=f[1][ix]);
      ie2|=(f[0][ipt[ix]]!=rf[0][ix].re);
      ie3|=(rf[0][ix].im!=0.0);
   }

   error(ie1!=0,1,"check_msfld [check1.c]",
         "gather followed by scatter does not reproduce the input field");
   error(ie2!=0,1,"check_msfld [check1.c]",
         "gather does not properly reorder the field values");
   error(ie3!=0,1,"check_msfld [check1.c]",
         "gather does not zero the imaginary part of the target field");
}


int main(int argc,char *argv[])
{
   int my_rank;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);

      printf("\n");
      printf("Check of the programs gather_msfld() and scatter_msfld()\n");
      printf("--------------------------------------------------------\n\n");

      print_lattice_sizes();
   }

   set_bc_parms(3,1.0,1.0,1.0,1.0,zero,zero,zero);

   start_ranlux(0,12345);
   geometry();
   alloc_msfld(2,1);

   gauss_dble(f[0],VOLUME);
   gauss_dble(f[1],VOLUME);
   gauss_dble((double*)(rf[0]),2*VOLUME);

   gather_msfld(f[0],rf[0]);
   scatter_msfld(rf[0],f[1]);
   check_msfld();

   if (my_rank==0)
   {
      printf("No errors discovered\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
