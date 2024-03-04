
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2009-2011, 2013, 2016, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the gauge and translation invariance of tcharge().
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "flags.h"
#include "su3fcts.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "devfcts.h"
#include "tcharge.h"
#include "global.h"


int main(int argc,char *argv[])
{
   int my_rank,bc,i,s[4];
   double phi[2],phi_prime[2],theta[3];
   double d,dmax1,dmax2;
   double Q1,Q2,q1,q2;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);
      printf("\n");
      printf("Gauge and translation invariance of tcharge()\n");
      printf("---------------------------------------------\n\n");

      print_lattice_sizes();

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check1.c]",
                    "Syntax: check1 [-bc <type>]");
   }

   check_machine();
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;
   set_bc_parms(bc,1.0,1.0,1.0,1.0,phi,phi_prime,theta);
   print_bc_parms(0x0);

   start_ranlux(0,12345);
   geometry();

   dmax1=0.0;
   dmax2=0.0;

   for (i=0;i<8;i++)
   {
      random_ud();

      Q1=tcharge();
      random_shift(s);
      shift_ud(s);
      Q2=tcharge();

      d=fabs(Q1-Q2);
      if (d>dmax1)
         dmax1=d;

      random_gtrans();
      apply_gtrans2ud();
      Q2=tcharge();

      d=fabs(Q1-Q2);
      if (d>dmax2)
         dmax2=d;

      q1=Q1;
      q2=Q2;

      MPI_Bcast(&q1,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&q2,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((q1!=Q1)||(q2!=Q2),1,"main [check1.c]",
            "Charge is not globally the same");
   }

   print_flags();

   if (my_rank==0)
   {
      printf("Last charge = %.16e\n",Q1);
      printf("Translation invariance = %.2e\n",dmax1);
      printf("Gauge invariance = %.2e\n\n",dmax2);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
