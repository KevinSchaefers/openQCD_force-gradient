
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2013, 2016 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of import/export functions for the ranlux generators
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "utils.h"
#include "global.h"

#define NRAN 10000

static float r[2*NRAN*NTHREAD];
static double rd[2*NRAN*NTHREAD];


int main(int argc,char *argv[])
{
   int my_rank,tag,k,ie,ied;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);

      printf("\n");
      printf("Check of import/export functions for the ranlux generators\n");
      printf("----------------------------------------------------------\n\n");

      print_lattice_sizes();
   }

   check_machine();
   start_ranlux(0,1234);

# pragma omp parallel private(k)
   {
      k=omp_get_thread_num();
      ranlxs(r+k*NRAN,NRAN);
      ranlxd(rd+k*NRAN,NRAN);
   }

   tag=98029;
   export_ranlux(tag,"check1.dat");

# pragma omp parallel private(k)
   {
      k=omp_get_thread_num();
      ranlxs(r+k*NRAN,NRAN);
      ranlxd(rd+k*NRAN,NRAN);
      ranlxs(r+k*NRAN+NTHREAD*NRAN,NRAN);
      ranlxd(rd+k*NRAN+NTHREAD*NRAN,NRAN);
   }

   k=import_ranlux("check1.dat");
   error(k!=tag,1,"main [check1.c]",
         "Import_ranlux() returns incorrect tag");

# pragma omp parallel private(k)
   {
      k=omp_get_thread_num();
      ranlxs(r+k*NRAN+NTHREAD*NRAN,NRAN);
      ranlxd(rd+k*NRAN+NTHREAD*NRAN,NRAN);
   }

   ie=0;
   ied=0;

   for (k=0;k<(NTHREAD*NRAN);k++)
   {
      ie|=(r[k]!=r[NTHREAD*NRAN+k]);
      ied|=(rd[k]!=rd[NTHREAD*NRAN+k]);
   }

   error((ie!=0)||(ied!=0),1,"main [check1.c]",
         "Export/import of the generator states failed");

   if (my_rank==0)
   {
      remove("check1.dat");
      printf("No errors detected --- all is fine\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
