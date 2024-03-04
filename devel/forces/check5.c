
/*******************************************************************************
*
* File check5.c
*
* Copyright (C) 2011-2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check and performance of the CG solver.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
#include "uflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "forces.h"
#include "global.h"

static int my_rank,first,last,step,nmx;
static double mu,res;
static iodat_t iodat[1];
static char nbase[NAME_SIZE],cnfg_file[NAME_SIZE];
static FILE *flog=NULL,*fin=NULL;


static void read_run_parms(void)
{
   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   read_iodat("Configurations","i",iodat);

   if (my_rank==0)
   {
      read_line("first","%d",&first);
      read_line("last","%d",&last);
      read_line("step","%d",&step);

      error_root((first<1)||(last<first)||(step<1)||((last-first)%step!=0),1,
                 "read_run_parms [check5.c]","Improper configuration range");
   }

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   read_lat_parms("Lattice parameters",0x2);

   if (my_rank==0)
      read_line("mu","%lf",&mu);

   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   read_bc_parms("Boundary conditions",0x3);

   if (my_rank==0)
   {
      find_section("CG");
      read_line("nmx","%d",&nmx);
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
}


static void check_files(void)
{
   error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
         "check_files [check5.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);
}


static void Dhatop_dble(spinor_dble *s,spinor_dble *r)
{
   Dwhat_dble(mu,s,r);
   mulg5_dble(VOLUME_TRD/2,2,r);
   mu=-mu;
}


int main(int argc,char *argv[])
{
   int icnfg,istop,ifail,status;
   double m0,rho,nrm,del;
   double wt1,wt2,wdt;
   qflt rqsm;
   complex_dble z;
   spinor_dble **psd;
   lat_parms_t lat;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check5.log","w",stdout);
      fin=freopen("check5.in","r",stdin);

      printf("\n");
      printf("Check and performance of the CG solver\n");
      printf("--------------------------------------\n\n");

      print_lattice_sizes();
   }

   read_run_parms();
   if (my_rank==0)
      fclose(fin);

   check_machine();
   print_lat_parms(0x2);
   print_bc_parms(0x3);

   if (my_rank==0)
   {
      printf("CG parameters:\n");
      printf("mu = %.6f\n\n",mu);
      printf("nmx = %d\n",nmx);
      printf("res = %.2e\n\n",res);

      print_iodat("i",iodat);

      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   start_ranlux(0,1234);
   geometry();
   check_files();

   lat=lat_parms();
   m0=lat.m0[0];
   set_sw_parms(m0);

   alloc_ws(5);
   alloc_wsd(6);
   psd=reserve_wsd(3);

   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      if (my_rank==0)
      {
         printf("Configuration no %d\n\n",icnfg);
         fflush(flog);
      }

      sprintf(cnfg_file,"%sn%d",nbase,icnfg);
      read_flds(iodat,cnfg_file,0x0,0x1);
      set_ud_phase();

      for (istop=0;istop<2;istop++)
      {
         random_sd(VOLUME_TRD,2,psd[0],1.0);
         bnd_sd2zero(ALL_PTS,psd[0]);
         assign_sd2sd(VOLUME_TRD,2,psd[0],psd[2]);

         if (istop)
            nrm=unorm_dble(VOLUME_TRD,3,psd[0]);
         else
         {
            rqsm=norm_square_dble(VOLUME_TRD,3,psd[0]);
            nrm=sqrt(rqsm.q[0]);
         }

         MPI_Barrier(MPI_COMM_WORLD);
         wt1=MPI_Wtime();

         rho=tmcg(nmx,istop,res,mu,psd[0],psd[1],&ifail,&status);

         MPI_Barrier(MPI_COMM_WORLD);
         wt2=MPI_Wtime();
         wdt=wt2-wt1;

         z.re=-1.0;
         z.im=0.0;
         mulc_spinor_add_dble(VOLUME_TRD,2,psd[2],psd[0],z);
         del=unorm_dble(VOLUME_TRD,3,psd[2]);
         error_root(del!=0.0,1,"main [check5.c]",
                    "Source field is not preserved");

         Dw_dble(mu,psd[1],psd[2]);
         mulg5_dble(VOLUME_TRD,2,psd[2]);
         Dw_dble(-mu,psd[2],psd[1]);
         mulg5_dble(VOLUME_TRD,2,psd[1]);
         mulc_spinor_add_dble(VOLUME_TRD,2,psd[1],psd[0],z);

         if (istop)
            del=unorm_dble(VOLUME_TRD,3,psd[1]);
         else
         {
            rqsm=norm_square_dble(VOLUME_TRD,3,psd[1]);
            del=sqrt(rqsm.q[0]);
         }

         if (my_rank==0)
         {
            printf("Solution w/o eo-preconditioning:\n");
            printf("istop = %d, ifail = %d, status = %d\n",istop,ifail,status);
            printf("rho   = %.2e, res   = %.2e\n",rho,res);
            printf("check = %.2e, check = %.2e\n",del,del/nrm);
            printf("time = %.2e sec (total)\n",wdt);
            if (ifail==0)
               printf("     = %.2e usec (per point, thread and CG iteration)",
                      (1.0e6*wdt)/((double)(status)*(double)(VOLUME_TRD)));
            printf("\n\n");
            fflush(flog);
         }

         random_sd(VOLUME_TRD,2,psd[0],1.0);
         bnd_sd2zero(EVEN_PTS,psd[0]);
         assign_sd2sd(VOLUME_TRD/2,2,psd[0],psd[2]);

         if (istop)
            nrm=unorm_dble(VOLUME_TRD/2,3,psd[0]);
         else
         {
            rqsm=norm_square_dble(VOLUME_TRD/2,3,psd[0]);
            nrm=sqrt(rqsm.q[0]);
         }

         MPI_Barrier(MPI_COMM_WORLD);
         wt1=MPI_Wtime();

         rho=tmcgeo(nmx,istop,res,mu,psd[0],psd[1],&ifail,&status);

         MPI_Barrier(MPI_COMM_WORLD);
         wt2=MPI_Wtime();
         wdt=wt2-wt1;

         z.re=-1.0;
         z.im=0.0;
         mulc_spinor_add_dble(VOLUME_TRD/2,2,psd[2],psd[0],z);
         del=unorm_dble(VOLUME_TRD/2,3,psd[2]);
         error_root(del!=0.0,1,"main [check5.c]",
                    "Source field is not preserved");

         Dhatop_dble(psd[1],psd[2]);
         Dhatop_dble(psd[2],psd[1]);
         mulc_spinor_add_dble(VOLUME_TRD/2,2,psd[1],psd[0],z);

         if (istop)
            del=unorm_dble(VOLUME_TRD/2,3,psd[1]);
         else
         {
            rqsm=norm_square_dble(VOLUME_TRD/2,3,psd[1]);
            del=sqrt(rqsm.q[0]);
         }

         if (my_rank==0)
         {
            printf("Solution with eo-preconditioning:\n");
            printf("istop = %d, ifail = %d, status = %d\n",istop,ifail,status);
            printf("rho   = %.2e, res   = %.2e\n",rho,res);
            printf("check = %.2e, check = %.2e\n",del,del/nrm);
            printf("time = %.2e sec (total)\n",wdt);
            if (ifail==0)
               printf("     = %.2e usec (per point, thread and CG iteration)",
                      (1.0e6*wdt)/((double)(status)*(double)(VOLUME_TRD)));
            printf("\n\n");
            fflush(flog);
         }
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
