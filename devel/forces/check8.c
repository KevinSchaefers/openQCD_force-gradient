
/*******************************************************************************
*
* File check8.c
*
* Copyright (C) 2012, 2016, 2018, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check and performance of the multi-shift CG solver.
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

static int my_rank,first,last,step,nmu,nmx;
static double *mu,*res;
static iodat_t iodat[1];
static char nbase[NAME_SIZE],cnfg_file[NAME_SIZE];
static FILE *flog=NULL,*fin=NULL;


static void read_run_parms(void)
{
   int nres;

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
                 "read_run_parms [check8.c]","Improper configuration range");
   }

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   read_lat_parms("Lattice parameters",0x2);

   if (my_rank==0)
         nmu=count_tokens("mu");

   MPI_Bcast(&nmu,1,MPI_INT,0,MPI_COMM_WORLD);
   mu=malloc(nmu*sizeof(*mu));
   error(mu==NULL,1,"read_run_parms [check8.c]",
         "Unable to allocate auxiliary array");

   if (my_rank==0)
      read_dprms("mu",nmu,mu);

   MPI_Bcast(mu,nmu,MPI_DOUBLE,0,MPI_COMM_WORLD);

   read_bc_parms("Boundary conditions",0x3);

   if (my_rank==0)
   {
      find_section("CG");
      read_line("nmx","%d",&nmx);
      nres=count_tokens("res");
   }

   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nres,1,MPI_INT,0,MPI_COMM_WORLD);

   error_root(nres<nmu,1,"read_run_parms [check8.c]",
              "Numbers of residues and twisted masses do not match");
   res=malloc(nres*sizeof(*res));
   error(res==NULL,1,"read_run_parms [check8.c]",
         "Unable to allocate auxiliary array");

   if (my_rank==0)
      read_dprms("res",nres,res);

   MPI_Bcast(res,nres,MPI_DOUBLE,0,MPI_COMM_WORLD);
}


static void check_files(void)
{
   error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
         "check_files [check8.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);
}


int main(int argc,char *argv[])
{
   int icnfg,istop,ifail,status,k,ie;
   double m0,nrm,del;
   double wt1,wt2,wdt;
   qflt rqsm;
   spinor_dble *eta,*chi,*phi,**psi,**wsd,**rsd;
   lat_parms_t lat;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check8.log","w",stdout);
      fin=freopen("check8.in","r",stdin);

      printf("\n");
      printf("Check and performance of the multi-shift CG solver\n");
      printf("--------------------------------------------------\n\n");

      print_lattice_sizes();
   }

   read_run_parms();
   if (my_rank==0)
      fclose(fin);

   check_machine();
   print_lat_parms(0x2);
   print_bc_parms(0x2);

   if (my_rank==0)
   {
      printf("CG parameters:\n");
      printf("mu = %.6f",mu[0]);
      for (k=1;k<nmu;k++)
         printf(", %.6f",mu[k]);
      printf("\n");
      printf("nmx = %d\n",nmx);
      printf("res = %.2e",res[0]);
      for (k=1;k<nmu;k++)
         printf(", %.2e",res[k]);
      printf("\n\n");

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

   if (nmu==1)
      alloc_wsd(8);
   else
      alloc_wsd(5+2*nmu);

   wsd=reserve_wsd(2);
   eta=wsd[0];
   chi=wsd[1];
   psi=reserve_wsd(nmu);
   ie=0;

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
         random_sd(VOLUME_TRD,2,eta,1.0);
         bnd_sd2zero(ALL_PTS,eta);
         assign_sd2sd(VOLUME_TRD,2,eta,chi);

         if (istop)
            nrm=unorm_dble(VOLUME_TRD/2,3,eta);
         else
         {
            rqsm=norm_square_dble(VOLUME_TRD/2,3,eta);
            nrm=sqrt(rqsm.q[0]);
         }

         MPI_Barrier(MPI_COMM_WORLD);
         wt1=MPI_Wtime();

         tmcgm(nmx,istop,res,nmu,mu,eta,psi,&ifail,&status);

         MPI_Barrier(MPI_COMM_WORLD);
         wt2=MPI_Wtime();
         wdt=wt2-wt1;

         mulr_spinor_add_dble(VOLUME_TRD,2,chi,eta,-1.0);
         del=unorm_dble(VOLUME_TRD,3,chi);
         error_root(del!=0.0,1,"main [check8.c]",
                    "Source field is not preserved");

         if (my_rank==0)
         {
            printf("istop = %d, ifail = %d, status = %d\n",istop,ifail,status);
            printf("time = %.2e sec (total)\n",wdt);
            if (ifail==0)
               printf("     = %.2e usec (per point, thread and CG iteration)\n",
                      (1.0e6*wdt)/((double)(status)*(double)(VOLUME_TRD)));
            fflush(flog);
            error_root(ifail<0,1,"main [check8.c]","Solver did not converge");
            printf("residues = ");
         }

         rsd=reserve_wsd(1);
         phi=rsd[0];
         status=0;

         for (k=0;k<nmu;k++)
         {
            Dwhat_dble(mu[k],psi[k],chi);
            mulg5_dble(VOLUME_TRD/2,2,chi);
            Dwhat_dble(-mu[k],chi,phi);
            mulg5_dble(VOLUME_TRD/2,2,phi);
            mulr_spinor_add_dble(VOLUME_TRD/2,2,phi,eta,-1.0);

            if (istop)
               del=unorm_dble(VOLUME_TRD/2,3,phi)/nrm;
            else
            {
               rqsm=norm_square_dble(VOLUME_TRD/2,3,phi);
               del=sqrt(rqsm.q[0])/nrm;
            }

            if (del<res[k])
               status+=1;

            if (my_rank==0)
            {
               if (k==0)
                  printf("%.2e",del);
               else
                  printf(", %.2e",del);
            }
         }

         release_wsd();
         ie+=(status<nmu);

         if (my_rank==0)
         {
            printf("\n");

            if (status==nmu)
               printf("All residues are as required\n\n");
            else
               printf("ERROR: %d residues are too large\n\n",nmu-status);

            fflush(flog);
         }
      }
   }

   if (my_rank==0)
   {
      if (ie==0)
         printf("No errors detected --- all seems fine!\n\n");
      else
         printf("ERROR: the residues are too large (%d configurations)\n\n",ie);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
