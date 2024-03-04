
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2011-2013, 2016, 2018, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check and performance of the SAP+GCR solver.
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
#include "dirac.h"
#include "sap.h"
#include "global.h"

static int my_rank,first,last,step;
static int nkv,nmx,istop,eoflg;
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
                 "read_run_parms [check3.c]","Improper configuration range");
   }

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   read_lat_parms("Lattice parameters",0x2);

   if (my_rank==0)
   {
      read_line("mu","%lf",&mu);
      read_line("eoflg","%d",&eoflg);
   }

   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&eoflg,1,MPI_INT,0,MPI_COMM_WORLD);

   read_bc_parms("Boundary conditions",0x3);
   read_sap_parms("SAP",0x3);
}


static void check_files(void)
{
   error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
         "check_files [check3.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);
}


static void read_GCR_parms(void)
{
   if (my_rank==0)
   {
      find_section("GCR");
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);
      read_line("istop","%d",&istop);
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&istop,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
}


int main(int argc,char *argv[])
{
   int isolv,icnfg,ifail,status;
   double m0,rho,nrm,del;
   double wt1,wt2,wdt;
   qflt qnrm;
   spinor_dble **psd;
   lat_parms_t lat;
   sap_parms_t sap;
   tm_parms_t tm;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      fin=freopen("check3.in","r",stdin);

      printf("\n");
      printf("Check and performance of the SAP+GCR solver\n");
      printf("-------------------------------------------\n\n");

      print_lattice_sizes();
   }

   read_run_parms();
   read_GCR_parms();

   if (my_rank==0)
      fclose(fin);

   check_machine();
   print_lat_parms(0x2);
   print_bc_parms(0x2);

   start_ranlux(0,1234);
   geometry();
   check_files();

   lat=lat_parms();
   m0=lat.m0[0];
   (void)(set_sw_parms(m0));
   sap=sap_parms();
   tm=set_tm_parms(eoflg);

   if (my_rank==0)
   {
      printf("SAP+GCR parameters:\n");
      printf("mu = %.6f\n",mu);
      printf("eoflg = %d\n\n",tm.eoflg);

      printf("bs = (%d,%d,%d,%d)\n",sap.bs[0],sap.bs[1],sap.bs[2],sap.bs[3]);
      printf("nmr = %d\n",sap.nmr);
      printf("ncy = %d\n\n",sap.ncy);

      printf("nkv = %d\n",nkv);
      printf("nmx = %d\n",nmx);
      printf("istop = %d\n",istop);
      printf("res = %.2e\n\n",res);

      print_iodat("i",iodat);

      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   alloc_ws(2*nkv+1);
   alloc_wsd(5);
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
      random_sd(VOLUME_TRD,2,psd[0],1.0);
      bnd_sd2zero(ALL_PTS,psd[0]);

      if (istop)
         nrm=unorm_dble(VOLUME_TRD,3,psd[0]);
      else
      {
         qnrm=norm_square_dble(VOLUME_TRD,3,psd[0]);
         nrm=sqrt(qnrm.q[0]);
      }

      for (isolv=0;isolv<2;isolv++)
      {
         assign_sd2sd(VOLUME_TRD,2,psd[0],psd[2]);
         set_sap_parms(sap.bs,isolv,sap.nmr,sap.ncy);

         rho=sap_gcr(nkv,nmx,istop,res,mu,psd[0],psd[1],&ifail,&status);

         mulr_spinor_add_dble(VOLUME_TRD,2,psd[2],psd[0],-1.0);
         del=unorm_dble(VOLUME_TRD,3,psd[2]);
         error_root(del!=0.0,1,"main [check3.c]",
                    "Source field is not preserved");

         Dw_dble(mu,psd[1],psd[2]);
         mulr_spinor_add_dble(VOLUME_TRD,2,psd[2],psd[0],-1.0);

         if (istop)
            del=unorm_dble(VOLUME_TRD,3,psd[2]);
         else
         {
            qnrm=norm_square_dble(VOLUME_TRD,3,psd[2]);
            del=sqrt(qnrm.q[0]);
         }

         if (my_rank==0)
         {
            printf("isolv = %d:\n",isolv);
            printf("ifail = %d, status = %d\n",ifail,status);
            printf("rho   = %.2e, res   = %.2e\n",rho,res);
            printf("check = %.2e, check = %.2e\n",del,del/nrm);
         }

         assign_sd2sd(VOLUME_TRD,2,psd[0],psd[2]);

         MPI_Barrier(MPI_COMM_WORLD);
         wt1=MPI_Wtime();

         rho=sap_gcr(nkv,nmx,istop,res,mu,psd[2],psd[2],&ifail,&status);

         MPI_Barrier(MPI_COMM_WORLD);
         wt2=MPI_Wtime();
         wdt=wt2-wt1;

         mulr_spinor_add_dble(VOLUME_TRD,2,psd[2],psd[1],-1.0);
         del=unorm_dble(VOLUME_TRD,3,psd[2]);
         del/=unorm_dble(VOLUME_TRD,3,psd[1]);

         if (my_rank==0)
         {
            printf("time = %.2e sec (total)\n",wdt);
            if (status>0)
               printf("     = %.2e usec "
                      "(per point, thread and GCR iteration)\n",
                      (1.0e6*wdt)/((double)(status)*(double)(VOLUME_TRD)));
            printf("Relative difference when in- and ouput fields coincide: "
                   "%.1e\n\n",del);
            fflush(flog);
         }
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
