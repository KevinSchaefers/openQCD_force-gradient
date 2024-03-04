
/*******************************************************************************
*
* File check4.c
*
* Copyright (C) 2011-2016, 2018, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check and performance of the deflated SAP+GCR solver.
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
#include "dfl.h"
#include "global.h"

static struct
{
   int nkv,nmx,istop;
   double mu,res;
} gcr_parms;

static int my_rank,first,last,step;
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
                 "read_run_parms [check4.c]","Improper configuration range");
   }

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   read_lat_parms("Lattice parameters",0x2);
   read_bc_parms("Boundary conditions",0x3);
   read_sap_parms("SAP",0x3);
   read_dfl_parms("Deflation subspace");
   read_dfl_pro_parms("Deflation projection");
   read_dfl_gen_parms("Deflation subspace generation");
}


static void read_gcr_parms(void)
{
   int nkv,nmx,istop,eoflg;
   double mu,res;

   if (my_rank==0)
   {
      find_section("GCR");

      read_line("mu","%lf",&mu);
      read_line("eoflg","%d",&eoflg);
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);
      read_line("istop","%d",&istop);
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&eoflg,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&istop,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   gcr_parms.mu=mu;
   gcr_parms.nkv=nkv;
   gcr_parms.nmx=nmx;
   gcr_parms.istop=istop;
   gcr_parms.res=res;

   (void)(set_tm_parms(eoflg));
}


static void check_files(void)
{
   error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
         "check_files [check4.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);
}


int main(int argc,char *argv[])
{
   int icnfg,ncnfg;
   int ifail,status[3],avgstat[3];
   int nkv,nmx,istop;
   double mu,res;
   double m0,rho,nrm,del,resm;
   double wt1,wt2,wdt,wta;
   qflt rqsm;
   spinor_dble **psd;
   lat_parms_t lat;
   dfl_parms_t dfl;
   dfl_pro_parms_t dpr;
   tm_parms_t tm;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check4.log","w",stdout);
      fin=freopen("check4.in","r",stdin);

      printf("\n");
      printf("Check and performance of the deflated SAP+GCR solver\n");
      printf("----------------------------------------------------\n\n");

      print_lattice_sizes();
   }

   read_run_parms();
   read_gcr_parms();

   if (my_rank==0)
      fclose(fin);

   check_machine();
   print_lat_parms(0x2);
   print_bc_parms(0x3);
   print_sap_parms(0x1);
   print_dfl_parms(0x0);

   start_ranlux(0,1234);
   geometry();
   check_files();

   lat=lat_parms();
   m0=lat.m0[0];
   set_sw_parms(m0);
   tm=tm_parms();
   mu=gcr_parms.mu;
   nkv=gcr_parms.nkv;
   nmx=gcr_parms.nmx;
   istop=gcr_parms.istop;
   res=gcr_parms.res;

   if (my_rank==0)
   {
      printf("GCR parameters:\n");
      printf("mu = %.6e\n",gcr_parms.mu);
      printf("eoflg = %d\n",tm.eoflg);
      printf("nkv = %d\n",nkv);
      printf("nmx = %d\n",nmx);
      printf("istop = %d\n",istop);
      printf("res = %.2e\n\n",res);

      print_iodat("i",iodat);

      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   dfl=dfl_parms();
   dpr=dfl_pro_parms();

   if (dfl.Ns<=(2*nkv))
      alloc_ws(2*nkv+2);
   else
      alloc_ws(dfl.Ns+2);
   alloc_wsd(5);
   alloc_wv(2*dpr.nmx_gcr+3);
   alloc_wvd(2*dpr.nkv+4);
   psd=reserve_wsd(3);

   ncnfg=0;
   avgstat[0]=0;
   avgstat[1]=0;
   avgstat[2]=0;
   resm=0.0;
   wta=0.0;

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

      dfl_modes(&ifail,status);

      if (my_rank==0)
      {
         printf("Subspace generation: ifail = %d, status = %d,%d\n",
                ifail,status[0],status[1]);
      }

      error_root(ifail<0,1,"main [check4.c]",
                 "Subspace generation failed");
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

      rho=dfl_sap_gcr(nkv,nmx,istop,res,mu,psd[0],psd[1],&ifail,status);

      mulr_spinor_add_dble(VOLUME_TRD,2,psd[2],psd[0],-1.0);
      del=unorm_dble(VOLUME_TRD,3,psd[2]);
      error_root(del!=0.0,1,"main [check4.c]",
                 "Source field is not preserved");

      Dw_dble(mu,psd[1],psd[2]);
      mulr_spinor_add_dble(VOLUME_TRD,2,psd[2],psd[0],-1.0);

      if (istop)
         del=unorm_dble(VOLUME_TRD,3,psd[2]);
      else
      {
         rqsm=norm_square_dble(VOLUME_TRD,3,psd[2]);
         del=sqrt(rqsm.q[0]);
      }

      if (my_rank==0)
      {
         printf("ifail = %d, status = %d,%d,%d\n",
                ifail,status[0],status[1],status[2]);
         printf("rho   = %.2e, res   = %.2e\n",rho,res);
         printf("check = %.2e, check = %.2e\n",del,del/nrm);
         fflush(flog);
      }

      if (ifail>=0)
      {
         ncnfg+=1;
         avgstat[0]+=status[0];
         avgstat[1]+=status[1];
         avgstat[2]+=status[2];
         del/=nrm;

         if (del>resm)
            resm=del;

         MPI_Barrier(MPI_COMM_WORLD);
         wt1=MPI_Wtime();

         rho=dfl_sap_gcr(nkv,nmx,istop,res,mu,psd[0],psd[0],&ifail,status);

         MPI_Barrier(MPI_COMM_WORLD);
         wt2=MPI_Wtime();
         wdt=wt2-wt1;
         wta+=wdt;

         mulr_spinor_add_dble(VOLUME_TRD,2,psd[0],psd[1],-1.0);
         del=unorm_dble(VOLUME_TRD,3,psd[0])/unorm_dble(VOLUME_TRD,3,psd[1]);

         if (my_rank==0)
         {
            printf("Relative difference when in- and ouput fields coincide: "
                   "%.1e\n",del);
            printf("Time = %.2e sec (w/o preparatory steps)\n",wdt);
            if (ifail>=0)
               printf("     = %.2e usec (per point, thread and GCR iteration)\n",
                      (1.0e6*wdt)/((double)(status[0])*(double)(VOLUME_TRD)));
            printf("\n");
            fflush(flog);
         }

      }
   }

   if (my_rank==0)
   {
      printf("Summary of results\n");
      printf("------------------\n\n");

      printf("Processed %d configurations\n",ncnfg);
      printf("Solver failed in %d cases\n",(last-first)/step+1-ncnfg);
      printf("Maximal relative residue = %.1e\n",resm);

      status[0]=(avgstat[0]+ncnfg/2)/ncnfg;
      status[1]=(avgstat[1]+ncnfg/2)/ncnfg;
      status[2]=(avgstat[2]+ncnfg/2)/ncnfg;
      wta/=(double)(ncnfg);

      printf("Average status = %d,%d,%d\n",status[0],status[1],status[2]);
      printf("Average time = %.2e sec (w/o preparatory steps)\n",wta);
      printf("             = %.2e usec (per point, thread and GCR iteration)\n",
             (1.0e6*wta)/((double)(status[0])*(double)(VOLUME_TRD)));
      printf("\n");

      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
