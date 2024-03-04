
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2011-2016, 2018, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the solver for the little Dirac equation.
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
#include "sw_term.h"
#include "sflds.h"
#include "vflds.h"
#include "linalg.h"
#include "dirac.h"
#include "sap.h"
#include "little.h"
#include "dfl.h"
#include "global.h"

static int my_rank,first,last,step,eoflg;
static double mu;
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
   read_dfl_parms("DFL");
   read_dfl_pro_parms("FGCR");
}


static void check_files(void)
{
   error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
         "check_files [check3.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);
}


static void new_subspace(void)
{
   int nb,isw,ifail;
   int Ns,n,nmax,k,l;
   spinor **mds,**ws;
   sap_parms_t sp;
   dfl_parms_t dfl;

   blk_list(SAP_BLOCKS,&nb,&isw);

   if (nb==0)
      alloc_bgr(SAP_BLOCKS);

   assign_ud2ubgr(SAP_BLOCKS);
   sw_term(NO_PTS);
   ifail=assign_swd2swbgr(SAP_BLOCKS,ODD_PTS);

   error(ifail!=0,1,"new_subspace [check3.c]",
         "Inversion of the SW term was not safe");

   sp=sap_parms();
   dfl=dfl_parms();
   Ns=dfl.Ns;
   nmax=6;
   ws=reserve_ws(Ns+1);
   mds=ws+1;

   for (k=0;k<Ns;k++)
   {
      random_s(VOLUME_TRD,2,mds[k],1.0f);
      bnd_s2zero(ALL_PTS,mds[k]);
   }

   for (n=0;n<nmax;n++)
   {
      for (k=0;k<Ns;k++)
      {
         assign_s2s(VOLUME_TRD,2,mds[k],ws[0]);
         set_s2zero(VOLUME_TRD,2,mds[k]);

         for (l=0;l<sp.ncy;l++)
            sap(0.01f,1,sp.nmr,mds[k],ws[0]);
      }

      for (k=0;k<Ns;k++)
      {
         for (l=0;l<k;l++)
            project(VOLUME_TRD,3,mds[k],mds[l]);

         (void)(normalize(VOLUME_TRD,3,mds[k]));
      }
   }

   dfl_subspace(mds);

   release_ws();
}


int main(int argc,char *argv[])
{
   int icnfg,nv,nvh,nvt,ifail,status[2];
   double m0,rho,nrm,del[2];
   double wt1,wt2,wdt;
   qflt rqsm;
   complex_dble z,**wvd;
   lat_parms_t lat;
   dfl_parms_t dfl;
   dfl_pro_parms_t dpr;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      fin=freopen("check3.in","r",stdin);

      printf("\n");
      printf("Check of the solver for the little Dirac equation\n");
      printf("-------------------------------------------------\n\n");

      print_lattice_sizes();
   }

   read_run_parms();

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
   dfl=dfl_parms();
   dpr=dfl_pro_parms();

   m0=lat.m0[0];
   set_sw_parms(m0);
   set_tm_parms(eoflg);
   nv=dfl.Ns*VOLUME/(dfl.bs[0]*dfl.bs[1]*dfl.bs[2]*dfl.bs[3]);
   nvh=nv/2;
   nvt=nvh/NTHREAD;

   if (my_rank==0)
   {
      printf("mu = %.6e\n",mu);
      printf("eoflg = %d\n\n",eoflg);

      print_iodat("i",iodat);

      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   alloc_ws(dfl.Ns+1);
   alloc_wv(2*dpr.nmx_gcr+3);
   alloc_wvd(2*dpr.nkv+7);
   wvd=reserve_wvd(4);

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

      new_subspace();
      set_Awhat(mu);
      random_vd(2*nvt,2,wvd[0],1.0);

      assign_vd2vd(nvt,2,wvd[0],wvd[1]);
      Awooinv_dble(wvd[0],wvd[1]);
      Aweo_dble(wvd[1],wvd[1]);
      Aweeinv_dble(wvd[1],wvd[2]);
      rqsm=vnorm_square_dble(nvt,3,wvd[2]);
      nrm=sqrt(rqsm.q[0]);

      assign_vd2vd(2*nvt,2,wvd[0],wvd[2]);

      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      rho=ltl_gcr(mu,wvd[0],wvd[1],&ifail,status);

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wdt=wt2-wt1;

      z.re=-1.0;
      z.im=0.0;
      mulc_vadd_dble(2*nvt,2,wvd[2],wvd[0],z);
      rqsm=vnorm_square_dble(2*nvt,3,wvd[2]);
      error_root(rqsm.q[0]!=0.0,1,"main [check3.c]",
                 "Source field is not preserved");

      Aw_dble(wvd[1],wvd[2]);
      mulc_vadd_dble(2*nvt,2,wvd[2],wvd[0],z);
      Aweeinv_dble(wvd[2],wvd[3]);
      Awooinv_dble(wvd[2],wvd[3]);
      rqsm=vnorm_square_dble(nvt,3,wvd[3]);
      del[0]=sqrt(rqsm.q[0]);
      rqsm=vnorm_square_dble(nvt,3,wvd[3]+nvh);
      del[1]=sqrt(rqsm.q[0]);

      if (my_rank==0)
      {
         printf("Solution when input and output fields are not the same:\n");
         printf("ifail = %d, status (fgcr4vd) = %d, status (gcr4v) = %d\n",
                ifail,status[0],status[1]);
         printf("rho (even) = %.2e (solver)\n",rho);
         printf("rho (even) = %.2e, rho (odd) = %.2e (check)\n",del[0],del[1]);
         printf("res = %.2e (required)\n",dpr.res);
         printf("    = %.2e (check)\n",del[0]/nrm);
         printf("time = %.2e sec (total)\n",wdt);
         printf("     = %.2e usec (per point, GCR iteration & thread)\n\n",
                (1.0e6*wdt)/((double)(status[1]*VOLUME_TRD)));
         fflush(flog);
      }

      rqsm=vnorm_square_dble(2*nvt,3,wvd[1]);
      nrm=rqsm.q[0];

      ltl_gcr(mu,wvd[0],wvd[0],&ifail,status);

      mulc_vadd_dble(2*nvt,2,wvd[0],wvd[1],z);
      rqsm=vnorm_square_dble(2*nvt,3,wvd[0]);
      del[0]=sqrt(rqsm.q[0]/nrm);

      if (my_rank==0)
      {
         printf("Solution when input and output fields coincide:\n");
         printf("ifail = %d, status (fgcr4vd) = %d, status (gcr4v) = %d\n",
                ifail,status[0],status[1]);
         printf("Relative difference = %.1e\n\n",del[0]);
         fflush(flog);
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
