
/*******************************************************************************
*
* File ms4.c
*
* Copyright (C) 2012, 2013, 2016-2019, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of quark propagators.
*
* Syntax: ms4 -i <input file>
*
* For usage instructions see the file README.ms4.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
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
#include "dfl.h"
#include "forces.h"
#include "version.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int my_rank,endian;
static int first,last,step;
static int level,seed,x0,nsrc;
static int ifail0[2],*stat0=NULL;
static double mus;

static iodat_t iodat[2];
static char log_dir[NAME_SIZE],end_file[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE];
static char nbase[NAME_SIZE],cnfg_file[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fend=NULL;


static void read_dirs(void)
{
   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);

      find_section("Log directory");
      read_line("log_dir","%s",log_dir);
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
}


static void setup_files(void)
{
   error(name_size("%s/%s.ms4.log~",log_dir,nbase)>=NAME_SIZE,1,
         "setup_files [ms4.c]","log_dir name is too long");

   sprintf(log_file,"%s/%s.ms4.log",log_dir,nbase);
   sprintf(end_file,"%s/%s.ms4.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);

   check_dir_root(log_dir);
}


static void read_cnfg_range(void)
{
   if (my_rank==0)
   {
      find_section("Configurations");

      read_line("first","%d",&first);
      read_line("last","%d",&last);
      read_line("step","%d",&step);

      error_root((first<1)||(last<first)||(step<1)||((last-first)%step!=0),1,
                 "read_cnfg_range [ms4.c]","Improper configuration range");
   }

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void read_ranlux_parms(void)
{
   if (my_rank==0)
   {
      find_section("Random number generator");
      read_line("level","%d",&level);
      read_line("seed","%d",&seed);
   }

   MPI_Bcast(&level,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void read_src_parms(void)
{
   int ie;

   if (my_rank==0)
   {
      find_section("Dirac operator");
      read_line("mu","%lf",&mus);

      find_section("Source fields");
      read_line("x0","%d",&x0);
      read_line("nsrc","%d",&nsrc);

      ie=((x0<0)||(x0>=N0));
      ie|=((x0==0)&&(bc_type()!=3));
      ie|=((x0==(N0-1))&&(bc_type()==0));

      error_root(ie!=0,1,"read_src_parms [ms4.c]",
                 "Source time x0 is out of range");
      error_root(nsrc<1,1,"read_src_parms [ms4.c]",
                 "The number of source fields must be at least 1");
   }

   MPI_Bcast(&mus,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&x0,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nsrc,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void read_solver(int *isap,int *idfl)
{
   solver_parms_t sp;

   read_solver_parms(0);
   sp=solver_parms(0);
   (*isap)=0;
   (*idfl)=0;

   if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
      (*isap)=1;

   if (sp.solver==DFL_SAP_GCR)
      (*idfl)=1;
}


static void read_infile(int argc,char *argv[])
{
   int ifile,isap,idfl;

   if (my_rank==0)
   {
      flog=freopen("STARTUP_ERROR","w",stdout);

      ifile=find_opt(argc,argv,"-i");
      endian=endianness();

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [ms4.c]",
                 "Syntax: ms4 -i <input file>");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [ms4.c]",
                 "Machine has unknown endianness");

      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [ms4.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);

   read_dirs();
   read_iodat("Configurations","i",iodat);
   read_iodat("Propagators","o",iodat+1);
   read_cnfg_range();
   read_ranlux_parms();
   read_lat_parms("Dirac operator",0x2);
   read_bc_parms("Boundary conditions",0x2);
   read_src_parms();
   read_solver(&isap,&idfl);
   if (isap)
      read_sap_parms("SAP",0x1);
   if (idfl)
   {
      read_dfl_parms("Deflation subspace");
      read_dfl_pro_parms("Deflation projection");
      read_dfl_gen_parms("Deflation subspace generation");
   }

   setup_files();

   if (my_rank==0)
      fclose(fin);
}


static void check_files(void)
{
   int ie;

   if (my_rank==0)
   {
      ie=check_file(log_file,"r");
      error_root(ie!=0,1,"check_files [ms4.c]",
                 "Attempt to overwrite old *.log file");
   }

   error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
         "check_files [ms4.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);

   error(name_size("%sn%d.s%d",nbase,last,nsrc-1)>=NAME_SIZE,2,
         "check_files [ms4.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d.s%d",nbase,last,nsrc-1);
   check_iodat(iodat+1,"o",0x1,cnfg_file);
}


static void print_info(void)
{
   int n,isap,idfl;
   long ip;
   lat_parms_t lat;

   if (my_rank==0)
   {
      ip=ftell(flog);
      fclose(flog);

      if (ip==0L)
         remove("STARTUP_ERROR");

      flog=freopen(log_file,"w",stdout);
      error_root(flog==NULL,1,"print_info [ms4.c]","Unable to open log file");
      printf("\n");

      printf("Computation of quark propagators\n");
      printf("--------------------------------\n\n");

      printf("Program version %s\n",openQCD_RELEASE);

      if (endian==LITTLE_ENDIAN)
         printf("The machine is little endian\n");
      else
         printf("The machine is big endian\n");

      print_lattice_sizes();

      printf("Random number generator:\n");
      printf("level = %d, seed = %d\n\n",level,seed);

      printf("Dirac operator:\n");
      lat=lat_parms();
      n=fdigits(lat.kappa[0]);
      printf("kappa = %.*f\n",IMAX(n,6),lat.kappa[0]);
      n=fdigits(mus);
      printf("mu = %.*f\n",IMAX(n,1),mus);
      n=fdigits(lat.csw);
      printf("isw = %d, csw = %.*f\n\n",lat.isw,IMAX(n,1),lat.csw);
      print_bc_parms(0x2);

      printf("Source fields:\n");
      printf("x0 = %d\n",x0);
      printf("nsrc = %d\n\n",nsrc);

      print_solver_parms(&isap,&idfl);

      if (isap)
         print_sap_parms(0x0);

      if (idfl)
         print_dfl_parms(0x0);

      print_iodat("i",iodat);
      print_iodat("o",iodat+1);

      printf("Configurations no %d -> %d in steps of %d\n\n",
             first,last,step);
      fflush(flog);
   }
}


static void maxn(int *n,int m)
{
   if ((*n)<m)
      (*n)=m;
}


static void dfl_wsize(int *nws,int *nwv,int *nwvd)
{
   dfl_parms_t dp;
   dfl_pro_parms_t dpr;

   dp=dfl_parms();
   dpr=dfl_pro_parms();

   maxn(nws,dp.Ns+2);
   maxn(nwv,2*dpr.nmx_gcr+3);
   maxn(nwvd,2*dpr.nkv+4);
}


static void solver_wsize(int isp,int nsds,int *nws,int *nwv,int *nwvd)
{
   solver_parms_t sp;

   sp=solver_parms(isp);

   if (sp.solver==CGNE)
      maxn(nws,nsds+11);
   else if (sp.solver==SAP_GCR)
      maxn(nws,nsds+2*sp.nkv+5);
   else if (sp.solver==DFL_SAP_GCR)
   {
      maxn(nws,nsds+2*sp.nkv+6);
      dfl_wsize(nws,nwv,nwvd);
   }
   else
      error_root(1,1,"solver_wsize [ms4.c]","Unknown or unsupported solver");
}


static void wsize(int *nws,int *nwv,int *nwvd)
{
   (*nws)=0;
   (*nwv)=0;
   (*nwvd)=0;

   solver_wsize(0,4,nws,nwv,nwvd);
}


static void init_stat(int *status)
{
   if (stat0==NULL)
      stat0=alloc_std_status();

   ifail0[0]=0;
   ifail0[1]=0;
   reset_std_status(stat0);
   reset_std_status(status);
}


static void print_stat(int *status)
{
   solver_parms_t sp;

   sp=solver_parms(0);

   if (sp.solver==CGNE)
      print_std_status("tmcgeo",NULL,status);
   else if (sp.solver==SAP_GCR)
      print_std_status("sap_gcr",NULL,status);
   else if (sp.solver==DFL_SAP_GCR)
      print_std_status("dfl_sap_gcr2",NULL,status);
}


static void random_source(spinor_dble *eta)
{
   int k,ofs,vol,ix,t;
   spinor_dble *sd;

#pragma omp parallel private(k,ofs,vol,ix,t,sd)
   {
      k=omp_get_thread_num();
      vol=VOLUME_TRD;
      ofs=k*vol;
      sd=eta+ofs;

      for (ix=ofs;ix<(ofs+vol);ix++)
      {
         t=global_time(ix);

         if (t==x0)
            random_sd(1,0,sd,1.0);
         else
            set_sd2zero(1,0,sd);

         sd+=1;
      }
   }
}


static void solve_dirac(spinor_dble *eta,spinor_dble *psi,int *status)
{
   solver_parms_t sp;
   sap_parms_t sap;

   sp=solver_parms(0);

   if (sp.solver==CGNE)
   {
      mulg5_dble(VOLUME_TRD,2,eta);

      tmcg(sp.nmx,sp.istop,sp.res,mus,eta,eta,ifail0,stat0);
      acc_std_status("tmcg",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("tmcg",ifail0,stat0);
         error_root(1,1,"solve_dirac [ms4.c]","CGNE solver failed "
                    "(mu = %.2e, parameter set no 0)",mus);
      }

      Dw_dble(-mus,eta,psi);
      mulg5_dble(VOLUME_TRD,2,psi);
   }
   else if (sp.solver==SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mus,eta,psi,ifail0,stat0);
      acc_std_status("sap_gcr",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("sap_gcr",ifail0,stat0);
         error_root(1,1,"solve_dirac [ms4.c]","SAP_GCR solver failed "
                    "(mu = %.2e, parameter set no 0)",mus);
      }
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mus,eta,psi,ifail0,stat0);
      acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);

      if ((ifail0[0]<-2)||(ifail0[1]<0))
      {
         print_status("dfl_sap_gcr2",ifail0,stat0);
         error_root(1,1,"solve_dirac [ms4.c]","DFL_SAP_GCR solver failed "
                    "(mu = %.2e, parameter set no 0)",mus);
      }
   }
   else
      error_root(1,1,"solve_dirac [ms4.c]",
                 "Unknown or unsupported solver");
}


static void save_prop(int icnfg,int isrc,spinor_dble *sd)
{
   int n,ib,types;

   types=iodat[1].types;

   if (types&0x1)
   {
      sprintf(cnfg_file,"%s/%sn%d.s%d",
              iodat[1].cnfg_dir,nbase,icnfg,isrc);
      export_sfld(cnfg_file,0,sd);
   }

   if (types&0x2)
   {
      set_nio_streams(iodat[1].nio_streams);
      ib=iodat[1].ib;
      n=iodat[1].nb/iodat[1].nio_nodes;
      sprintf(cnfg_file,"%s/%d/%d/%sn%d.s%d_b%d",
              iodat[1].block_dir,ib/n,ib%n,nbase,icnfg,isrc,ib);
      blk_export_sfld(iodat[1].bs,cnfg_file,0,sd);
   }

   if (types&0x4)
   {
      set_nio_streams(iodat[1].nio_streams);
      n=NPROC/iodat[1].nio_nodes;
      sprintf(cnfg_file,"%s/%d/%d/%sn%d.s%d_%d",
              iodat[1].local_dir,my_rank/n,my_rank%n,nbase,icnfg,isrc,my_rank);
      write_sfld(cnfg_file,0,sd);
   }
}


static void propagator(int nc,int *status,double *wtsum)
{
   int isrc;
   double wt[2];
   spinor_dble *eta,*psi,**wsd;

   init_stat(status);

   wsd=reserve_wsd(2);
   eta=wsd[0];
   psi=wsd[1];
   wtsum[0]=0.0;
   wtsum[1]=0.0;

   for (isrc=0;isrc<nsrc;isrc++)
   {
      random_source(eta);

      MPI_Barrier(MPI_COMM_WORLD);
      wt[0]=MPI_Wtime();

      solve_dirac(eta,psi,status);

      MPI_Barrier(MPI_COMM_WORLD);
      wt[1]=MPI_Wtime();
      wtsum[0]+=(wt[1]-wt[0]);

      save_prop(nc,isrc,psi);
      MPI_Barrier(MPI_COMM_WORLD);
      wt[0]=MPI_Wtime();
      wtsum[1]+=(wt[0]-wt[1]);
   }

   avg_std_status(nsrc,status);
   wtsum[0]/=(double)(nsrc);
   wtsum[1]/=(double)(nsrc);

   release_wsd();
}


static void check_endflag(int *iend)
{
   if (my_rank==0)
   {
      fend=fopen(end_file,"r");

      if (fend!=NULL)
      {
         fclose(fend);
         remove(end_file);
         (*iend)=1;
         printf("End flag set, run stopped\n\n");
      }
      else
         (*iend)=0;
   }

   MPI_Bcast(iend,1,MPI_INT,0,MPI_COMM_WORLD);
}


int main(int argc,char *argv[])
{
   int nc,iend,*status;
   int nws,nwv,nwvd,n;
   double wt[2],wtavg[2];
   dfl_parms_t dfl;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   check_machine();
   geometry();
   check_files();
   print_info();
   start_ranlux(level,seed);

   wsize(&nws,&nwv,&nwvd);
   alloc_ws(nws);
   wsd_uses_ws();
   alloc_wv(nwv);
   alloc_wvd(nwvd);
   status=alloc_std_status();

   iend=0;
   wtavg[0]=0.0;
   wtavg[1]=0.0;

   set_sw_parms(sea_quark_mass(0));
   dfl=dfl_parms();

   for (nc=first;(iend==0)&&(nc<=last);nc+=step)
   {
      if (my_rank==0)
      {
         printf("Configuration no %d\n",nc);
         fflush(flog);
      }

      sprintf(cnfg_file,"%sn%d",nbase,nc);
      read_flds(iodat,cnfg_file,0x0,0x1);
      set_ud_phase();

     if (dfl.Ns)
      {
         dfl_modes2(ifail0,status);

         if (my_rank==0)
         {
            printf("Generation of the deflation subspace\n");
            print_status("dfl_modes2",ifail0,status);
         }

         error_root((ifail0[0]<-2)||(ifail0[1]<0),1,"main [ms4.c]",
                    "Deflation subspace generation failed");
      }

      propagator(nc,status,wt);
      wtavg[0]+=wt[0];
      wtavg[1]+=wt[1];

      if (my_rank==0)
      {
         printf("Computation of propagator completed\n");
         print_stat(status);
         n=(nc-first)/step+1;

         printf("Dirac equation solved in %.2e sec per source field "
                "(average %.2e sec)\n",wt[0],wtavg[0]/(double)(n));
         printf("Solution saved in %.2e sec (average %.2e sec)\n",
                wt[1],wtavg[1]/(double)(n));
         printf("Configuration no %d fully processed in %.2e sec ",
                nc,(double)(nsrc)*(wt[0]+wt[1]));
         printf("(average = %.2e sec)\n\n",
                (double)(nsrc)*(wtavg[0]+wtavg[1])/(double)(n));

         fflush(flog);
         copy_file(log_file,log_save);
      }

      check_endflag(&iend);
   }

   if (my_rank==0)
   {
      fflush(flog);
      copy_file(log_file,log_save);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
