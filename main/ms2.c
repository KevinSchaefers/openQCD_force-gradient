
/*******************************************************************************
*
* File ms2.c
*
* Copyright (C) 2012, 2013, 2016-2019, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the spectral range of the Hermitian Dirac operator.
*
* Syntax: ms2 -i <input file>
*
* For usage instructions see the file README.ms2.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
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
#include "sap.h"
#include "dfl.h"
#include "ratfcts.h"
#include "forces.h"
#include "version.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int my_rank,endian;
static int first,last,step,np_ra,np_rb;
static int ifail0[2],ifail1[2],*stat0=NULL,*stat1;
static double ar[256];

static iodat_t iodat[1];
static char nbase[NAME_SIZE],log_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE],end_file[NAME_SIZE];
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
   error(name_size("%s/%s.ms2.log~",log_dir,nbase)>=NAME_SIZE,1,
         "setup_files [ms2.c]","log_dir name is too long");

   sprintf(log_file,"%s/%s.ms2.log",log_dir,nbase);
   sprintf(end_file,"%s/%s.ms2.end",log_dir,nbase);
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
                 "read_cnfg_range [ms2.c]","Improper configuration range");
   }

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void read_power_parms(void)
{
      if (my_rank==0)
   {
      find_section("Power method");
      read_line("np_ra","%d",&np_ra);
      read_line("np_rb","%d",&np_rb);
      error_root((np_ra<1)||(np_rb<1),1,"read_power_parms [ms2.c]",
                 "Power method iteration numbers must be at least 1");
   }

   MPI_Bcast(&np_ra,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&np_rb,1,MPI_INT,0,MPI_COMM_WORLD);
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

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [ms2.c]",
                 "Syntax: ms2 -i <input file>");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [ms2.c]",
                 "Machine has unknown endianness");

      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [ms2.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);

   read_dirs();
   read_iodat("Configurations","i",iodat);
   read_cnfg_range();
   read_lat_parms("Dirac operator",0x2);
   read_bc_parms("Boundary conditions",0x2);
   read_power_parms();
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
      error_root(ie!=0,1,"check_files [ms2.c]",
                 "Attempt to overwrite old *.log file");
   }

   error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
         "check_files [ms2.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);
}


static void print_info(void)
{
   int isap,idfl;
   long ip;

   if (my_rank==0)
   {
      ip=ftell(flog);
      fclose(flog);

      if (ip==0L)
         remove("STARTUP_ERROR");

      flog=freopen(log_file,"w",stdout);
      error_root(flog==NULL,1,"print_info [ms2.c]","Unable to open log file");
      printf("\n");

      printf("Spectral range of the Hermitian Dirac operator\n");
      printf("----------------------------------------------\n\n");

      printf("Program version %s\n",openQCD_RELEASE);

      if (endian==LITTLE_ENDIAN)
         printf("The machine is little endian\n");
      else
         printf("The machine is big endian\n");

      print_lattice_sizes();
      print_lat_parms(0x2);
      print_bc_parms(0x2);

      printf("Power method:\n");
      printf("np_ra = %d\n",np_ra);
      printf("np_rb = %d\n\n",np_rb);

      print_solver_parms(&isap,&idfl);

      if (isap)
         print_sap_parms(0x0);

      if (idfl)
         print_dfl_parms(0x0);

      print_iodat("i",iodat);
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
      error_root(1,1,"solver_wsize [ms2.c]","Unknown or unsupported solver");
}


static void wsize(int *nws,int *nwv,int *nwvd)
{
   (*nws)=0;
   (*nwv)=0;
   (*nwvd)=0;

   solver_wsize(0,2,nws,nwv,nwvd);
}


static void init_stat(int *status)
{
   if (stat0==NULL)
   {
      stat0=alloc_std_status();
      stat1=alloc_std_status();
   }

   ifail0[0]=0;
   ifail0[1]=0;
   ifail1[0]=0;
   ifail1[1]=0;
   reset_std_status(stat0);
   reset_std_status(stat1);
   reset_std_status(status);
}


static void print_stat(int *status)
{
   solver_parms_t sp;

   sp=solver_parms(0);

   if (sp.solver==CGNE)
      print_std_status("tmcgeo",NULL,status);
   else if (sp.solver==SAP_GCR)
      print_std_status("sap_gcr","sap_gcr",status);
   else if (sp.solver==DFL_SAP_GCR)
      print_std_status("dfl_sap_gcr2","dfl_sap_gcr2",status);
}


static double power1(int *status)
{
   int k;
   double r;
   spinor_dble *phi,**wsd;
   solver_parms_t sp;
   sap_parms_t sap;

   init_stat(status);
   set_sw_parms(sea_quark_mass(0));
   sp=solver_parms(0);

   if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
   }

   wsd=reserve_wsd(1);
   phi=wsd[0];
   random_sd(VOLUME_TRD/2,2,phi,1.0);
   bnd_sd2zero(EVEN_PTS,phi);
   r=normalize_dble(VOLUME_TRD/2,3,phi);

   for (k=0;k<np_ra;k++)
   {
      if (sp.solver==CGNE)
      {
         tmcgeo(sp.nmx,sp.istop,sp.res,0.0,phi,phi,ifail0,stat0);
         acc_std_status("tmcgeo",ifail0,stat0,0,status);

         if (ifail0[0]<0)
         {
            print_status("tmcgeo",ifail0,stat0);
            error_root(1,1,"power1 [ms2.c]","CGNE solver failed "
                       "(mu = 0.0, parameter set no 0)");
         }
      }
      else if (sp.solver==SAP_GCR)
      {
         mulg5_dble(VOLUME_TRD/2,2,phi);
         set_sd2zero(VOLUME_TRD/2,2,phi+(VOLUME/2));
         sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,0.0,phi,phi,ifail0,stat0);
         mulg5_dble(VOLUME_TRD/2,2,phi);
         set_sd2zero(VOLUME_TRD/2,2,phi+(VOLUME/2));
         sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,0.0,phi,phi,ifail1,stat1);
         acc_std_status("sap_gcr",ifail0,stat0,0,status);
         acc_std_status("sap_gcr",ifail1,stat1,1,status);

         if ((ifail0[0]<0)||(ifail1[0]<0))
         {
            print_status("sap_gcr",ifail0,stat0);
            print_status("sap_gcr",ifail1,stat1);
            error_root(1,1,"power1 [ms2.c]","SAP_GCR solver failed "
                       "(mu = 0.0, parameter set no 0)");
         }
      }
      else if (sp.solver==DFL_SAP_GCR)
      {
         mulg5_dble(VOLUME_TRD/2,2,phi);
         set_sd2zero(VOLUME_TRD/2,2,phi+(VOLUME/2));
         dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,0.0,phi,phi,ifail0,stat0);
         mulg5_dble(VOLUME_TRD/2,2,phi);
         set_sd2zero(VOLUME_TRD/2,2,phi+(VOLUME/2));
         dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,0.0,phi,phi,ifail1,stat1);
         acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);
         acc_std_status("dfl_sap_gcr2",ifail1,stat1,1,status);

         if (((ifail0[0]<-2)||(ifail0[1]<0))||((ifail1[0]<-2)||(ifail1[1]<0)))
         {
            print_status("dfl_sap_gcr2",ifail0,stat0);
            print_status("dfl_sap_gcr2",ifail1,stat1);
            error_root(1,1,"power1 [ms2.c]","DFL_SAP_GCR solver failed "
                       "(mu = 0.0, parameter set no 0)");
         }
      }

      r=normalize_dble(VOLUME_TRD/2,3,phi);
   }

   avg_std_status(np_ra,status);
   release_wsd();

   return 1.0/sqrt(r);
}


static double power2(void)
{
   int k;
   double r;
   spinor_dble *phi,*psi,**wsd;

   set_sw_parms(sea_quark_mass(0));
   sw_term(ODD_PTS);

   wsd=reserve_wsd(2);
   phi=wsd[0];
   psi=wsd[1];

   random_sd(VOLUME_TRD/2,2,phi,1.0);
   bnd_sd2zero(EVEN_PTS,phi);
   r=normalize_dble(VOLUME_TRD/2,3,phi);

   for (k=0;k<np_rb;k++)
   {
      Dwhat_dble(0.0,phi,psi);
      mulg5_dble(VOLUME_TRD/2,2,psi);
      Dwhat_dble(0.0,psi,phi);
      mulg5_dble(VOLUME_TRD/2,2,phi);

      r=normalize_dble(VOLUME_TRD/2,3,phi);
   }

   release_wsd();

   return sqrt(r);
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
   int nws,nwv,nwvd,n,bc;
   double ra,ramin,ramax,raavg;
   double rb,rbmin,rbmax,rbavg;
   double A,eps,delta,Ne,d1,d2;
   double wt1,wt2,wtavg;
   dfl_parms_t dfl;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   check_machine();
   geometry();
   check_files();
   print_info();
   start_ranlux(0,1234);

   wsize(&nws,&nwv,&nwvd);
   alloc_ws(nws);
   wsd_uses_ws();
   alloc_wv(nwv);
   alloc_wvd(nwvd);
   status=alloc_std_status();

   dfl=dfl_parms();

   ramin=0.0;
   ramax=0.0;
   raavg=0.0;

   rbmin=0.0;
   rbmax=0.0;
   rbavg=0.0;

   iend=0;
   wtavg=0.0;

   for (nc=first;(iend==0)&&(nc<=last);nc+=step)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

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

         if ((ifail0[0]<-2)||(ifail0[1]<0))
         {
            print_status("dfl_modes2",ifail0,status);
            error_root(1,1,"main [ms2.c]",
                       "Deflation subspace generation failed");
         }
      }

      ra=power1(status);
      rb=power2();

      if (nc==first)
      {
         ramin=ra;
         ramax=ra;
         raavg=ra;

         rbmin=rb;
         rbmax=rb;
         rbavg=rb;
      }
      else
      {
         if (ra<ramin)
            ramin=ra;
         if (ra>ramax)
            ramax=ra;
         raavg+=ra;

         if (rb<rbmin)
            rbmin=rb;
         if (rb>rbmax)
            rbmax=rb;
         rbavg+=rb;
      }

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wtavg+=(wt2-wt1);

      if (my_rank==0)
      {
         printf("ra = %.2e, rb = %.2e, ",ra,rb);
         print_stat(status);

         printf("Configuration no %d fully processed in %.2e sec ",
                nc,wt2-wt1);
         printf("(average = %.2e sec)\n\n",
                wtavg/(double)((nc-first)/step+1));

         fflush(flog);
         copy_file(log_file,log_save);
      }

      check_endflag(&iend);
   }

   if (my_rank==0)
   {
      last=nc-step;
      nc=(last-first)/step+1;

      printf("Summary\n");
      printf("-------\n\n");

      printf("Considered %d configurations in the range %d -> %d\n\n",
             nc,first,last);

      printf("The three figures quoted in each case are the minimal,\n");
      printf("maximal and average values\n\n");

      printf("Spectral gap ra    = %.2e, %.2e, %.2e\n",
             ramin,ramax,raavg/(double)(nc));
      printf("Spectral radius rb = %.2e, %.2e, %.2e\n\n",
             rbmin,rbmax,rbavg/(double)(nc));

      ra=0.90*ramin;
      rb=1.10*rbmax;
      eps=ra/rb;
      eps=eps*eps;

      bc=bc_type();
      Ne=0.5*(double)(N1*N2*N3);

      if (bc==0)
         Ne*=(double)(N0-2);
      else if ((bc==1)||(bc==2))
         Ne*=(double)(N0-1);
      else
         Ne*=(double)(N0);

      printf("Zolotarev rational approximation:\n\n");

      printf("n: number of poles\n");
      printf("delta: approximation error\n");
      printf("Ne: number of even lattice points\n");
      printf("Suggested spectral range = [%.2e,%.2e]\n\n",ra,rb);

      printf("     n      delta    12*Ne*delta     12*Ne*delta^2\n");

      for (n=6;n<=128;n++)
      {
         zolotarev(n,eps,&A,ar,&delta);
         d1=12.0*Ne*delta;
         d2=d1*delta;

         printf("   %3d     %.1e      %.1e         %.1e\n",n,delta,d1,d2);

         if ((d1<1.0e-2)&&(d2<1.0e-4))
            break;
      }

      printf("\n");
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
