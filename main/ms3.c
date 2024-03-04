
/*******************************************************************************
*
* File ms3.c
*
* Copyright (C) 2012, 2013, 2017, 2019, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of Wilson (gradient) flow observables.
*
* Syntax: ms3 -i <input file> [-a]
*
* For usage instructions see the file README.ms3.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "flags.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "archive.h"
#include "tcharge.h"
#include "wflow.h"
#include "version.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static struct
{
   int dn,nn,tmax;
   double eps;
} file_head;

static struct
{
   int nc;
   double **Wsl,**Ysl,**Qsl;
} data;

static int my_rank,append,endian;
static int first,last,step;
static int ipgrd[3];
static double *Wact,*Yact,*Qtop;

iodat_t iodat[1];
static char nbase[NAME_SIZE];
static char log_dir[NAME_SIZE],dat_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE];
static char dat_file[NAME_SIZE],dat_save[NAME_SIZE];
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE],end_file[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fdat=NULL,*fend=NULL;


static void set_file_head(void)
{
   wflow_parms_t wfl;

   wfl=wflow_parms();

   file_head.dn=wfl.dnms;
   file_head.nn=wfl.ntot/wfl.dnms;
   file_head.tmax=N0;
   file_head.eps=wfl.eps;
}


static void alloc_data(void)
{
   int nn,tmax;
   int in,k;
   double **pp,*p;

   nn=file_head.nn;
   tmax=file_head.tmax;

   pp=amalloc(3*(nn+1)*sizeof(*pp),3);
   p=amalloc(3*(nn+1)*(tmax+1)*sizeof(*p),4);
   error((pp==NULL)||(p==NULL),1,"alloc_data [ms3.c]",
         "Unable to allocate data arrays");
   for (k=0;k<(3*(nn+1)*(tmax+1));k++)
      p[k]=0.0;

   data.Wsl=pp;
   data.Ysl=pp+nn+1;
   data.Qsl=pp+2*(nn+1);

   for (in=0;in<(3*(nn+1));in++)
   {
      *pp=p;
      pp+=1;
      p+=tmax;
   }

   Wact=p;
   p+=nn+1;
   Yact=p;
   p+=nn+1;
   Qtop=p;
}


static void write_file_head(void)
{
   int iw;
   stdint_t istd[3];
   double dstd[1];

   istd[0]=(stdint_t)(file_head.dn);
   istd[1]=(stdint_t)(file_head.nn);
   istd[2]=(stdint_t)(file_head.tmax);
   dstd[0]=file_head.eps;

   if (endian==BIG_ENDIAN)
   {
      bswap_int(3,istd);
      bswap_double(1,dstd);
   }

   iw=fwrite(istd,sizeof(stdint_t),3,fdat);
   iw+=fwrite(dstd,sizeof(double),1,fdat);

   error_root(iw!=4,1,"write_file_head [ms3.c]",
              "Incorrect write count");
}


static void check_file_head(void)
{
   int ir;
   stdint_t istd[3];
   double dstd[1];

   ir=fread(istd,sizeof(stdint_t),3,fdat);
   ir+=fread(dstd,sizeof(double),1,fdat);

   error_root(ir!=4,1,"check_file_head [ms3.c]",
              "Incorrect read count");

   if (endian==BIG_ENDIAN)
   {
      bswap_int(3,istd);
      bswap_double(1,dstd);
   }

   error_root(((int)(istd[0])!=file_head.dn)||
              ((int)(istd[1])!=file_head.nn)||
              ((int)(istd[2])!=file_head.tmax)||
              (dstd[0]!=file_head.eps),1,"check_file_head [ms3.c]",
              "Unexpected value of dn,nn,tmax or eps");
}


static void write_data(void)
{
   int iw,nn,tmax;
   int k,in,t;
   stdint_t istd[1];
   double **Xsl[3],dstd[1];

   istd[0]=(stdint_t)(data.nc);

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   iw=fwrite(istd,sizeof(stdint_t),1,fdat);

   Xsl[0]=data.Wsl;
   Xsl[1]=data.Ysl;
   Xsl[2]=data.Qsl;
   nn=file_head.nn;
   tmax=file_head.tmax;

   for (k=0;k<3;k++)
   {
      for (in=0;in<=nn;in++)
      {
         for (t=0;t<tmax;t++)
         {
            dstd[0]=Xsl[k][in][t];

            if (endian==BIG_ENDIAN)
               bswap_double(1,dstd);

            iw+=fwrite(dstd,sizeof(double),1,fdat);
         }
      }
   }

   error_root(iw!=(1+3*(nn+1)*tmax),1,"write_data [ms3.c]",
              "Incorrect write count");
}


static int read_data(void)
{
   int ir,nn,tmax;
   int k,in,t;
   stdint_t istd[1];
   double **Xsl[3],dstd[1];

   ir=fread(istd,sizeof(stdint_t),1,fdat);

   if (ir!=1)
      return 0;

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   data.nc=(int)(istd[0]);

   Xsl[0]=data.Wsl;
   Xsl[1]=data.Ysl;
   Xsl[2]=data.Qsl;
   nn=file_head.nn;
   tmax=file_head.tmax;

   for (k=0;k<3;k++)
   {
      for (in=0;in<=nn;in++)
      {
         for (t=0;t<tmax;t++)
         {
            ir+=fread(dstd,sizeof(double),1,fdat);

            if (endian==BIG_ENDIAN)
               bswap_double(1,dstd);

            Xsl[k][in][t]=dstd[0];
         }
      }
   }

   error_root(ir!=(1+3*(nn+1)*tmax),1,"read_data [ms3.c]",
              "Read error or incomplete data record");

   return 1;
}


static void read_dirs(void)
{
   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);

      find_section("Log and data directories");
      read_line("log_dir","%s",log_dir);
      read_line("dat_dir","%s",dat_dir);
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(dat_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
}


static void setup_files(void)
{
   error(name_size("%s/%s.ms3.log~",log_dir,nbase)>=NAME_SIZE,1,
         "setup_files [ms3.c]","log_dir name is too long");
   error(name_size("%s/%s.ms3.dat~",dat_dir,nbase)>=NAME_SIZE,1,
         "setup_files [ms3.c]","dat_dir name is too long");

   sprintf(log_file,"%s/%s.ms3.log",log_dir,nbase);
   sprintf(par_file,"%s/%s.ms3.par",dat_dir,nbase);
   sprintf(dat_file,"%s/%s.ms3.dat",dat_dir,nbase);
   sprintf(end_file,"%s/%s.ms3.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);
   sprintf(dat_save,"%s~",dat_file);

   check_dir_root(log_dir);
   check_dir_root(dat_dir);
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
                 "read_cnfg_range [ms3.c]","Improper configuration range");
   }

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void read_infile(int argc,char *argv[])
{
   int ifile;

   if (my_rank==0)
   {
      flog=freopen("STARTUP_ERROR","w",stdout);

      ifile=find_opt(argc,argv,"-i");
      endian=endianness();

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [ms3.c]",
                 "Syntax: ms3 -i <input file> [-a]");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [ms3.c]",
                 "Machine has unknown endianness");

      append=find_opt(argc,argv,"-a");

      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [ms3.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&append,1,MPI_INT,0,MPI_COMM_WORLD);

   read_dirs();
   read_iodat("Configurations","i",iodat);
   read_cnfg_range();
   read_bc_parms("Boundary conditions",0x0);
   read_wflow_parms("Wilson flow",0x1);

   setup_files();

   if (my_rank==0)
   {
      fdat=fopen(par_file,"rb");

      if (append==0)
      {
         error_root(fdat!=NULL,1,"read_infile [ms3.c]",
                    "Attempt to overwrite old parameter file");
         fdat=fopen(par_file,"wb");
      }

      error_root(fdat==NULL,1,"read_infile [ms3.c]",
                 "Unable to open parameter file");
   }

   if (append)
   {
      check_bc_parms(fdat);
      check_wflow_parms(fdat);
   }
   else
   {
      write_bc_parms(fdat);
      write_wflow_parms(fdat);
   }

   if (my_rank==0)
   {
      fclose(fin);
      fclose(fdat);

      if (append==0)
         copy_file(par_file,par_save);
   }
}


static void check_old_log(int *fst,int *lst,int *stp)
{
   int ie,ic,isv;
   int fc,lc,dc,pc;
   int nt,np[4],bp[4];
   char line[NAME_SIZE];

   fend=fopen(log_file,"r");
   error_root(fend==NULL,1,"check_old_log [ms3.c]",
              "Unable to open log file");

   fc=0;
   lc=0;
   dc=0;
   pc=0;

   ie=0x0;
   ic=0;
   isv=0;

   while (fgets(line,NAME_SIZE,fend)!=NULL)
   {
      if (strstr(line,"MPI process grid")!=NULL)
      {
         if (sscanf(line,"%dx%dx%dx%d MPI process grid, %dx%dx%dx%d",
                    np,np+1,np+2,np+3,bp,bp+1,bp+2,bp+3)==8)
         {
            ipgrd[0]=((np[0]!=NPROC0)||(np[1]!=NPROC1)||
                      (np[2]!=NPROC2)||(np[3]!=NPROC3));
            ipgrd[1]=((bp[0]!=NPROC0_BLK)||(bp[1]!=NPROC1_BLK)||
                      (bp[2]!=NPROC2_BLK)||(bp[3]!=NPROC3_BLK));
         }
         else
            ie|=0x1;
      }
      else if (strstr(line,"OpenMP thread")!=NULL)
      {
         if (sscanf(line,"%d OpenMP thread",&nt)==1)
            ipgrd[2]=(nt!=NTHREAD);
         else
            ie|=0x1;
      }
      else if (strstr(line,"fully processed")!=NULL)
      {
         pc=lc;

         if (sscanf(line,"Configuration no %d",&lc)==1)
         {
            ic+=1;
            isv=1;
         }
         else
            ie|=0x1;

         if (ic==1)
            fc=lc;
         else if (ic==2)
            dc=lc-fc;
         else if ((ic>2)&&(lc!=(pc+dc)))
            ie|=0x2;
      }
      else if (strstr(line,"Configuration no")!=NULL)
         isv=0;
   }

   fclose(fend);

   error_root((ie&0x1)!=0x0,1,"check_old_log [ms3.c]",
              "Incorrect read count");
   error_root((ie&0x2)!=0x0,1,"check_old_log [ms3.c]",
              "Configuration numbers are not equally spaced");
   error_root(isv==0,1,"check_old_log [ms3.c]",
              "Log file extends beyond the last configuration save");

   (*fst)=fc;
   (*lst)=lc;
   (*stp)=dc;
}


static void check_old_dat(int fst,int lst,int stp)
{
   int ie,ic;
   int fc,lc,dc,pc;

   fdat=fopen(dat_file,"rb");
   error_root(fdat==NULL,1,"check_old_dat [ms3.c]",
              "Unable to open data file");

   check_file_head();

   fc=0;
   lc=0;
   dc=0;
   pc=0;

   ie=0x0;
   ic=0;

   while (read_data()==1)
   {
      pc=lc;
      lc=data.nc;
      ic+=1;

      if (ic==1)
         fc=lc;
      else if (ic==2)
         dc=lc-fc;
      else if ((ic>2)&&(lc!=(pc+dc)))
         ie|=0x1;
   }

   fclose(fdat);

   error_root(ic==0,1,"check_old_dat [ms3.c]",
              "No data records found");
   error_root((ie&0x1)!=0x0,1,"check_old_dat [ms3.c]",
              "Configuration numbers are not equally spaced");
   error_root((fst!=fc)||(lst!=lc)||(stp!=dc),1,"check_old_dat [ms3.c]",
              "Configuration range is not as reported in the log file");
}


static void check_files(void)
{
   int ie;
   int fst,lst,stp;

   ipgrd[0]=0;
   ipgrd[1]=0;
   ipgrd[2]=0;

   if (my_rank==0)
   {
      if (append)
      {
         check_old_log(&fst,&lst,&stp);
         check_old_dat(fst,lst,stp);

         error_root((fst!=lst)&&(stp!=step),1,"check_files [ms3.c]",
                    "Continuation run:\n"
                    "Previous run had a different configuration separation");
         error_root(first!=lst+step,1,"check_files [ms3.c]",
                    "Continuation run:\n"
                    "Configuration range does not continue the previous one");
      }
      else
      {
         ie=check_file(log_file,"r");
         ie|=check_file(dat_file,"rb");

         error_root(ie!=0,1,"check_files [ms3.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         fdat=fopen(dat_file,"wb");
         error_root(fdat==NULL,1,"check_files [ms3.c]",
                    "Unable to open data file");
         write_file_head();
         fclose(fdat);
      }
   }

   error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
         "check_files [ms3.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);
}


static void print_info(void)
{
   long ip;

   if (my_rank==0)
   {
      ip=ftell(flog);
      fclose(flog);

      if (ip==0L)
         remove("STARTUP_ERROR");

      if (append)
         flog=freopen(log_file,"a",stdout);
      else
         flog=freopen(log_file,"w",stdout);

      error_root(flog==NULL,1,"print_info [ms3.c]","Unable to open log file");
      printf("\n");

      if (append)
         printf("Continuation run\n\n");
      else
      {
         printf("Computation of Wilson flow observables\n");
         printf("--------------------------------------\n\n");
      }

      printf("Program version %s\n",openQCD_RELEASE);

      if (endian==LITTLE_ENDIAN)
         printf("The machine is little endian\n");
      else
         printf("The machine is big endian\n");

      if ((ipgrd[0])||(ipgrd[1])||(ipgrd[2]))
         printf("MPI process grid, process blocks or OpenMP thread no "
                "changed:\n");

      if ((append==0)||(ipgrd[0])||(ipgrd[1])||(ipgrd[2]))
         print_lattice_sizes();

      if (append==0)
      {
         print_bc_parms(0x0);
         print_wflow_parms();
      }

      print_iodat("i",iodat);
      printf("Configurations no %d -> %d in steps of %d\n\n",
             first,last,step);
      fflush(flog);
   }
}


static void integrate_wflow(double t0,double t1,double eps)
{
   int ns,rule;
   double dt,del;
   wflow_parms_t wfl;

   wfl=wflow_parms();
   rule=wfl.rule;

   if (t0<1.0)
   {
      if (t1>1.0)
         dt=1.0-t0;
      else
         dt=t1-t0;

      if (rule==0)
         del=eps*0.1;
      else if (rule==1)
         del=eps*0.31623;
      else
         del=eps*0.46416;
   }
   else
   {
      dt=t1-t0;
      del=eps;
   }

   ns=(int)(dt/del);
   dt=dt-(double)(ns)*del;

   if (rule==0)
   {
      if (ns>0)
         fwd_euler(ns,del);
      if (dt>0.0)
         fwd_euler(1,dt);
   }
   else if (rule==1)
   {
      if (ns>0)
         fwd_rk2(ns,del);
      if (dt>0.0)
         fwd_rk2(1,dt);
   }
   else
   {
      if (ns>0)
         fwd_rk3(ns,del);
      if (dt>0.0)
         fwd_rk3(1,dt);
   }

   if ((t0<1.0)&&(t1>1.0))
      integrate_wflow(1.0,t1,eps);
}


static void set_data(int nc)
{
   int in,dn,nn;
   double eps,dt,t;

   data.nc=nc;
   dn=file_head.dn;
   nn=file_head.nn;
   eps=file_head.eps;
   dt=(double)(dn)*eps;
   t=0.0;

   for (in=0;in<nn;in++)
   {
      Wact[in]=plaq_action_slices(data.Wsl[in]);
      Yact[in]=ym_action_slices(data.Ysl[in]);
      Qtop[in]=tcharge_slices(data.Qsl[in]);

      integrate_wflow(t,t+dt,eps);
      t+=dt;
   }

   Wact[in]=plaq_action_slices(data.Wsl[in]);
   Yact[in]=ym_action_slices(data.Ysl[in]);
   Qtop[in]=tcharge_slices(data.Qsl[in]);
}


static void save_data(void)
{
   if (my_rank==0)
   {
      fdat=fopen(dat_file,"ab");
      error_root(fdat==NULL,1,"save_data [ms3.c]",
                 "Unable to open data file");
      write_data();
      fclose(fdat);
   }
}


static void print_log(void)
{
   int in,dn,nn,din;
   double eps;

   if (my_rank==0)
   {
      dn=file_head.dn;
      nn=file_head.nn;
      eps=file_head.eps;

      din=nn/10;
      if (din<1)
         din=1;

      for (in=0;in<=nn;in+=din)
         printf("n = %3d, t = %.2e, Wact = %.6e, Yact = %.6e, Q = % .2e\n",
                in*dn,eps*(double)(in*dn),Wact[in],Yact[in],Qtop[in]);
   }
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
   int nc,iend;
   double wt1,wt2,wtavg;
   wflow_parms_t wfl;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   check_machine();
   set_file_head();
   alloc_data();
   geometry();
   check_files();
   print_info();

   wfl=wflow_parms();
   if (wfl.rule)
      alloc_wfd(1);

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
      set_data(nc);
      save_data();
      print_log();

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wtavg+=(wt2-wt1);

      if (my_rank==0)
      {
         printf("Configuration no %d fully processed in %.2e sec ",
                nc,wt2-wt1);
         printf("(average = %.2e sec)\n\n",
                wtavg/(double)((nc-first)/step+1));
         fflush(flog);

         copy_file(log_file,log_save);
         copy_file(dat_file,dat_save);
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
