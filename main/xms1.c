
/*******************************************************************************
*
* File xms1.c
*
* Copyright (C) 2017-2019, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the average Yang-Mills action density (XXL lattices).
*
* Syntax: xms1 -i <filename> [[-noflds]|[-a]] [-nosm]
*
* For usage instructions see the file README.xms1.
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
#include "mdflds.h"
#include "archive.h"
#include "msfcts.h"
#include "tcharge.h"
#include "wflow.h"
#include "version.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

typedef union
{
   su3_alg_dble X;
   complex_dble c[4];
   double r[8];
} flds_t;

static struct
{
   int ntm,dmax;
   double *tm;
} file_head;

static struct
{
   int nc;
   double *Q;
   double *E,***var;
} data;

static int my_rank,noflds,nosm,append,endian;
static int start,first,last,step,ipsm,*nacc;
static int ipgrd[3],i3d;
static double Qmax,zero[3]={0.0,0.0,0.0};

static double **f;
static complex_dble **rf;
static u3_alg_dble **ft;

static iodat_t iodat[2];
static char nbase[NAME_SIZE];
static char log_dir[NAME_SIZE],dat_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE];
static char dat0_file[NAME_SIZE],dat0_save[NAME_SIZE];
static char dat1_file[NAME_SIZE],dat1_save[NAME_SIZE];
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE],end_file[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fdat=NULL,*fend=NULL;


static void write_file_head(int idat)
{
   int ntm,i,iw;
   stdint_t istd[2];
   double dstd[1];

   if (idat==0)
      fdat=fopen(dat0_file,"wb");
   else
      fdat=fopen(dat1_file,"wb");
   error_root(fdat==NULL,1,"write_file_head [xms1.c]",
              "Unable to open data file");

   istd[0]=(stdint_t)(file_head.ntm);
   istd[1]=(stdint_t)(file_head.dmax);

   if (endian==BIG_ENDIAN)
      bswap_int(2,istd);

   iw=fwrite(istd,sizeof(stdint_t),2,fdat);
   ntm=file_head.ntm;

   for (i=0;i<ntm;i++)
   {
      dstd[0]=file_head.tm[i];

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   error_root(iw!=(2+ntm),1,"write_file_head [xms1.c]",
              "Incorrect write count");
   fclose(fdat);
}


static void check_file_head(void)
{
   int ntm,i,ir,ie;
   stdint_t istd[2];
   double dstd[1];

   ir=fread(istd,sizeof(stdint_t),2,fdat);

   if (endian==BIG_ENDIAN)
      bswap_int(2,istd);

   ie=0;
   ie|=((int)(istd[0])!=file_head.ntm);
   ie|=((int)(istd[1])!=file_head.dmax);

   error_root(ir!=2,1,"check_file_head [xms1.c]",
              "Incorrect read count");
   error_root(ie!=0,1,"check_file_head [xms1.c]",
              "Unexpected data file header data");

   ntm=file_head.ntm;

   for (i=0;i<ntm;i++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      ie|=(dstd[0]!=file_head.tm[i]);
   }

   error_root(ir!=(2+ntm),1,"check_file_head [xms1.c]",
              "Incorrect read count");
   error_root(ie!=0,1,"check_file_head [xms1.c]",
              "Unexpected data file header data");
}


static void alloc_data(void)
{
   int ntm,dmax,i;
   double *p1,**p2,***p3;

   ntm=file_head.ntm;
   dmax=file_head.dmax+1;

   p3=malloc(ntm*sizeof(*p3));
   p2=malloc(ntm*ntm*sizeof(*p2));
   p1=malloc((ntm*ntm*dmax+2*ntm)*sizeof(*p1));
   error((p1==NULL)||(p2==NULL)||(p3==NULL),1,"alloc_data [xms1.c]",
         "Unable to allocate data arrays");
   for (i=0;i<(ntm*ntm*dmax+2*ntm);i++)
      p1[i]=0.0;

   data.Q=p1;
   data.E=p1+ntm;
   data.var=p3;
   p1+=2*ntm;

   for (i=0;i<(ntm*ntm);i++)
   {
      p2[i]=p1;
      p1+=dmax;
   }

   for (i=0;i<ntm;i++)
   {
      p3[i]=p2;
      p2+=ntm;
   }
}


static void write_data(int idat)
{
   int ntm,dmax,iw;
   int i,j,k;
   stdint_t istd[1];
   double dstd[1];

   istd[0]=(stdint_t)(data.nc);

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   iw=fwrite(istd,sizeof(stdint_t),1,fdat);

   ntm=file_head.ntm;
   dmax=file_head.dmax+1;

   if (idat==0)
   {
      for (i=0;i<ntm;i++)
      {
         dstd[0]=data.Q[i];

         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);

         iw+=fwrite(dstd,sizeof(double),1,fdat);
      }
   }
   else
   {
      for (i=0;i<ntm;i++)
      {
         istd[0]=(stdint_t)(nacc[i]);

         if (endian==BIG_ENDIAN)
            bswap_int(1,istd);

         iw+=fwrite(istd,sizeof(stdint_t),1,fdat);
      }
   }

   for (i=0;i<ntm;i++)
   {
      dstd[0]=data.E[i];

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      iw+=fwrite(dstd,sizeof(double),1,fdat);
   }

   for (i=0;i<ntm;i++)
   {
      for (j=0;j<ntm;j++)
      {
         for (k=0;k<dmax;k++)
         {
            dstd[0]=data.var[i][j][k];

            if (endian==BIG_ENDIAN)
               bswap_double(1,dstd);

            iw+=fwrite(dstd,sizeof(double),1,fdat);
         }
      }
   }

   error_root(iw!=(1+2*ntm+ntm*ntm*dmax),1,"write_data [xms1.c]",
              "Incorrect write count");
}


static int read_data(void)
{
   int ntm,dmax,ir;
   int i,j,k;
   stdint_t istd[1];
   double dstd[1];

   ir=fread(istd,sizeof(stdint_t),1,fdat);

   if (ir!=1)
      return 0;

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   data.nc=(int)(istd[0]);

   ntm=file_head.ntm;
   dmax=file_head.dmax+1;

   for (i=0;i<ntm;i++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      data.Q[i]=dstd[0];
   }

   for (i=0;i<ntm;i++)
   {
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_double(1,dstd);

      data.E[i]=dstd[0];
   }

   for (i=0;i<ntm;i++)
   {
      for (j=0;j<ntm;j++)
      {
         for (k=0;k<dmax;k++)
         {
            ir+=fread(dstd,sizeof(double),1,fdat);

            if (endian==BIG_ENDIAN)
               bswap_double(1,dstd);

            data.var[i][j][k]=dstd[0];
         }
      }
   }

   error_root(ir!=(1+2*ntm+ntm*ntm*dmax),1,"read_data [xms1.c]",
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
   error(name_size("%s/%s.xms1.log~",log_dir,nbase)>=NAME_SIZE,1,
         "setup_files [xms1.c]","log_dir name is too long");
   error(name_size("%s/%s.xms1.0.dat~",dat_dir,nbase)>=NAME_SIZE,1,
         "setup_files [xms1.c]","dat_dir name is too long");

   sprintf(log_file,"%s/%s.xms1.log",log_dir,nbase);
   sprintf(par_file,"%s/%s.xms1.par",dat_dir,nbase);
   sprintf(dat0_file,"%s/%s.xms1.0.dat",dat_dir,nbase);
   sprintf(dat1_file,"%s/%s.xms1.1.dat",dat_dir,nbase);
   sprintf(end_file,"%s/%s.xms1.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);
   sprintf(dat0_save,"%s~",dat0_file);
   sprintf(dat1_save,"%s~",dat1_file);

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
                 "read_cnfg_range [xms1.c]","Improper configuration range");
   }

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void read_obs_parms(void)
{
   int dmax;
   wflow_parms_t wfl;

   if (my_rank==0)
   {
      find_section("Observables");
      read_line("i3d","%d",&i3d);
      read_line("dmax","%d",&dmax);
      read_line("Qmax","%lf",&Qmax);
   }

   MPI_Bcast(&i3d,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&dmax,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Qmax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   error_root((i3d<0)||(i3d>1)||(dmax<0)||(Qmax<=0.0),1,
              "read_obs_parms [xms1.c]","Parameters are out of range");

   wfl=wflow_parms();
   file_head.dmax=dmax;
   file_head.ntm=wfl.ntm;
   file_head.tm=wfl.tm;
}


static void write_obs_parms(void)
{
   int iw;
   stdint_t istd[2];
   double dstd[1];

   if (my_rank==0)
   {
      istd[0]=(stdint_t)(i3d);
      istd[1]=(stdint_t)(file_head.dmax);
      dstd[0]=Qmax;

      if (endian==BIG_ENDIAN)
      {
         bswap_int(2,istd);
         bswap_double(1,dstd);
      }

      iw=fwrite(istd,sizeof(stdint_t),2,fdat);
      iw+=fwrite(dstd,sizeof(double),1,fdat);

      error_root(iw!=3,1,"write_obs_parms [xms1.c]",
                 "Incorrect write count");
   }
}


static void check_obs_parms(void)
{
   int ir,ie;
   stdint_t istd[2];
   double dstd[1];

   if (my_rank==0)
   {
      ir=fread(istd,sizeof(stdint_t),2,fdat);
      ir+=fread(dstd,sizeof(double),1,fdat);

      if (endian==BIG_ENDIAN)
      {
         bswap_int(2,istd);
         bswap_double(1,dstd);
      }

      ie=0;
      ie|=(istd[0]!=(stdint_t)(i3d));
      ie|=(istd[1]!=(stdint_t)(file_head.dmax));
      ie|=(dstd[0]!=Qmax);

      error_root(ir!=3,1,"read_obs_parms [xms1.c]",
                 "Incorrect read count");
      error_root(ie!=0,1,"read_obs_parms [xms1.c]",
                 "Parameters do not match previous run");
   }
}


static void read_infile(int argc,char *argv[])
{
   int ifile;

   if (my_rank==0)
   {
      flog=freopen("STARTUP_ERROR","w",stdout);

      ifile=find_opt(argc,argv,"-i");
      endian=endianness();

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [xms1.c]",
                 "Syntax: xms1 -i <filename> [[-noflds]|[-a]] [-nosm]");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [xms1.c]",
                 "Machine has unknown endianness");

      noflds=find_opt(argc,argv,"-noflds");
      append=find_opt(argc,argv,"-a");
      nosm=find_opt(argc,argv,"-nosm");

      error_root((noflds!=0)&&(append!=0),1,"read_infile [xms1.c]",
                 "The options -noflds and -a are mutually exclusive");

      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [xms1.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&noflds,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&append,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nosm,1,MPI_INT,0,MPI_COMM_WORLD);

   read_dirs();
   read_iodat("Configurations","i",iodat);
   read_cnfg_range();
   read_iodat("Observable fields","o",iodat+1);
   read_wflow_parms("Wilson flow",0x2);
   read_obs_parms();

   setup_files();

   if (my_rank==0)
   {
      fdat=fopen(par_file,"rb");

      if (append==0)
      {
         error_root(fdat!=NULL,1,"read_infile [xms1.c]",
                    "Attempt to overwrite old parameter file");
         fdat=fopen(par_file,"wb");
      }

      error_root(fdat==NULL,1,"read_infile [xms1.c]",
                 "Unable to open parameter file");
   }

   if (append)
   {
      check_iodat_parms(fdat,iodat+1);
      check_wflow_parms(fdat);
      check_obs_parms();
   }
   else
   {
      write_iodat_parms(fdat,iodat+1);
      write_wflow_parms(fdat);
      write_obs_parms();
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
   int i,p,nt,np[4],bp[4];
   char line[NAME_SIZE],*s;

   fend=fopen(log_file,"r");
   error_root(fend==NULL,1,"check_old_log [xms1.c]",
              "Unable to open log file");

   fc=0;
   lc=0;
   dc=0;
   pc=0;

   ie=0x0;
   ic=0;
   isv=0;

   while ((ie==0x0)&&(fgets(line,NAME_SIZE,fend)!=NULL))
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
      else if (strstr(line,"Configuration no")!=NULL)
      {
         isv=0;
         pc=lc;

         if (sscanf(line,"Configuration no %d",&lc)==1)
         {
            ic+=1;

            if (ic==1)
               fc=lc;
            else if (ic==2)
               dc=lc-fc;
            else if ((ic>2)&&(lc!=(pc+dc)))
               ie|=0x2;
         }
         else
            ie|=0x1;

      }
      else if (strstr(line,"Current sample sizes:")!=NULL)
      {
         s=line+strlen("Current sample sizes:");

         for (i=0;(ie==0x0)&&(i<file_head.ntm);i++)
         {
            ie|=(sscanf(s,"%d%n",nacc+i,&p)<=0);
            s+=p;
         }
      }
      else if (strstr(line,"Configuration fully processed")!=NULL)
         isv=1;
   }

   fclose(fend);

   error_root((ie&0x1)!=0x0,1,"check_old_log [xms1.c]",
              "Incorrect read count");
   error_root((ie&0x2)!=0x0,1,"check_old_log [xms1.c]",
              "Configuration numbers are not equally spaced");
   error_root(fc==0,1,"check_old_log [xms1.c]",
              "No configurations processed in previous run");
   error_root(isv==0,1,"check_old_log [xms1.c]",
              "Log file extends beyond the last completed measurement cycle");

   (*fst)=fc;
   (*lst)=lc;
   (*stp)=dc;
}


static void check_old_dat(int fst,int lst,int stp)
{
   int ie,ic;
   int fc,lc,dc,pc;

   fdat=fopen(dat0_file,"rb");
   error_root(fdat==NULL,1,"check_old_dat [xms1.c]",
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

   error_root(ic==0,1,"check_old_dat [xms1.c]",
              "No data records found");
   error_root((ie&0x1)!=0x0,1,"check_old_dat [xms1.c]",
              "Configuration numbers are not equally spaced");
   error_root((fst!=fc)||(lst!=lc)||(stp!=dc),1,"check_old_dat [xms1.c]",
              "Configuration range is not as reported in the log file");
}


static void check_files(void)
{
   int i,ie,ntm;
   int fst,lst,stp;

   ipgrd[0]=0;
   ipgrd[1]=0;
   ipgrd[2]=0;

   ntm=file_head.ntm;
   nacc=malloc(ntm*sizeof(*nacc));
   error(nacc==NULL,1,"check_files [xms1.c]",
         "Unable to allocate auxiliary array");

   for (i=0;i<ntm;i++)
      nacc[i]=0;

   if (my_rank==0)
   {
      if (append)
      {
         check_old_log(&fst,&lst,&stp);
         check_old_dat(fst,lst,stp);

         error_root((fst!=lst)&&(stp!=step),1,"check_files [xms1.c]",
                    "Continuation run:\n"
                    "Previous run had a different configuration separation");
         error_root((first>(lst+step))||(last<lst),1,"check_files [xms1.c]",
                    "Continuation run:\n"
                    "Configuration range does not continue the previous one");
      }
      else
      {
         fst=first;
         lst=last;
         stp=step;

         ie=check_file(log_file,"r");
         ie|=check_file(dat0_file,"rb");

         error_root(ie!=0,1,"check_files [xms1.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         write_file_head(0);
      }
   }

   MPI_Bcast(&fst,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&lst,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&stp,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(nacc,ntm,MPI_INT,0,MPI_COMM_WORLD);

   if ((append)&&(last==lst))
   {
      start=fst;
      ipsm=1;
   }
   else if (append)
   {
      start=fst;
      first=lst+step;
      ipsm=0;
   }
   else
   {
      start=first;
      ipsm=0;
   }

   if (ipsm==0)
   {
      error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
            "check_files [xms1.c]","Configuration base name is too long");
      sprintf(cnfg_file,"%sn%d",nbase,last);
      check_iodat(iodat,"i",0x1,cnfg_file);
   }

   if (noflds==0)
   {
      if (ipsm==0)
      {
         error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
               "check_files [xms1.c]","Configuration base name is too long");
         sprintf(cnfg_file,"%sn%d",nbase,last);
         check_iodat(iodat+1,"o",0x8,cnfg_file);
      }

      if (append)
      {
         if (ipsm==0)
         {
            error(name_size("%sn%d",nbase,first-step)>=NAME_SIZE,1,
                  "check_files [xms1.c]","Configuration base name is too long");
            sprintf(cnfg_file,"%sn%d",nbase,first-step);
         }
         else
         {
            error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
                  "check_files [xms1.c]","Configuration base name is too long");
            sprintf(cnfg_file,"%sn%d",nbase,last);
         }

         check_iodat(iodat+1,"i",0x8,cnfg_file);
      }
   }
}


static void print_info(void)
{
   int n;
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

      error_root(flog==NULL,1,"print_info [xms1.c]","Unable to open log file");
      printf("\n");

      if (append)
      {
         if (ipsm)
            printf("Pure sample measurement run\n\n");
         else
            printf("Continuation run\n\n");
      }
      else
      {
         printf("Computation of <E> (XXL lattices)\n");
         printf("---------------------------------\n\n");
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
      else
         printf("\n");

      if (append==0)
      {
         print_bc_parms(0x0);
         print_wflow_parms();

         printf("Observable parameters:\n");
         printf("i3d = %d\n",i3d);
         printf("dmax = %d\n",file_head.dmax);
         n=fdigits(Qmax);
         printf("Qmax = %.*f\n\n",IMAX(n,1),Qmax);
      }

      if (ipsm==0)
      {
         print_iodat("i",iodat);
         if (noflds==0)
            print_iodat("o",iodat+1);

         printf("Configurations no %d -> %d in steps of %d",
                first,last,step);
         if ((append)&&(nosm==0))
            printf(" (full range starts from no %d)\n\n",start);
         else
            printf("\n\n");
      }
      else
      {
         print_iodat("i",iodat+1);

         printf("Configuration range %d -> %d in steps of %d\n\n",
                start,last,step);
      }

      fflush(flog);
   }
}


static void alloc_flds(int nf,int nrf)
{
   int i,j,nf0,nrf0;
   double *pf;
   complex_dble *prf;
   mdflds_t *mdfs;
   flds_t *fld,*flm;

   if (nf>0)
      f=malloc(nf*sizeof(*f));
   else
      f=NULL;
   if (nrf>0)
      rf=malloc(nrf*sizeof(*rf));
   else
      rf=NULL;

   error(((nf>0)&&(f==NULL))||((nrf>0)&&(rf==NULL)),1,
         "alloc_flds [xms1.c]","Unable to allocate fields");

   mdfs=mdflds();
   fld=(flds_t*)((*mdfs).mom);
   flm=fld+4*VOLUME;
   nf0=0;
   nrf0=0;

   for (i=0;(i<nrf)&&(fld<flm);i++)
   {
      rf[i]=(complex_dble*)(fld);
      fld+=(VOLUME/4);
      nrf0+=1;
   }

   for (i=0;(i<nf)&&(fld<flm);i++)
   {
      f[i]=(double*)(fld);
      fld+=(VOLUME/8);
      nf0+=1;
   }

   if (nf0<nf)
   {
      pf=amalloc((nf-nf0)*VOLUME*sizeof(*pf),ALIGN);
      error(pf==NULL,1,"alloc_flds [xms1.c]","Unable to allocate fields");

      for (i=nf0;i<nf;i++)
      {
         f[i]=pf;
         pf+=VOLUME;
      }
   }

   if (nrf0<nrf)
   {
      prf=amalloc((nrf-nrf0)*VOLUME*sizeof(*prf),ALIGN);
      error(prf==NULL,1,"alloc_flds [xms1.c]","Unable to allocate fields");

      for (i=nrf0;i<nrf;i++)
      {
         rf[i]=prf;
         prf+=VOLUME;
      }
   }

   for (i=0;i<nf;i++)
   {
      for (j=0;j<VOLUME;j++)
         f[i][j]=0.0;
   }

   for (i=0;i<nrf;i++)
   {
      for (j=0;j<VOLUME;j++)
      {
         rf[i][j].re=0.0;
         rf[i][j].im=0.0;
      }
   }
}


static void integrate_wflow(double t0,double t1)
{
   int ns,rule;
   double dt,del,eps;
   wflow_parms_t wfl;

   wfl=wflow_parms();
   rule=wfl.rule;
   eps=wfl.eps;

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
      integrate_wflow(1.0,t1);
}


static double prodXX(u3_alg_dble *X)
{
   double p;

   p=(-2.0/3.0)*((*X).c1+(*X).c2+(*X).c3)*((*X).c1+(*X).c2+(*X).c3)+
      2.0*((*X).c1*(*X).c1+(*X).c2*(*X).c2+(*X).c3*(*X).c3)+
      4.0*((*X).c4*(*X).c4+(*X).c5*(*X).c5+(*X).c6*(*X).c6+
           (*X).c7*(*X).c7+(*X).c8*(*X).c8+(*X).c9*(*X).c9);

   return p;
}


static double prodXY(u3_alg_dble *X,u3_alg_dble *Y)
{
   double p;

   p=(-2.0/3.0)*((*X).c1+(*X).c2+(*X).c3)*((*Y).c1+(*Y).c2+(*Y).c3)+
      2.0*((*X).c1*(*Y).c1+(*X).c2*(*Y).c2+(*X).c3*(*Y).c3)+
      4.0*((*X).c4*(*Y).c4+(*X).c5*(*Y).c5+(*X).c6*(*Y).c6+
           (*X).c7*(*Y).c7+(*X).c8*(*Y).c8+(*X).c9*(*Y).c9);

   return p;
}


static double qdensity(int ix)
{
   double dn;

   dn=prodXY(ft[0]+ix,ft[3]+ix)+
      prodXY(ft[1]+ix,ft[4]+ix)+
      prodXY(ft[2]+ix,ft[5]+ix);

   return dn;
}


static double sdensity(int ix)
{
   double dn;

   dn=prodXX(ft[0]+ix)+prodXX(ft[1]+ix)+prodXX(ft[2]+ix)+
      prodXX(ft[3]+ix)+prodXX(ft[4]+ix)+prodXX(ft[5]+ix);

   return 0.5*dn;
}


static void set_flds(void)
{
   int ntm,i,is,ix;
   double pi,s,r,v,*tm;
   double **fE,**fEs;

   ntm=file_head.ntm;
   tm=file_head.tm;

   pi=4.0*atan(1.0);
   s=1.0/(8.0*pi*pi);
   v=(double)(N0*N1)*(double)(N2*N3);

   fE=f+1;
   fEs=fE+ntm;

   for (i=0;i<ntm;i++)
   {
      if (i==0)
         integrate_wflow(0.0,tm[0]);
      else
         integrate_wflow(tm[i-1],tm[i]);

      ft=ftensor();

      for (ix=0;ix<VOLUME;ix++)
         f[0][ix]=s*qdensity(ix);

      data.Q[i]=v*avg_msfld(f[0]);

      if (fabs(data.Q[i])<=Qmax)
      {
         is=1;
         nacc[i]+=1;
      }
      else
         is=0;

      r=tm[i]*tm[i];

      for (ix=0;ix<VOLUME;ix++)
         fE[i][ix]=r*sdensity(ix);

      if (is)
         add_msfld(fEs[i],fE[i]);
   }
}


static void set_data(int idat)
{
   int ntm,dmax,i,j,k;
   double rn,**fE,***var;

   ntm=file_head.ntm;

   if (idat==1)
   {
      fE=f+1+ntm;

      for (i=0;i<ntm;i++)
      {
         if (nacc[i]>1)
         {
            rn=1.0/(double)(nacc[i]);
            mulr_msfld(rn,fE[i]);
         }
      }
   }
   else
      fE=f+1;

   dmax=file_head.dmax;
   var=data.var;

   for (i=0;i<ntm;i++)
   {
      data.E[i]=center_msfld(fE[i]);

      for (j=0;j<=i;j++)
      {
         if (i3d)
            cov3d_msfld(dmax,fE[i],fE[j],rf[0],rf[1],f[0],var[i][j]);
         else
            cov_msfld(dmax,fE[i],fE[j],rf[0],rf[1],f[0],var[i][j]);

         if (i!=j)
         {
            for (k=0;k<=dmax;k++)
               var[j][i][k]=var[i][j][k];
         }
      }
   }
}


static void save_data(int idat)
{
   if (my_rank==0)
   {
      if (idat==0)
         fdat=fopen(dat0_file,"ab");
      else
         fdat=fopen(dat1_file,"ab");
      error_root(fdat==NULL,1,"save_data [xms1.c]",
                 "Unable to open data file");
      write_data(idat);
      fclose(fdat);
   }
}


static void print_log(int idat)
{
   int ntm,dmax,i,j;
   double *tm;

   if (my_rank==0)
   {
      ntm=file_head.ntm;
      dmax=file_head.dmax;
      tm=file_head.tm;

      printf("\n");
      printf("     t      ");

      if (idat==1)
         printf("  nacc   ");
      else
         printf("    Q    ");

      printf("   t^2*<E>    ");

      for (i=0;i<=dmax;i++)
         printf("  err[%2d]",i);

      printf("\n");

      for (i=0;i<ntm;i++)
      {
         if (idat==1)
            printf(" %.3e    %3d     %.6e ",tm[i],nacc[i],data.E[i]);
         else
            printf(" %.3e  % .1e  %.6e ",tm[i],data.Q[i],data.E[i]);

         for (j=0;j<=dmax;j++)
            printf("  %.1e",sqrt(data.var[i][i][j]));

         printf("\n");
      }

      printf("\n");

      if ((idat==0)&&((noflds==0)||(nosm==0)))
      {
         printf("Current sample sizes:");

         for (i=0;i<ntm;i++)
            printf(" %d",nacc[i]);

         printf("\n");
      }
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
   int nc,iend,iact[1];
   double wt1,wt2,wtavg;
   wflow_parms_t wfl;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   check_machine();
   iact[0]=0;
   set_hmc_parms(1,iact,0,0,NULL,1,1.0);
   set_bc_parms(3,1.0,1.0,1.0,1.0,zero,zero,zero);
   alloc_data();
   geometry();
   check_files();
   print_info();

   wfl=wflow_parms();
   if (wfl.rule)
      alloc_wfd(1);
   alloc_flds(1+2*wfl.ntm,2);

   if (append)
   {
      if (ipsm==0)
      {
         sprintf(cnfg_file,"%sn%d",nbase,first-step);
         read_msflds(iodat+1,cnfg_file,wfl.ntm,f+1+wfl.ntm);
      }
      else
      {
         sprintf(cnfg_file,"%sn%d",nbase,last);
         read_msflds(iodat+1,cnfg_file,wfl.ntm,f+1+wfl.ntm);
      }

      if (my_rank==0)
         printf("\n");
   }

   if (ipsm==0)
   {
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

         set_flds();
         data.nc=nc;
         set_data(0);
         save_data(0);
         print_log(0);

         if (noflds==0)
            save_msflds(nc,nbase,iodat+1,wfl.ntm,f+1+wfl.ntm);

         MPI_Barrier(MPI_COMM_WORLD);
         wt2=MPI_Wtime();
         wtavg+=(wt2-wt1);

         if (my_rank==0)
         {
            printf("Configuration fully processed in %.2e sec ",wt2-wt1);
            printf("(average = %.2e sec)\n\n",
                   wtavg/(double)((nc-first)/step+1));
            fflush(flog);
            copy_file(log_file,log_save);
            copy_file(dat0_file,dat0_save);
         }

         if ((noflds==0)&&((nc>first)||(append)))
            remove_flds(nc-step,nbase,iodat+1,0x8);

         check_endflag(&iend);
      }

      last=nc-step;
   }

   if ((nosm==0)&&(iend==0)&&(start<last))
   {
      nc=(last-start)/step+1;

      if (my_rank==0)
      {
         printf("Sample average\n");
         printf("--------------\n\n");

         printf("Processed %d configurations "
                "(no %d to %d in steps of  %d)\n",nc,start,last,step);
      }

      write_file_head(1);
      data.nc=nc;
      set_data(1);
      save_data(1);
      print_log(1);

      if (my_rank==0)
      {
         fflush(flog);
         copy_file(log_file,log_save);
         copy_file(dat1_file,dat1_save);
      }
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
