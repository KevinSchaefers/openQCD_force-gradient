
/*******************************************************************************
*
* File qcd1.c
*
* Copyright (C) 2011-2019, 2022 Martin Luescher, Isabel Campos
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* HMC simulation program for QCD with Wilson quarks.
*
* Syntax: qcd1 -i <filename> [-c <filename> [-a [-norng]]|[-mask <int>]]
*                            [-rmold] [-noms]
*
* For usage instructions see the file README.qcd1.
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
#include "su3fcts.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "archive.h"
#include "forces.h"
#include "update.h"
#include "wflow.h"
#include "tcharge.h"
#include "version.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

typedef struct
{
   int nt,iac;
   double dH,avpl;
} dat_t;

static struct
{
   int dn,nn,tmax;
   double eps;
} file_head;

static struct
{
   int nt;
   double **Wsl,**Ysl,**Qsl;
} data;

static int my_rank,mask,norng,rmold,noms;
static int scnfg,append,endian,level,seed;
static int nth,ntr,dtr_log,dtr_ms,dtr_cnfg;
static int ipgrd[3];
static double *Wact,*Yact,*Qtop;

static iodat_t iodat[2];
static char log_dir[NAME_SIZE],dat_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE];
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char dat_file[NAME_SIZE],dat_save[NAME_SIZE];
static char msdat_file[NAME_SIZE],msdat_save[NAME_SIZE];
static char rng_file[NAME_SIZE],rng_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE],end_file[NAME_SIZE];
static char nbase[NAME_SIZE],cnfg[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fdat=NULL,*fend=NULL;


static int write_dat(int n,dat_t *ndat)
{
   int i,iw,ic;
   stdint_t istd[2];
   double dstd[2];

   ic=0;

   for (i=0;i<n;i++)
   {
      istd[0]=(stdint_t)((*ndat).nt);
      istd[1]=(stdint_t)((*ndat).iac);

      dstd[0]=(*ndat).dH;
      dstd[1]=(*ndat).avpl;

      if (endian==BIG_ENDIAN)
      {
         bswap_int(2,istd);
         bswap_double(2,dstd);
      }

      iw=fwrite(istd,sizeof(stdint_t),2,fdat);
      iw+=fwrite(dstd,sizeof(double),2,fdat);

      if (iw!=4)
         return ic;

      ic+=1;
      ndat+=1;
   }

   return ic;
}


static int read_dat(int n,dat_t *ndat)
{
   int i,ir,ic;
   stdint_t istd[2];
   double dstd[2];

   ic=0;

   for (i=0;i<n;i++)
   {
      ir=fread(istd,sizeof(stdint_t),2,fdat);
      ir+=fread(dstd,sizeof(double),2,fdat);

      if (ir!=4)
         return ic;

      if (endian==BIG_ENDIAN)
      {
         bswap_int(2,istd);
         bswap_double(2,dstd);
      }

      (*ndat).nt=(int)(istd[0]);
      (*ndat).iac=(int)(istd[1]);

      (*ndat).dH=dstd[0];
      (*ndat).avpl=dstd[1];

      ic+=1;
      ndat+=1;
   }

   return ic;
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
   error((pp==NULL)||(p==NULL),1,"alloc_data [qcd1.c]",
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


static void set_file_head(void)
{
   wflow_parms_t wfl;

   wfl=wflow_parms();

   file_head.dn=wfl.dnms;
   file_head.nn=wfl.ntot/wfl.dnms;
   file_head.tmax=N0;
   file_head.eps=wfl.eps;
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

   error_root(iw!=4,1,"write_file_head [qcd1.c]",
              "Incorrect write count");
}


static void check_file_head(void)
{
   int ir;
   stdint_t istd[3];
   double dstd[1];

   ir=fread(istd,sizeof(stdint_t),3,fdat);
   ir+=fread(dstd,sizeof(double),1,fdat);

   error_root(ir!=4,1,"check_file_head [qcd1.c]",
              "Incorrect read count");

   if (endian==BIG_ENDIAN)
   {
      bswap_int(3,istd);
      bswap_double(1,dstd);
   }

   error_root(((int)(istd[0])!=file_head.dn)||
              ((int)(istd[1])!=file_head.nn)||
              ((int)(istd[2])!=file_head.tmax)||
              (dstd[0]!=file_head.eps),1,"check_file_head [qcd1.c]",
              "Unexpected value of dn,nn,tmax or eps");
}


static void write_data(void)
{
   int iw,nn,tmax;
   int k,in,t;
   stdint_t istd[1];
   double **Xsl[3],dstd[1];

   istd[0]=(stdint_t)(data.nt);

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

   error_root(iw!=(1+3*(nn+1)*tmax),1,"write_data [qcd1.c]",
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

   data.nt=(int)(istd[0]);

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

   error_root(ir!=(1+3*(nn+1)*tmax),1,"read_data [qcd1.c]",
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
   error(name_size("%s/%s.log~",log_dir,nbase)>=NAME_SIZE,1,
         "setup_files [qcd1.c]","log_dir name is too long");
   error(name_size("%s/%s.ms.dat~",dat_dir,nbase)>=NAME_SIZE,1,
         "setup_files [qcd1.c]","dat_dir name is too long");

   sprintf(log_file,"%s/%s.log",log_dir,nbase);
   sprintf(par_file,"%s/%s.par",dat_dir,nbase);
   sprintf(dat_file,"%s/%s.dat",dat_dir,nbase);
   sprintf(msdat_file,"%s/%s.ms.dat",dat_dir,nbase);
   sprintf(rng_file,"%s/%s.rng",dat_dir,nbase);
   sprintf(end_file,"%s/%s.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);
   sprintf(dat_save,"%s~",dat_file);
   sprintf(msdat_save,"%s~",msdat_file);
   sprintf(rng_save,"%s~",rng_file);

   check_dir_root(log_dir);
   check_dir_root(dat_dir);
}


static void read_ranlux_parms(void)
{
   if (my_rank==0)
   {
      if ((append==0)||(norng))
      {
         find_section("Random number generator");
         read_line("level","%d",&level);
         read_line("seed","%d",&seed);
      }
      else
      {
         level=0;
         seed=0;
      }
   }

   MPI_Bcast(&level,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void read_schedule(void)
{
   int ie;

   if (my_rank==0)
   {
      find_section("MD trajectories");
      read_line("nth","%d",&nth);
      read_line("ntr","%d",&ntr);
      read_line("dtr_log","%d",&dtr_log);
      read_line("dtr_cnfg","%d",&dtr_cnfg);
      if (noms==0)
         read_line("dtr_ms","%d",&dtr_ms);
      else
         dtr_ms=0;

      error_root((append!=0)&&(nth!=0),1,"read_schedule [qcd1.c]",
                 "Continuation run: nth must be equal to zero");

      ie=0;
      ie|=(nth<0);
      ie|=(ntr<1);
      ie|=(dtr_log<1);
      ie|=(dtr_log>dtr_cnfg);
      ie|=((dtr_cnfg%dtr_log)!=0);
      ie|=((nth%dtr_cnfg)!=0);
      ie|=((ntr%dtr_cnfg)!=0);

      if (noms==0)
      {
         ie|=(dtr_ms<dtr_log);
         ie|=(dtr_ms>dtr_cnfg);
         ie|=((dtr_ms%dtr_log)!=0);
         ie|=((dtr_cnfg%dtr_ms)!=0);
      }

      error_root(ie!=0,1,"read_schedule [qcd1.c]",
                 "Improper value of nth,ntr,dtr_log,dtr_ms or dtr_cnfg");
   }

   MPI_Bcast(&nth,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ntr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&dtr_log,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&dtr_cnfg,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&dtr_ms,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void write_schedule(void)
{
   int iw;
   stdint_t istd[3];

   if (my_rank==0)
   {
      istd[0]=(stdint_t)(dtr_log);
      istd[1]=(stdint_t)(dtr_cnfg);
      istd[2]=(stdint_t)(dtr_ms);

      if (endian==BIG_ENDIAN)
         bswap_int(3,istd);

      iw=fwrite(istd,sizeof(stdint_t),3,fdat);
      error_root(iw!=3,1,"write_schedule [qcd1.c]",
                 "Incorrect write count");
   }
}


static void check_schedule(void)
{
   int ir,ie;
   stdint_t istd[3];

   if (my_rank==0)
   {
      ir=fread(istd,sizeof(stdint_t),3,fdat);
      error_root(ir!=3,1,"check_schedule [qcd1.c]",
                 "Incorrect read count");

      if (endian==BIG_ENDIAN)
         bswap_int(3,istd);

      ie=(((dtr_ms==0)&&((int)(istd[2])!=0))||
          ((dtr_ms!=0)&&((int)(istd[2])==0)));

      error_root(ie!=0,1,"check_schedule [qcd1.c]",
                 "Attempt to mix measurement with other runs");

      ie|=(istd[0]!=(stdint_t)(dtr_log));
      ie|=(istd[1]!=(stdint_t)(dtr_cnfg));
      ie|=(istd[2]!=(stdint_t)(dtr_ms));

      error_root(ie!=0,1,"check_schedule [qcd1.c]",
                 "Parameters do not match previous run");
   }
}


static void read_infile(int argc,char *argv[])
{
   int ifile,imask,ir;
   int isap,idfl;

   if (my_rank==0)
   {
      flog=freopen("STARTUP_ERROR","w",stdout);

      ifile=find_opt(argc,argv,"-i");
      scnfg=find_opt(argc,argv,"-c");
      append=find_opt(argc,argv,"-a");
      norng=find_opt(argc,argv,"-norng");
      imask=find_opt(argc,argv,"-mask");
      rmold=find_opt(argc,argv,"-rmold");
      noms=find_opt(argc,argv,"-noms");
      endian=endianness();

      error_root((ifile==0)||(ifile==(argc-1))||(scnfg==(argc-1))||
                 ((append!=0)&&(scnfg==0)),1,"read_infile [qcd1.c]",
                 "Syntax: qcd1 -i <filename> "
                 "[-c <filename> [-a [-norng]]|[-mask <int>]] "
                 "[-rmold] [-noms]");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [qcd1.c]",
                 "Machine has unknown endianness");

      if ((scnfg)&&(!append)&&(imask))
      {
         ir=sscanf(argv[imask+1],"%i",&mask);

         error_root(ir!=1,1,"read_infile [qcd1.c]",
                    "Syntax: qcd1 -i <filename> "
                    "[-c <filename> [-a [-norng]]|[-mask <int>]] "
                    "[-rmold] [-noms]");
         error_root((mask<0x0)||(mask>0xf),1,"read_infile [qcd1.c]",
                    "Command line argument 'mask' is out of range");
      }
      else
         mask=0x0;

      if (scnfg)
      {
         strncpy(cnfg,argv[scnfg+1],NAME_SIZE-1);
         cnfg[NAME_SIZE-1]='\0';
      }
      else
         cnfg[0]='\0';

      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [qcd1.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&scnfg,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&append,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&norng,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&mask,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&rmold,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&noms,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(cnfg,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   read_dirs();

   if (scnfg)
      read_iodat("Initial configuration","i",iodat);
   read_iodat("Configurations","o",iodat+1);
   read_lat_parms("Lattice parameters",0x3);
   read_bc_parms("Boundary conditions",0x3);
   read_schedule();
   read_ranlux_parms();
   read_hmc_parms("HMC parameters",0x3);
   read_all_action_parms();
   read_all_force_parms();
   read_all_mdint_parms();
   read_all_solver_parms(&isap,&idfl);
   if (isap)
      read_sap_parms("SAP",0x1);
   if (idfl)
   {
      read_dfl_parms("Deflation subspace");
      read_dfl_pro_parms("Deflation projection");
      read_dfl_gen_parms("Deflation subspace generation");
      read_dfl_upd_parms("Deflation update scheme");
   }
   if (noms==0)
      read_wflow_parms("Wilson flow",0x1);

   setup_files();

   if (my_rank==0)
   {
      fdat=fopen(par_file,"rb");

      if (append==0)
      {
         error_root(fdat!=NULL,1,"read_infile [qcd1.c]",
                    "Attempt to overwrite old parameter file");
         fdat=fopen(par_file,"wb");
      }

      error_root(fdat==NULL,1,"read_infile [qcd1.c]",
                 "Unable to open parameter file");
   }

   if (append)
   {
      check_iodat_parms(fdat,iodat+1);
      check_lat_parms(fdat);
      check_bc_parms(fdat);
      check_schedule();
      check_hmc_parms(fdat);
      check_action_parms(fdat);
      check_rat_parms(fdat);
      check_mdint_parms(fdat);
      check_force_parms(fdat);
      check_sap_parms(fdat);
      check_dfl_parms(fdat);
      check_solver_parms(fdat);
      if (noms==0)
         check_wflow_parms(fdat);
   }
   else
   {
      write_iodat_parms(fdat,iodat+1);
      write_lat_parms(fdat);
      write_bc_parms(fdat);
      write_schedule();
      write_hmc_parms(fdat);
      write_action_parms(fdat);
      write_rat_parms(fdat);
      write_mdint_parms(fdat);
      write_force_parms(fdat);
      write_sap_parms(fdat);
      write_dfl_parms(fdat);
      write_solver_parms(fdat);
      if (noms==0)
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


static void check_old_log(int ic,int *nl,int *icnfg)
{
   int ir,isv,irg,lv,sd;
   int nt,np[4],bp[4];
   char line[NAME_SIZE];

   fend=fopen(log_file,"r");
   error_root(fend==NULL,1,"check_old_log [qcd1.c]",
              "Unable to open log file");
   (*nl)=0;
   (*icnfg)=0;
   ir=1;
   isv=0;
   irg=0;

   while (fgets(line,NAME_SIZE,fend)!=NULL)
   {
      if (strstr(line,"MPI process grid")!=NULL)
      {
         ir&=(sscanf(line,"%dx%dx%dx%d MPI process grid, %dx%dx%dx%d",
                     np,np+1,np+2,np+3,bp,bp+1,bp+2,bp+3)==8);

         ipgrd[0]=((np[0]!=NPROC0)||(np[1]!=NPROC1)||
                   (np[2]!=NPROC2)||(np[3]!=NPROC3));
         ipgrd[1]=((bp[0]!=NPROC0_BLK)||(bp[1]!=NPROC1_BLK)||
                   (bp[2]!=NPROC2_BLK)||(bp[3]!=NPROC3_BLK));
      }
      else if (strstr(line,"OpenMP thread")!=NULL)
      {
         ir&=(sscanf(line,"%d OpenMP thread",&nt)==1);
         ipgrd[2]=(nt!=NTHREAD);
      }
      else if (strstr(line,"Trajectory no")!=NULL)
      {
         ir&=(sscanf(line,"Trajectory no %d",nl)==1);
         isv=0;
      }
      else if (strstr(line,"Configuration no")!=NULL)
      {
         ir&=(sscanf(line,"Configuration no %d",icnfg)==1);
         isv=1;
      }
      else if (norng)
      {
         if ((strstr(line,"level =")!=NULL)&&(strstr(line,"seed =")!=NULL))
         {
            ir&=(sscanf(strstr(line,"level ="),
                        "level = %d, seed = %d",&lv,&sd)==2);
            irg|=((lv==level)&&(sd==seed));
         }
      }
   }

   fclose(fend);

   error_root(ir!=1,1,"check_old_log [qcd1.c]","Incorrect read count");

   error_root(ic!=(*icnfg),1,"check_old_log [qcd1.c]",
              "Continuation run:\n"
              "Initial configuration is not the last one of the previous run");

   error_root(isv==0,1,"check_old_log [qcd1.c]",
              "Continuation run:\n"
              "The log file extends beyond the last configuration save");

   error_root(irg!=0,1,"check_old_log [qcd1.c]",
              "Continuation run:\n"
              "Attempt to reuse previously used ranlux level and seed");

   error_root((ipgrd[0])&&(norng==0),1,"check_old_log [qcd1.c]",
              "Continuation run:\n"
              "MPI process grid changed, but -norng is not set");

   error_root((ipgrd[2])&&(norng==0),1,"check_old_log [qcd1.c]",
              "Continuation run:\n"
              "No of OpenMP threads changed, but -norng is not set");
}


static void check_old_dat(int nl)
{
   int nt;
   dat_t ndat;

   fdat=fopen(dat_file,"rb");
   error_root(fdat==NULL,1,"check_old_dat [qcd1.c]",
              "Unable to open data file");
   nt=0;

   while (read_dat(1,&ndat)==1)
      nt=ndat.nt;

   fclose(fdat);

   error_root(nt!=nl,1,"check_old_dat [qcd1.c]",
              "Continuation run: Incomplete or too many data records");
}


static void check_old_msdat(int nl)
{
   int ic,ir,nt,pnt,dnt;

   fdat=fopen(msdat_file,"rb");
   error_root(fdat==NULL,1,"check_old_msdat [qcd1.c]",
              "Unable to open data file");

   check_file_head();

   nt=0;
   dnt=0;
   pnt=0;

   for (ic=0;;ic++)
   {
      ir=read_data();

      if (ir==0)
      {
         error_root(ic==0,1,"check_old_msdat [qcd1.c]",
                    "No data records found");
         break;
      }

      nt=data.nt;

      if (ic==1)
      {
         dnt=nt-pnt;
         error_root(dnt<1,1,"check_old_msdat [qcd1.c]",
                    "Incorrect trajectory separation");
      }
      else if (ic>1)
         error_root(nt!=(pnt+dnt),1,"check_old_msdat [qcd1.c]",
                    "Trajectory sequence is not equally spaced");

      pnt=nt;
   }

   fclose(fdat);

   error_root((nt!=nl)||((ic>1)&&(dnt!=dtr_ms)),1,
              "check_old_msdat [qcd1.c]","Last trajectory numbers "
              "or the trajectory separations do not match");
}


static void check_files(int *nl,int *icnfg)
{
   int ie,ic,icmax;

   ipgrd[0]=0;
   ipgrd[1]=0;
   ipgrd[2]=0;

   if (my_rank==0)
   {
      if (append)
      {
         error_root(strstr(cnfg,nbase)!=cnfg,1,
                    "check_files [qcd1.c]","Continuation run:\n"
                    "Run name does not match the previous one");
         error_root(sscanf(cnfg+strlen(nbase),"n%d",&ic)!=1,1,
                    "check_files [qcd1.c]","Continuation run:\n"
                    "Unable to read configuration number from file name");
         error_root(((iodat[0].types&iodat[1].types)==0x0)||
                    ((iodat[0].types&0x6)&&
                     (iodat[0].nio_nodes!=iodat[1].nio_nodes)),1,
                    "check_files [qcd1.c]","Continuation run:\n"
                    "Unexpected initial configuration storage type");

         check_old_log(ic,nl,icnfg);
         check_old_dat(*nl);
         if (noms==0)
            check_old_msdat(*nl);

         (*icnfg)+=1;
      }
      else
      {
         ie=check_file(log_file,"r");
         ie|=check_file(dat_file,"rb");

         if (noms==0)
            ie|=check_file(msdat_file,"rb");

         error_root(ie!=0,1,"check_files [qcd1.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         if (noms==0)
         {
            fdat=fopen(msdat_file,"wb");
            error_root(fdat==NULL,1,"check_files [qcd1.c]",
                       "Unable to open measurement data file");
            write_file_head();
            fclose(fdat);
         }

         (*nl)=0;
         (*icnfg)=1;
      }
   }

   MPI_Bcast(nl,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(icnfg,1,MPI_INT,0,MPI_COMM_WORLD);
   icmax=(*icnfg)+(ntr-nth)/dtr_cnfg;

   error(name_size("%sn%d",nbase,icmax)>=NAME_SIZE,1,
         "check_files [qcd1.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,icmax);
   check_iodat(iodat+1,"o",0x1,cnfg_file);

   if (scnfg)
      check_iodat(iodat,"i",0x1,cnfg);
}


static void init_rng(int icnfg)
{
   int ic;

   if (append)
   {
      if (norng)
         start_ranlux(level,seed);
      else
      {
         ic=import_ranlux(rng_file);
         error_root(ic!=(icnfg-1),1,"init_rng [qcd1.c]",
                    "Configuration number mismatch (*.rng file)");
      }
   }
   else
      start_ranlux(level,seed);
}


static void init_ud(void)
{
   if (scnfg)
   {
      if (append)
         read_flds(iodat,cnfg,0x0,0x1);
      else
         read_flds(iodat,cnfg,mask,0x1);

      if (my_rank==0)
         printf("\n");
   }
   else
      random_ud();
}


static void store_ud(su3_dble *usv)
{
   su3_dble *udb;

   udb=udfld();
   assign_ud2ud(4*VOLUME_TRD,2,udb,usv);
}


static void recall_ud(su3_dble *usv)
{
   su3_dble *udb;

   udb=udfld();
   assign_ud2ud(4*VOLUME_TRD,2,usv,udb);
   set_flags(UPDATED_UD);
}


static void set_data(int nt)
{
   int in,dn,nn;
   double eps;
   wflow_parms_t wfl;

   wfl=wflow_parms();

   data.nt=nt;
   dn=file_head.dn;
   nn=file_head.nn;
   eps=file_head.eps;

   for (in=0;in<nn;in++)
   {
      Wact[in]=plaq_action_slices(data.Wsl[in]);
      Yact[in]=ym_action_slices(data.Ysl[in]);
      Qtop[in]=tcharge_slices(data.Qsl[in]);

      if (wfl.rule==0)
         fwd_euler(dn,eps);
      else if (wfl.rule==1)
         fwd_rk2(dn,eps);
      else
         fwd_rk3(dn,eps);
   }

   Wact[in]=plaq_action_slices(data.Wsl[in]);
   Yact[in]=ym_action_slices(data.Ysl[in]);
   Qtop[in]=tcharge_slices(data.Qsl[in]);
}


static void print_info(int icnfg)
{
   int isap,idfl;
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

      error_root(flog==NULL,1,"print_info [qcd1.c]","Unable to open log file");

      if (append)
         printf("Continuation run, start from configuration %s\n\n",cnfg);
      else
      {
         printf("\n");
         printf("Simulation of QCD with Wilson quarks\n");
         printf("------------------------------------\n\n");

         if (scnfg)
         {
            if (iodat[0].types&0x3)
               printf("New run, start from configuration %s "
                      "(extension mask = %#x)\n\n",cnfg,(unsigned int)(mask));
            else
               printf("New run, start from configuration %s\n\n",cnfg);
         }
         else
            printf("New run, start from random configuration\n\n");

         printf("Using the HMC algorithm\n");
         printf("Program version %s\n",openQCD_RELEASE);
      }

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
         print_lat_parms(0x3);
         print_bc_parms(0x3);
      }

      printf("Random number generator:\n");

      if (append)
      {
         if (norng)
            printf("Reinitialized state using level = %d, seed = %d\n\n",
                   level,seed);
         else
            printf("Reset to the last exported state\n\n");
      }
      else
         printf("Initialized state using level = %d, seed = %d\n\n",
                level,seed);

      if (append)
      {
         printf("Trajectories:\n");
         printf("ntr = %d\n\n",ntr);
      }
      else
      {
         print_hmc_parms();
         printf("Trajectories:\n");
         printf("nth = %d, ntr = %d\n",nth,ntr);

         if (noms)
         {
            printf("dtr_log = %d, dtr_cnfg = %d\n\n",
                   dtr_log,dtr_cnfg);
            printf("Wilson flow observables are not measured\n\n");
         }
         else
         {
            printf("dtr_log = %d, dtr_ms = %d, dtr_cnfg = %d\n\n",
                   dtr_log,dtr_ms,dtr_cnfg);
            printf("Online measurement of Wilson flow observables\n\n");
         }

         print_mdint_parms();
         print_action_parms();
         print_rat_parms();
         print_force_parms(0x0);
         print_solver_parms(&isap,&idfl);

         if (isap)
            print_sap_parms(0x0);

         if (idfl)
            print_dfl_parms(0x1);

         if (noms==0)
            print_wflow_parms();
      }

      if (scnfg)
         print_iodat("i",iodat);
      print_iodat("o",iodat+1);
      if (rmold)
         printf("Old configurations are deleted\n\n");

      fflush(flog);
   }
}


static void print_log(dat_t *ndat)
{
   if (my_rank==0)
   {
      printf("Trajectory no %d\n",(*ndat).nt);
      printf("dH = %+.1e, ",(*ndat).dH);
      printf("iac = %d\n",(*ndat).iac);
      printf("Average plaquette = %.6f\n",(*ndat).avpl);
      print_all_avgstat();
   }
}


static void save_dat(int n,double siac,double wtcyc,double wtall,dat_t *ndat)
{
   int iw;

   if (my_rank==0)
   {
      fdat=fopen(dat_file,"ab");
      error_root(fdat==NULL,1,"save_dat [qcd1.c]",
                 "Unable to open data file");

      iw=write_dat(1,ndat);
      error_root(iw!=1,1,"save_dat [qcd1.c]",
                 "Incorrect write count");
      fclose(fdat);

      printf("Acceptance rate = %.6f\n",siac/(double)(n+1));
      printf("Time per trajectory = %.2e sec (average = %.2e sec)\n\n",
             wtcyc/(double)(dtr_log),wtall/(double)(n+1));
      fflush(flog);
   }
}


static void save_msdat(int n,double wtms,double wtmsall)
{
   int nms,in,dn,nn,din;
   double eps;

   if (my_rank==0)
   {
      fdat=fopen(msdat_file,"ab");
      error_root(fdat==NULL,1,"save_msdat [qcd1.c]",
                 "Unable to open data file");
      write_data();
      fclose(fdat);

      nms=(n+1-nth)/dtr_ms+(nth>0);
      dn=file_head.dn;
      nn=file_head.nn;
      eps=file_head.eps;

      din=nn/10;
      if (din<1)
         din=1;

      printf("Measurement run:\n\n");

      for (in=0;in<=nn;in+=din)
         printf("n = %3d, t = %.2e, Wact = %.6e, Yact = %.6e, Q = % .2e\n",
                in*dn,eps*(double)(in*dn),Wact[in],Yact[in],Qtop[in]);

      printf("\n");
      printf("Configuration fully processed in %.2e sec ",wtms);
      printf("(average = %.2e sec)\n",wtmsall/(double)(nms));
      printf("Measured data saved\n\n");
      fflush(flog);
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
   int nl,icnfg;
   int nwud,nwsds,nwv,nwvd;
   int n,iend,iac,i;
   double npl,siac,*qsm[1];
   double wt1,wt2,wtcyc,wtall,wtms,wtmsall;
   qflt *act0,*act1;
   su3_dble **usv;
   wflow_parms_t wfl;
   hmc_parms_t hmc;
   dat_t ndat;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   check_machine();
   if (noms==0)
   {
      set_file_head();
      alloc_data();
   }
   geometry();
   check_files(&nl,&icnfg);
   print_info(icnfg);

   hmc_sanity_check();
   hmc_wsize(&nwud,&nwsds,&nwv,&nwvd);
   alloc_wud(nwud);
   alloc_ws(nwsds);
   wsd_uses_ws();
   alloc_wv(nwv);
   alloc_wvd(nwvd);
   if (noms==0)
   {
      wfl=wflow_parms();
      if (wfl.rule)
         alloc_wfd(1);
   }

   hmc=hmc_parms();
   act0=malloc(2*(hmc.nact+1)*sizeof(*act0));
   act1=act0+hmc.nact+1;
   error(act0==NULL,1,"main [qcd1.c]","Unable to allocate action arrays");

   set_mdsteps();
   setup_counters();
   setup_chrono();
   if (!append)
      print_msize(1);
   init_rng(icnfg);
   init_ud();

   if (bc_type()==0)
      npl=(double)(6*(N0-1)*N1)*(double)(N2*N3);
   else
      npl=(double)(6*N0*N1)*(double)(N2*N3);

   iend=0;
   siac=0.0;
   wtcyc=0.0;
   wtall=0.0;
   wtms=0.0;
   wtmsall=0.0;

   for (n=0;(iend==0)&&(n<ntr);n++)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      iac=run_hmc(act0,act1);

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      siac+=(double)(iac);
      wtcyc+=(wt2-wt1);

      if (((ntr-n-1)%dtr_log)==0)
      {
         for (i=0;i<=hmc.nact;i++)
         {
            act0[i].q[0]=-act0[i].q[0];
            act0[i].q[1]=-act0[i].q[1];
            add_qflt(act0[i].q,act1[i].q,act1[i].q);
            if (i>0)
               add_qflt(act1[i].q,act1[0].q,act1[0].q);
         }

         qsm[0]=act1[0].q;
         global_qsum(1,qsm,qsm);

         ndat.nt=nl+n+1;
         ndat.iac=iac;
         ndat.dH=act1[0].q[0];
         ndat.avpl=plaq_wsum_dble(1)/npl;

         print_log(&ndat);
         wtall+=wtcyc;
         save_dat(n,siac,wtcyc,wtall,&ndat);
         wtcyc=0.0;

         if ((noms==0)&&((n+1)>=nth)&&(((ntr-n-1)%dtr_ms)==0))
         {
            MPI_Barrier(MPI_COMM_WORLD);
            wt1=MPI_Wtime();

            usv=reserve_wud(1);
            store_ud(usv[0]);
            set_data(nl+n+1);
            recall_ud(usv[0]);
            release_wud();

            MPI_Barrier(MPI_COMM_WORLD);
            wt2=MPI_Wtime();

            wtms=wt2-wt1;
            wtmsall+=wtms;
            save_msdat(n,wtms,wtmsall);
         }
      }

      if (((n+1)>=nth)&&(((ntr-n-1)%dtr_cnfg)==0))
      {
         save_flds(icnfg,nbase,iodat+1,0x1);
         export_ranlux(icnfg,rng_file);
         check_endflag(&iend);

         if (my_rank==0)
         {
            fflush(flog);
            copy_file(log_file,log_save);
            copy_file(dat_file,dat_save);
            if (noms==0)
               copy_file(msdat_file,msdat_save);
            copy_file(rng_file,rng_save);
         }

         if ((rmold)&&(icnfg>1))
            remove_flds(icnfg-1,nbase,iodat+1,0x1);

         icnfg+=1;
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
