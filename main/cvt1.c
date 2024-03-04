
/*******************************************************************************
*
* File cvt1.c
*
* Copyright (C) 2017-2019, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Conversion of gauge-field configurations from one storage format to another.
*
* Syntax: cvt1 -i <filename> [-rmold]
*
* For usage instructions see the file README.cvt1.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "random.h"
#include "archive.h"
#include "version.h"
#include "global.h"

static int my_rank,rmold;
static int first,last,step;
static iodat_t iodat[2];
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

   check_dir_root(log_dir);
   error(name_size("%s/%s.log~",log_dir,nbase)>=NAME_SIZE,1,
         "read_dirs [cvt1.c]","log_dir name is too long");
   sprintf(log_file,"%s/%s.log",log_dir,nbase);
   sprintf(end_file,"%s/%s.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);
}


static void read_range(void)
{
   if (my_rank==0)
   {
      find_section("Input configurations");
      read_line("first","%d",&first);
      read_line("last","%d",&last);
      read_line("step","%d",&step);

      error_root((first<1)||(last<first)||(step<1)||((last-first)%step!=0),1,
                 "read_range [cvt1.c]","Improper configuration range");
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
      rmold=find_opt(argc,argv,"-rmold");

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [cvt1.c]",
                 "Syntax: cvt1 -i <filename> [-rmold]");

      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [cvt1.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&rmold,1,MPI_INT,0,MPI_COMM_WORLD);

   read_dirs();
   read_bc_parms("Boundary conditions",0x0);
   read_iodat("Input configurations","i",iodat);
   read_range();
   read_iodat("Output configurations","o",iodat+1);

   if (my_rank==0)
      fclose(fin);
}


static void check_files(void)
{
   error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
         "check_files [cvt1.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);
   check_iodat(iodat+1,"o",0x1,cnfg_file);
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

      flog=freopen(log_file,"w",stdout);
      error_root(flog==NULL,1,"print_info [cvt1.c]","Unable to open log file");

      printf("\n");
      printf("Conversion of gauge-field configurations\n");
      printf("----------------------------------------\n\n");

      printf("Program version %s\n",openQCD_RELEASE);

      print_lattice_sizes();
      print_bc_parms(0x0);
      print_iodat("i",iodat);
      print_iodat("o",iodat+1);

      printf("Configurations no %d -> %d in steps of %d\n",
             first,last,step);
      if (rmold)
         printf("Input configurations are deleted after copying\n");
      printf("\n");

      fflush(flog);
   }
}


static void print_log(int icnfg,double *wt,double *wtall)
{
   double r;

   if (my_rank==0)
   {
      r=(double)((icnfg-first)/step+1);

      printf("Configuration no %d fully processed in %.2e sec "
             "(average = %.2e)\n",icnfg,wt[0]+wt[1],(wtall[0]+wtall[1])/r);
      printf("Average read time = %.2e, average write time = %.2e\n\n",
             wtall[0]/r,wtall[1]/r);

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
   int icnfg,iend;
   double wt1,wt2,wt3,wt[2],wtall[2];

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   check_machine();
   geometry();
   check_files();
   print_info();

   iend=0;
   wtall[0]=0.0;
   wtall[1]=0.0;

   if ((iodat[0].types&0x4)||(iodat[1].types&0x4))
      start_ranlux(0,1);

   for (icnfg=first;(iend==0)&&(icnfg<=last);icnfg+=step)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      sprintf(cnfg_file,"%sn%d",nbase,icnfg);
      read_flds(iodat,cnfg_file,0x0,0x1);

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      save_flds(icnfg,nbase,iodat+1,0x1);

      MPI_Barrier(MPI_COMM_WORLD);
      wt3=MPI_Wtime();

      wt[0]=wt2-wt1;
      wt[1]=wt3-wt2;
      wtall[0]+=wt[0];
      wtall[1]+=wt[1];
      print_log(icnfg,wt,wtall);

      if (my_rank==0)
      {
         fflush(flog);
         copy_file(log_file,log_save);
      }

      if (rmold)
         remove_flds(icnfg,nbase,iodat,0x1);

      check_endflag(&iend);
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
