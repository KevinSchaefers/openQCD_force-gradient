
/*******************************************************************************
*
* File check6.c
*
* Copyright (C) 2017, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Writing and reading observable fields.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "flags.h"
#include "random.h"
#include "utils.h"
#include "archive.h"
#include "lattice.h"
#include "msfcts.h"
#include "global.h"

static int ns=0;
static double **fs;


static void alloc_flds(int n)
{
   int i;
   double *p1,**p2;

   if (n>ns)
   {
      if (ns>0)
      {
         free(fs[0]);
         free(fs);
      }

      if (n>0)
      {
         p1=malloc(n*VOLUME*sizeof(*p1));
         p2=malloc(n*sizeof(*p2));

         error((p1==NULL)||(p2==NULL),1,"alloc_fs [check6.c]",
               "Unable to allocate field arrays");

         for (i=0;i<(n*VOLUME);i++)
            p1[i]=0.0;

         fs=p2;

         for (i=0;i<n;i++)
         {
            p2[0]=p1;
            p2+=1;
            p1+=VOLUME;
         }
      }

      ns=n;
   }
}


static void random_flds(int n,double **f)
{
   int i;

   for (i=0;i<n;i++)
      gauss_dble(f[i],VOLUME);
}


static int cmp_flds(int n,double **f1,double **f2)
{
   int i,j,k;

   k=0;

   for (i=0;i<n;i++)
   {
      for (j=0;j<VOLUME;j++)
         k|=(f1[i][j]!=f2[i][j]);
   }

   if (NPROC>1)
   {
      i=k;
      MPI_Reduce(&i,&k,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&k,1,MPI_INT,0,MPI_COMM_WORLD);
   }

   return k;
}


int main(int argc,char *argv[])
{
   int my_rank,ie;
   int nion,nios;
   char local_dir[NAME_SIZE],flds[NAME_SIZE];
   char loc_sub_dir[NAME_SIZE];
   FILE *flog=NULL,*fin=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check6.log","w",stdout);
      fin=freopen("check6.in","r",stdin);

      printf("\n");
      printf("Writing and reading observable fields\n");
      printf("-------------------------------------\n\n");

      print_lattice_sizes();

      read_line("local_dir","%s",local_dir);
      read_line("nio_nodes","%d",&nion);
      read_line("nio_streams","%d",&nios);
      fclose(fin);

      error_root((nion<1)||(nios<1)||((NPROC%nion)!=0)||((NPROC%nios)!=0),1,
                 "main [check6.c]","Improper nio_nodes or nio_streams");
      error_root(name_size("%s/%d/%d/test.fld_%d",local_dir,nion-1,
                           NPROC/nion-1,NPROC-1)>=NAME_SIZE,1,
                 "main [check6.c]","local_dir name is too long");
   }

   MPI_Bcast(local_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&nion,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nios,1,MPI_INT,0,MPI_COMM_WORLD);

   start_ranlux(0,123456);
   geometry();
   alloc_flds(6);
   sprintf(loc_sub_dir,"%s/%d/%d",local_dir,
           my_rank/(NPROC/nion),my_rank%(NPROC/nion));
   check_dir(loc_sub_dir);
   sprintf(flds,"%s/test.fld_%d",loc_sub_dir,my_rank);
   set_nio_streams(nios);

   if (my_rank==0)
   {
      printf("nio_nodes=%d\n",nion);
      printf("nio_streams=%d\n\n",nios);

      printf("Write random observable fields to the files\n"
             "%s/*/*/test.fld_*\n"
             "on the local disks.\n\n",local_dir);
      printf("Then read the fields from there, compare with the saved fields\n"
             "and remove all files.\n\n");
   }

   random_flds(6,fs);
   write_msfld(flds,3,fs);
   read_msfld(flds,3,fs+3);
   remove(flds);

   ie=cmp_flds(3,fs,fs+3);
   error(ie!=0,1,"main [check6.c]",
         "The observable fields are not properly restored");

   if (my_rank==0)
   {
      printf("No errors detected --- the fields are correctly written\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
