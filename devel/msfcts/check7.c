
/*******************************************************************************
*
* File check7.c
*
* Copyright (C) 2017, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Exporting and importing observable fields.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "flags.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
#include "msfcts.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

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

         error((p1==NULL)||(p2==NULL),1,"alloc_fs [check7.c]",
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
   int my_rank,n,b,ie;
   int nio_nodes,nio_streams,nb,ib,nl[4],bs[4];
   char cnfg_dir[NAME_SIZE],block_dir[NAME_SIZE];
   char blk_sub_dir[NAME_SIZE],flds[NAME_SIZE];
   FILE *flog=NULL,*fin=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check7.log","w",stdout);
      fin=freopen("check7.in","r",stdin);

      printf("\n");
      printf("Exporting and importing observable fields\n");
      printf("-----------------------------------------\n\n");

      print_lattice_sizes();

      read_line("cnfg_dir","%s",cnfg_dir);
      read_line("block_dir","%s",block_dir);
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      read_line("nio_nodes","%d",&nio_nodes);
      read_line("nio_streams","%d",&nio_streams);
      fclose(fin);
   }

   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(block_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nio_nodes,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nio_streams,1,MPI_INT,0,MPI_COMM_WORLD);

   start_ranlux(0,123456);
   geometry();
   alloc_flds(6);

   if (my_rank==0)
   {
      printf("Export random observable fields to the files\n"
             "%s/test.fld.\n",cnfg_dir);
      printf("Then read the fields from there and compare with the saved "
             "fields.\n\n");
   }

   check_dir_root(cnfg_dir);
   error(name_size("%s/test.fld",cnfg_dir)>=NAME_SIZE,1,
         "main [check7.c]","cnfg_dir name is too long");

   random_flds(6,fs);
   sprintf(flds,"%s/test.fld",cnfg_dir);
   export_msfld(flds,3,fs);
   import_msfld(flds,3,fs+3);

   ie=cmp_flds(3,fs,fs+3);
   error(ie!=0,1,"main [check7.c]",
         "The observable fields are not properly restored");

   if (my_rank==0)
   {
      printf("Block-export random fields to the files\n"
             "%s/*/*/test.fld_b*\n",block_dir);
      printf("Then read the fields from there and compare with the saved "
             "fields.\n\n");
   }

   nl[0]=N0;
   nl[1]=N1;
   nl[2]=N2;
   nl[3]=N3;
   ib=blk_index(nl,bs,&nb);

   if (my_rank==0)
   {
      printf("Block size = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
      printf("Number of blocks = %d\n",nb);
      printf("nio_nodes = %d\n",nio_nodes);
      printf("nio_streams = %d\n\n",nio_streams);
   }

   error_root((nb%nio_nodes)!=0,1,"main [check7.c]",
              "nb must be an integer multiple of nio_nodes");
   error((ib<0)||(ib>=nb),1,"main [check7.c]","Unexpected block index");
   n=ib/(nb/nio_nodes);
   b=ib%(nb/nio_nodes);
   set_nio_streams(nio_streams);

   error(name_size("%s/%d/%d/test.fld_b%d",block_dir,nio_nodes-1,
                   nb/nio_nodes-1,nb-1)>=NAME_SIZE,1,
         "main [check7.c]","block_dir name is too long");

   sprintf(blk_sub_dir,"%s/%d/%d",block_dir,n,b);
   if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
       ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0))
      check_dir(blk_sub_dir);
   sprintf(flds,"%s/test.fld_b%d",blk_sub_dir,ib);

   random_flds(6,fs);
   blk_export_msfld(bs,flds,3,fs);
   blk_import_msfld(flds,3,fs+3);

   ie=cmp_flds(3,fs,fs+3);
   error(ie!=0,1,"main [check7.c]",
         "The observable fields are not properly restored");

   if (my_rank==0)
   {
      printf("No errors detected --- the fields are correctly exported\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
