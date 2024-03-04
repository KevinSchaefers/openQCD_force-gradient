
/*******************************************************************************
*
* File check8.c
*
* Copyright (C) 2017, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Importing observable fields previously exported by check7.
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
#include "linalg.h"
#include "archive.h"
#include "msfcts.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int ns=0;
static double snrm0[3],snrm1[3],**fs;
static array_t *afs;


static void alloc_flds(int n)
{
   size_t nd[2];

   if (n>ns)
   {
      if (ns>0)
         free_array(afs);

      nd[0]=n;
      nd[1]=VOLUME;
      afs=alloc_array(2,nd,sizeof(double),0);
      fs=(double**)((*afs).a);
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


static double square_norm(double *f)
{
   qflt rqsm;

   rqsm=vnorm_square_dble(VOLUME_TRD/2,3,(complex_dble*)(f));

   return rqsm.q[0];
}


int main(int argc,char *argv[])
{
   int my_rank,n,b,ir,ie,i;
   int nio_nodes,nio_streams,nb,ib,nl[4],bs[4],bo[4];
   stdint_t l[13];
   char cnfg_dir[NAME_SIZE],block_dir[NAME_SIZE];
   char blk_sub_dir[NAME_SIZE],flds[NAME_SIZE];
   FILE *flog=NULL,*fin=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check8.log","w",stdout);
      fin=freopen("check7.in","r",stdin);

      printf("\n");
      printf("Importing observable fields exported by check7\n");
      printf("----------------------------------------------\n\n");

      print_lattice_sizes();

      read_line("cnfg_dir","%s\n",cnfg_dir);
      read_line("block_dir","%s",block_dir);
      read_line("bs","%d %d %d %d",bo,bo+1,bo+2,bo+3);
      read_line("nio_nodes","%d",&nio_nodes);
      read_line("nio_streams","%d",&nio_streams);
      fclose(fin);
   }

   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(block_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(bo,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nio_nodes,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nio_streams,1,MPI_INT,0,MPI_COMM_WORLD);

   start_ranlux(0,123456);
   geometry();
   alloc_flds(6);

   error(name_size("%s/test.fld",cnfg_dir)>=NAME_SIZE,1,
         "main [check8.c]","cnfg_dir name is too long");
   sprintf(flds,"%s/test.fld",cnfg_dir);
   random_flds(6,fs);

   for (i=0;i<3;i++)
      snrm0[i]=square_norm(fs[i]);

   if (my_rank==0)
   {
      fin=fopen(flds,"rb");
      error_root(fin==NULL,1,"main [check8.c]","Unable to open input file");

      ir=fread(l,sizeof(stdint_t),5,fin);
      ir+=fread(snrm1,sizeof(double),3,fin);
      error_root(ir!=8,1,"main [check8.c]","Incorrect read count");
      fclose(fin);

      if (endianness()==BIG_ENDIAN)
      {
         bswap_int(5,l);
         bswap_double(3,snrm1);
      }

      printf("Read %d exported fields from file\n"
             "%s:\n\n",(int)(l[4]),flds);
      printf("Lattice size = %d %d %d %d\n",
             (int)(l[0]),(int)(l[1]),(int)(l[2]),(int)(l[3]));
      printf("Square norms = % .15e, % .15e, % .15e\n",
             snrm1[0],snrm1[1],snrm1[2]);
      printf("Should be    = % .15e, % .15e, % .15e\n\n",
             snrm0[0],snrm0[1],snrm0[2]);

      ie=0;
      ie|=((int)(l[0])!=N0);
      ie|=((int)(l[1])!=N1);
      ie|=((int)(l[2])!=N2);
      ie|=((int)(l[3])!=N3);
      ie|=((int)(l[4])!=3);

      error_root(ie!=0,1,"main [check8.c]","Unexpected header data");
   }

   import_msfld(flds,3,fs+3);
   ie=cmp_flds(3,fs,fs+3);
   error(ie!=0,1,"main [check8.c]",
         "The observable fields are not properly restored");
   remove(flds);

   if (my_rank==0)
      printf("Read block-exported fields from the files\n"
             "%s/*/*/test.fld_b*\n\n",block_dir);

   error(name_size("%s/0/0/test.fld_b0",block_dir)>=NAME_SIZE,1,
         "main [check8.c]","block_dir name is too long");
   sprintf(flds,"%s/0/0/test.fld_b0",block_dir);
   blk_sizes(flds,nl,bs);
   ie=0;
   ie|=((nl[0]!=N0)||(bs[0]!=bo[0]));
   ie|=((nl[1]!=N1)||(bs[1]!=bo[1]));
   ie|=((nl[2]!=N2)||(bs[2]!=bo[2]));
   ie|=((nl[3]!=N3)||(bs[3]!=bo[3]));
   error(ie!=0,1,"main [check8.c]",
         "Block sizes determined by blk_sizes() are incorrect");

   ib=blk_index(nl,bs,&nb);
   n=ib/(nb/nio_nodes);
   b=ib%(nb/nio_nodes);
   set_nio_streams(nio_streams);

   if (my_rank==0)
   {
      printf("Lattice size = %d %d %d %d\n",nl[0],nl[1],nl[2],nl[3]);
      printf("Block size = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
      printf("Number of blocks = %d\n",nb);
   }

   error(name_size("%s/%d/%d/test_b%d",block_dir,nio_nodes-1,
                   nb/nio_nodes-1,nb-1)>=NAME_SIZE,1,
         "main [check8.c]","block_dir name is too long");
   sprintf(blk_sub_dir,"%s/%d/%d",block_dir,n,b);
   random_flds(6,fs);

   for (i=0;i<3;i++)
      snrm0[i]=square_norm(fs[i]);

   if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
       ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0))
   {
      sprintf(flds,"%s/test.fld_b%d",blk_sub_dir,ib);
      fin=fopen(flds,"rb");
      error_loc(fin==NULL,1,"main [check8.c]","Unable to open input file");

      ir=fread(l,sizeof(stdint_t),13,fin);
      ir+=fread(snrm1,sizeof(double),3,fin);
      error_loc(ir!=16,1,"main [check8.c]","Incorrect read count");
      fclose(fin);

      if (endianness()==BIG_ENDIAN)
      {
         bswap_int(13,l);
         bswap_double(3,snrm1);
      }

      bo[0]=cpr[0]*L0;
      bo[1]=cpr[1]*L1;
      bo[2]=cpr[2]*L2;
      bo[3]=cpr[3]*L3;

      ie=0;
      ie|=(((int)(l[0])!=nl[0])||((int)(l[4])!=bs[0])||((int)(l[8])!=bo[0]));
      ie|=(((int)(l[1])!=nl[1])||((int)(l[5])!=bs[1])||((int)(l[9])!=bo[1]));
      ie|=(((int)(l[2])!=nl[2])||((int)(l[6])!=bs[2])||((int)(l[10])!=bo[2]));
      ie|=(((int)(l[3])!=nl[3])||((int)(l[7])!=bs[3])||((int)(l[11])!=bo[3]));
      ie|=((int)(l[12])!=3);

      error_loc(ie!=0,1,"main [check8.c]","Unexpected file header data");
   }

   blk_import_msfld(flds,3,fs+3);
   ie=cmp_flds(3,fs,fs+3);
   error(ie!=0,1,"main [check8.c]",
         "The observable fields are not properly restored");

   if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
       ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0))
      remove(flds);

   for (i=0;i<3;i++)
      snrm1[i]=square_norm(fs[i+3]);

   if (my_rank==0)
   {
      printf("Square norms = % .15e, % .15e, % .15e\n",
             snrm1[0],snrm1[1],snrm1[2]);
      printf("Should be    = % .15e, % .15e, % .15e\n\n",
             snrm0[0],snrm0[1],snrm0[2]);

      printf("No errors detected -- the fields are correctly restored\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
