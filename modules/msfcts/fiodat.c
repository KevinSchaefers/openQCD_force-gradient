
/*******************************************************************************
*
* File fiodat.c
*
* Copyright (C) 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* I/O utility programs for real-valued observable fields.
*
*   void read_msflds(iodat_t *iodat,char *cnfg,int n,double **f)
*     Reads field configuration files according to the specified I/O data
*     iodat and assigns the fields to f[0],..,f[n-1]. The configuration file
*     name "cnfg" is assumed to be like the one for exported observable fields
*     and is complemented as required for other field and storage types.
*
*   void save_msflds(int icnfg,char *nbase,iodat_t *iodat,int n,double **f)
*     Saves the fields f[0],..,f[n-1] to disk according to the specified I/O
*     data iodat. The configuration file name is set to "nbase"n"icnfg".fld,
*     complemented as required for the different storage types.
*
* See archive/README.iodat for a description of the storage formats, the
* associated configuration directory structures and the iodat_t structures.
*
* For the programs read_msflds() and save_msflds() to work correctly, the
* iodat_t structures must be properly initialized. This is guaranteed if the
* parameters are read from a file, using read_iodat(), and if the contents of
* the structure are checked by running check_iodat() before any fields are
* read or written (see archive/iodat.c).
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define FIODAT_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"
#include "utils.h"
#include "archive.h"
#include "msfcts.h"
#include "global.h"


void read_msflds(iodat_t *iodat,char *cnfg,int n,double **f)
{
   int my_rank,types;
   int m,ib;
   double wt1,wt2;
   char cnfg_file[NAME_SIZE];
   char *cnfg_dir,*block_dir,*local_dir;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   MPI_Barrier(MPI_COMM_WORLD);
   wt1=MPI_Wtime();

   types=iodat[0].types;

   if (types&0x1)
   {
      cnfg_dir=iodat[0].cnfg_dir;

      error_loc(name_size("%s/%s.fld",cnfg_dir,cnfg)>=NAME_SIZE,1,
                "read_msflds [fiodat.c]","File name is too long");
      sprintf(cnfg_file,"%s/%s.fld",cnfg_dir,cnfg);
      import_msfld(cnfg_file,n,f);
   }
   else if (types&0x2)
   {
      if (iodat[0].nb==0)
         check_iodat(iodat,"i",0x8,cnfg);

      set_nio_streams(iodat[0].nio_streams);
      block_dir=iodat[0].block_dir;
      ib=iodat[0].ib;
      m=iodat[0].nb/iodat[0].nio_nodes;

      error_loc(name_size("%s/%d/%d/%s.fld_b%d",
                          block_dir,ib/m,ib%m,cnfg,ib)>=NAME_SIZE,1,
                "read_msflds [fiodat.c]","File name is too long");
      sprintf(cnfg_file,"%s/%d/%d/%s.fld_b%d",
              block_dir,ib/m,ib%m,cnfg,ib);
      blk_import_msfld(cnfg_file,n,f);
   }
   else
   {
      set_nio_streams(iodat[0].nio_streams);
      local_dir=iodat[0].local_dir;
      m=NPROC/iodat[0].nio_nodes;

      error_loc(name_size("%s/%d/%d/%s.fld_%d",
                          local_dir,my_rank/m,my_rank%m,cnfg,my_rank)
                >=NAME_SIZE,1,"read_msflds [fiodat.c]","File name is too long");
      sprintf(cnfg_file,"%s/%d/%d/%s.fld_%d",
              local_dir,my_rank/m,my_rank%m,cnfg,my_rank);
      read_msfld(cnfg_file,n,f);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   wt2=MPI_Wtime();

   if (my_rank==0)
   {
      printf("Observable fields %s read from disk in %.2e sec\n",
             cnfg,wt2-wt1);
      fflush(stdout);
   }
}


void save_msflds(int icnfg,char *nbase,iodat_t *iodat,int n,double **f)
{
   int my_rank,types;
   int m,ib,*bs;
   double wt1,wt2;
   char cnfg_file[NAME_SIZE];
   char *cnfg_dir,*block_dir,*local_dir;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   MPI_Barrier(MPI_COMM_WORLD);
   wt1=MPI_Wtime();

   types=iodat[0].types;

   if (types&0x1)
   {
      cnfg_dir=iodat[0].cnfg_dir;

      error_loc(name_size("%s/%sn%d.fld",cnfg_dir,nbase,icnfg)
                >=NAME_SIZE,1,"save_msflds [fiodat.c]","File name is too long");
      sprintf(cnfg_file,"%s/%sn%d.fld",cnfg_dir,nbase,icnfg);
      export_msfld(cnfg_file,n,f);
   }

   if (types&0x2)
   {
      set_nio_streams(iodat[0].nio_streams);
      block_dir=iodat[0].block_dir;
      bs=iodat[0].bs;
      ib=iodat[0].ib;
      m=iodat[0].nb/iodat[0].nio_nodes;

      error_loc(name_size("%s/%d/%d/%sn%d.fld_b%d",
                          block_dir,ib/m,ib%m,nbase,icnfg,ib)
                >=NAME_SIZE,1,"save_msflds [fiodat.c]","File name is too long");
      sprintf(cnfg_file,"%s/%d/%d/%sn%d.fld_b%d",
              block_dir,ib/m,ib%m,nbase,icnfg,ib);
      blk_export_msfld(bs,cnfg_file,n,f);
   }

   if (types&0x4)
   {
      set_nio_streams(iodat[0].nio_streams);
      local_dir=iodat[0].local_dir;
      m=NPROC/iodat[0].nio_nodes;

      error_loc(name_size("%s/%d/%d/%sn%d.fld_%d",local_dir,
                          my_rank/m,my_rank%m,nbase,icnfg,my_rank)
                >=NAME_SIZE,1,"save_msflds [fiodat.c]","File name is too long");
      sprintf(cnfg_file,"%s/%d/%d/%sn%d.fld_%d",
              local_dir,my_rank/m,my_rank%m,nbase,icnfg,my_rank);
      write_msfld(cnfg_file,n,f);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   wt2=MPI_Wtime();

   if (my_rank==0)
   {
      printf("Observable field configuration no %d ",icnfg);

      if (types==0x1)
         printf("exported");
      else if (types==0x2)
         printf("block-exported");
      else if (types==0x3)
         printf("exported and block-exported");
      else if (types==0x4)
         printf("locally stored");
      else if (types==0x5)
         printf("exported and locally stored");
      else if (types==0x6)
         printf("block-exported and locally stored");
      else
         printf("exported, block-exported and locally stored");

      printf(" in %.2e sec\n\n",wt2-wt1);
   }
}
