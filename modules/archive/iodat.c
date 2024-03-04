
/*******************************************************************************
*
* File iodat.c
*
* Copyright (C) 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* I/O utility programs for gauge, momentum and pseudo-fermion fields.
*
*   void read_iodat(char *section,char *io,iodat_t *iodat)
*     Reads configuration I/O data on MPI process 0 from the specified
*     parameter section on stdin. Depending on whether io="i" or io="o",
*     the parameters are assumed to be for configuration input or output.
*     On exit the I/O data are stored in the structure iodat.
*
*   void print_iodat(char *io,iodat_t *iodat)
*     Prints storage information contained in the structure iodat to
*     stdout on MPI process 0. Depending on whether io="i" or io="o",
*     the parameters are assumed to be for configuration input or output.
*
*   void read_flds(iodat_t *iodat,char *cnfg,int mask,int ioflg)
*     Reads field configuration files according to the specified I/O data
*     iodat, the periodic-extension mask and the field-selection flag ioflg.
*     The configuration file name "cnfg" is assumed to be like the one for
*     exported gauge fields and is complemented as required for other field
*     and storage types.
*
*   void save_flds(int icnfg,char *nbase,iodat_t *iodat,int ioflg)
*     Saves fields to disk according to the specified I/O data iodat and
*     field-selection flag ioflg. The configuration file names are set to
*     "nbase"n"icnfg", complemented as required for the different field
*     and storage types.
*
*   void remove_flds(int icnfg,char *nbase,iodat_t *iodat,int ioflg)
*     Removes the gauge, momentum and pseudo-fermion configuration files
*     according to the specified I/O data iodat and field-selection flag
*     ioflg. The file names are set to "nbase"n"icnfg", complemented as
*     required for the different field and storage types.
*
*   void check_iodat(iodat_t *iodat,char *io,int ioflg,char *cnfg)
*     Checks whether field configuration files can be read (io="i") or
*     saved (io="o") according to the specified I/O data iodat and field-
*     selection flag ioflg. The configuration file name "cnfg" is assumed
*     to be like the one for exported gauge fields and is complemented as
*     required for other field and storage types. The program includes
*     accessibility, name-size and I/O data compatibility checks.
*
*   void write_iodat_parms(FILE *fdat,iodat_t *iodat)
*     Writes the parameters contained in the structure iodat to the file
*     fdat on MPI process 0. The parameters nb,ib and the directory names
*     are not written.
*
*   void check_iodat_parms(FILE *fdat,iodat_t *iodat)
*     Compares the parameters contained in the structure iodat with the
*     values stored on the file fdat on MPI process 0, assuming the latter
*     were written to the file by the program write_iodat_parms().
*
* See README.iodat for a description of the storage formats, the associated
* configuration directory structures, the contents of the iodat_t structures
* and of the parameter sections expected by the program read_iodat().
*
* The bits of the field selection flag ioflg select the type of fields to
* be considered:
*
*   ioflg&0x1     Selects the global double-precision gauge field. For
*                 exported storage format, there is no file name extension
*                 in this case.
*
*   ioflg&0x2     Selects the momentum field (mdflds/mdflds.c) The file
*                 name extension is ".mom".
*
*   ioflg&0x4     Selects the allocated pseudo-fermion fields. The file
*                 name extension is ".pf<in>", where <int> is the number
*                 of the field counted from 0 in steps of 1.
*
*   ioflg&0x8     Selects real-valued observable fields. The file name
*                 extension is ".fld". This bit of ioflg is only supported
*                 by remove_flds() and check_iodat(). See msfcts/fiodat.c
*                 and msfcts/farchive.c for programs to read and write
*                 these fields.
*
* For the programs read_flds(), save_flds() and remove_flds() to function
* correctly, the iodat_t structures must be properly initialized. This is
* guaranteed if the parameters are read from a file, using read_iodat(),
* and if the contents of the structure are checked by running check_iodat()
* before any field configurations are read, written or removed.
*
* The periodic extension of input gauge configurations and the associated
* extension masks are explained on the top of the module archive.c.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define IODAT_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "mdflds.h"
#include "archive.h"
#include "global.h"


void read_iodat(char *section,char *io,iodat_t *iodat)
{
   int my_rank,types,nion,nios,bs[4];
   char *cnfg_dir,*block_dir,*local_dir;
   char line[NAME_SIZE];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      error_root((strlen(io)!=1)||
                 ((strchr(io,'i')==NULL)&&(strchr(io,'o')==NULL)),1,
                 "read_iodat [iodat.c]","Improper parameter 'io'");

      find_section(section);

      if (strchr(io,'i')!=NULL)
         read_line("type","%s",line);
      else
         read_line("types","%s",line);

      types=0x0;
      if (strchr(line,'e')!=NULL)
         types|=0x1;
      if (strchr(line,'b')!=NULL)
         types|=0x2;
      if (strchr(line,'l')!=NULL)
         types|=0x4;

      error_root((strchr(io,'i')!=NULL)&&
                 (types!=0x1)&&(types!=0x2)&&(types!=0x4),1,
                 "read_iodat [iodat.c]","Improper storage type");
      error_root(types==0x0,1,
                 "read_iodat [iodat.c]","Improper storage types");
   }

   MPI_Bcast(&types,1,MPI_INT,0,MPI_COMM_WORLD);

   if (types&0x1)
   {
      cnfg_dir=malloc(NAME_SIZE*sizeof(*cnfg_dir));
      error(cnfg_dir==NULL,1,"read_iodat [iodat.c]",
            "Unable to read configuration directory");
      if (my_rank==0)
         read_line("cnfg_dir","%s",cnfg_dir);
      MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   }
   else
      cnfg_dir=NULL;

   if (types&0x2)
   {
      block_dir=malloc(NAME_SIZE*sizeof(*block_dir));
      error(block_dir==NULL,2,"read_iodat [iodat.c]",
            "Unable to read configuration directory");
      if (my_rank==0)
         read_line("block_dir","%s",block_dir);
      MPI_Bcast(block_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   }
   else
      block_dir=NULL;

   if (types&0x4)
   {
      local_dir=malloc(NAME_SIZE*sizeof(*local_dir));
      error(local_dir==NULL,3,"read_iodat [iodat.c]",
            "Unable to read configuration directory");
      if (my_rank==0)
         read_line("local_dir","%s",local_dir);
      MPI_Bcast(local_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   }
   else
      local_dir=NULL;

   if (my_rank==0)
   {
      if ((strchr(io,'o')!=NULL)&&(types&0x2))
         read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      else
      {
         bs[0]=0;
         bs[1]=0;
         bs[2]=0;
         bs[3]=0;
      }

      if (types&0x6)
      {
         read_line("nio_nodes","%d",&nion);
         read_line("nio_streams","%d",&nios);
      }
      else
      {
         nion=1;
         nios=0;
      }
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nion,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nios,1,MPI_INT,0,MPI_COMM_WORLD);

   iodat[0].types=types;
   iodat[0].nio_nodes=nion;
   iodat[0].nio_streams=nios;
   iodat[0].nb=0;
   iodat[0].ib=NPROC;
   iodat[0].bs[0]=bs[0];
   iodat[0].bs[1]=bs[1];
   iodat[0].bs[2]=bs[2];
   iodat[0].bs[3]=bs[3];
   iodat[0].cnfg_dir=cnfg_dir;
   iodat[0].block_dir=block_dir;
   iodat[0].local_dir=local_dir;
}


void print_iodat(char *io,iodat_t *iodat)
{
   int my_rank,types;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      error_root((strlen(io)!=1)||
                 ((strchr(io,'i')==NULL)&&(strchr(io,'o')==NULL)),1,
                 "print_iodat [iodat.c]","Improper parameter 'io'");

      if (strchr(io,'i')!=NULL)
         printf("Input configuration storage type = ");
      else
         printf("Output configuration storage types = ");

      types=iodat[0].types;

      if (types&0x1)
         printf("exported");
      if (types&0x2)
      {
         if (types&0x1)
            printf(" and ");
         printf("block-exported");
      }
      if (types&0x4)
      {
         if (types&0x3)
            printf(" and ");
         printf("local");
      }

      printf("\n");

      if (types&0x2)
      {
         printf("Block size = %dx%dx%dx%d\n",
                iodat[0].bs[0],iodat[0].bs[1],iodat[0].bs[2],iodat[0].bs[3]);
      }

      if (types&0x6)
      {
         printf("Parallel I/O parameters: "
                "nio_nodes = %d, nio_streams = %d\n",
                iodat[0].nio_nodes,iodat[0].nio_streams);
      }

      printf("\n");
   }
}


void read_flds(iodat_t *iodat,char *cnfg,int mask,int ioflg)
{
   int my_rank,types,npf,ipf,k,*eo;
   int n,ib;
   double wt1,wt2;
   spinor_dble **pf;
   mdflds_t *mdfs;
   char cnfg_file[NAME_SIZE];
   char *cnfg_dir,*block_dir,*local_dir;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   MPI_Barrier(MPI_COMM_WORLD);
   wt1=MPI_Wtime();

   types=iodat[0].types;
   mdfs=mdflds();
   npf=(*mdfs).npf;
   eo=(*mdfs).eo;
   pf=(*mdfs).pf;

   if (types&0x1)
   {
      cnfg_dir=iodat[0].cnfg_dir;

      if (ioflg&0x1)
      {
         error_loc(name_size("%s/%s",cnfg_dir,cnfg)>=NAME_SIZE,1,
                   "read_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%s",cnfg_dir,cnfg);
         import_cnfg(cnfg_file,mask);
      }

      if (ioflg&0x2)
      {
         error_loc(name_size("%s/%s.mom",cnfg_dir,cnfg)>=NAME_SIZE,1,
                   "read_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%s.mom",cnfg_dir,cnfg);
         import_mfld(cnfg_file);
      }

      if (ioflg&0x4)
      {
         error_loc(name_size("%s/%s.pf%d",cnfg_dir,cnfg,npf-1)>=NAME_SIZE,1,
                   "read_flds [iodat.c]","File name is too long");
         k=0;

         for (ipf=0;ipf<npf;ipf++)
         {
            if (pf[ipf]!=NULL)
            {
               sprintf(cnfg_file,"%s/%s.pf%d",cnfg_dir,cnfg,k);
               import_sfld(cnfg_file,eo[ipf],pf[ipf]);
               k+=1;
            }
         }
      }
   }
   else if (types&0x2)
   {
      if (iodat[0].nb==0)
         check_iodat(iodat,"i",ioflg,cnfg);

      set_nio_streams(iodat[0].nio_streams);
      block_dir=iodat[0].block_dir;
      ib=iodat[0].ib;
      n=iodat[0].nb/iodat[0].nio_nodes;

      if (ioflg&0x1)
      {
         error_loc(name_size("%s/%d/%d/%s_b%d",
                             block_dir,ib/n,ib%n,cnfg,ib)>=NAME_SIZE,1,
                   "read_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%d/%d/%s_b%d",
                 block_dir,ib/n,ib%n,cnfg,ib);
         blk_import_cnfg(cnfg_file,mask);
      }

      if (ioflg&0x2)
      {
         error_loc(name_size("%s/%d/%d/%s.mom_b%d",
                             block_dir,ib/n,ib%n,cnfg,ib)>=NAME_SIZE,1,
                   "read_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%d/%d/%s.mom_b%d",
                 block_dir,ib/n,ib%n,cnfg,ib);
         blk_import_mfld(cnfg_file);
      }

      if (ioflg&0x4)
      {
         error_loc(name_size("%s/%d/%d/%s.pf%d_b%d",
                             block_dir,ib/n,ib%n,cnfg,npf-1,ib)>=NAME_SIZE,1,
                   "read_flds [iodat.c]","File name is too long");
         k=0;

         for (ipf=0;ipf<npf;ipf++)
         {
            if (pf[ipf]!=NULL)
            {
               sprintf(cnfg_file,"%s/%d/%d/%s.pf%d_b%d",
                       block_dir,ib/n,ib%n,cnfg,k,ib);
               blk_import_sfld(cnfg_file,eo[ipf],pf[ipf]);
               k+=1;
            }
         }
      }
   }
   else
   {
      set_nio_streams(iodat[0].nio_streams);
      local_dir=iodat[0].local_dir;
      n=NPROC/iodat[0].nio_nodes;

      if (ioflg&0x1)
      {
         error_loc(name_size("%s/%d/%d/%s_%d",
                             local_dir,my_rank/n,my_rank%n,cnfg,my_rank)
                   >=NAME_SIZE,1,"read_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%d/%d/%s_%d",
                 local_dir,my_rank/n,my_rank%n,cnfg,my_rank);
         read_cnfg(cnfg_file);
      }

      if (ioflg&0x2)
      {
         error_loc(name_size("%s/%d/%d/%s.mom_%d",
                             local_dir,my_rank/n,my_rank%n,cnfg,my_rank)
                   >=NAME_SIZE,1,"read_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%d/%d/%s.mom_%d",
                 local_dir,my_rank/n,my_rank%n,cnfg,my_rank);
         read_mfld(cnfg_file);
      }

      if (ioflg&0x4)
      {
         error_loc(name_size("%s/%d/%d/%s.pf%d_%d",
                             local_dir,my_rank/n,my_rank%n,cnfg,npf-1,my_rank)
                   >=NAME_SIZE,1,"read_flds [iodat.c]","File name is too long");
         k=0;

         for (ipf=0;ipf<npf;ipf++)
         {
            if (pf[ipf]!=NULL)
            {
               sprintf(cnfg_file,"%s/%d/%d/%s.pf%d_%d",
                       local_dir,my_rank/n,my_rank%n,cnfg,k,my_rank);
               read_sfld(cnfg_file,eo[ipf],pf[ipf]);
               k+=1;
            }
         }
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
   wt2=MPI_Wtime();

   if (my_rank==0)
   {
      printf("Configuration %s read from disk in %.2e sec\n",
             cnfg,wt2-wt1);
      fflush(stdout);
   }
}


void save_flds(int icnfg,char *nbase,iodat_t *iodat,int ioflg)
{
   int my_rank,types,ipf,npf,k,*eo;
   int n,ib,ie,*bs;
   double wt1,wt2;
   spinor_dble **pf;
   mdflds_t *mdfs;
   char cnfg_file[NAME_SIZE];
   char *cnfg_dir,*block_dir,*local_dir;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   MPI_Barrier(MPI_COMM_WORLD);
   wt1=MPI_Wtime();

   ie=(query_flags(UD_PHASE_SET)||(check_bc(0.0)==0));
   error_root(ie!=0,1,"save_flds [iodat.c]",
              "Phase-modified field or unexpected boundary values");
   types=iodat[0].types;
   mdfs=mdflds();
   npf=(*mdfs).npf;
   eo=(*mdfs).eo;
   pf=(*mdfs).pf;

   if (types&0x1)
   {
      cnfg_dir=iodat[0].cnfg_dir;

      if (ioflg&0x1)
      {
         error_loc(name_size("%s/%sn%d",cnfg_dir,nbase,icnfg)
                   >=NAME_SIZE,1,"save_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
         export_cnfg(cnfg_file);
      }

      if (ioflg&0x2)
      {
         error_loc(name_size("%s/%sn%d.mom",cnfg_dir,nbase,icnfg)
                   >=NAME_SIZE,1,"save_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%sn%d.mom",cnfg_dir,nbase,icnfg);
         export_mfld(cnfg_file);
      }

      if (ioflg&0x4)
      {
         error_loc(name_size("%s/%sn%d.pf%d",cnfg_dir,nbase,icnfg,npf-1)
                   >=NAME_SIZE,1,"save_flds [iodat.c]","File name is too long");
         k=0;

         for (ipf=0;ipf<npf;ipf++)
         {
            if (pf[ipf]!=NULL)
            {
               sprintf(cnfg_file,"%s/%sn%d.pf%d",cnfg_dir,nbase,icnfg,k);
               export_sfld(cnfg_file,eo[ipf],pf[ipf]);
               k+=1;
            }
         }
      }
   }

   if (types&0x2)
   {
      set_nio_streams(iodat[0].nio_streams);
      block_dir=iodat[0].block_dir;
      bs=iodat[0].bs;
      ib=iodat[0].ib;
      n=iodat[0].nb/iodat[0].nio_nodes;

      if (ioflg&0x1)
      {
         error_loc(name_size("%s/%d/%d/%sn%d_b%d",
                             block_dir,ib/n,ib%n,nbase,icnfg,ib)
                   >=NAME_SIZE,1,"save_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%d/%d/%sn%d_b%d",
                 block_dir,ib/n,ib%n,nbase,icnfg,ib);
         blk_export_cnfg(bs,cnfg_file);
      }

      if (ioflg&0x2)
      {
         error_loc(name_size("%s/%d/%d/%sn%d.mom_b%d",
                             block_dir,ib/n,ib%n,nbase,icnfg,ib)
                   >=NAME_SIZE,1,"save_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%d/%d/%sn%d.mom_b%d",
                 block_dir,ib/n,ib%n,nbase,icnfg,ib);
         blk_export_mfld(bs,cnfg_file);
      }

      if (ioflg&0x4)
      {
         error_loc(name_size("%s/%d/%d/%sn%d.pf%d_b%d",
                             block_dir,ib/n,ib%n,nbase,icnfg,npf-1,ib)
                   >=NAME_SIZE,1,"save_flds [iodat.c]","File name is too long");
         k=0;

         for (ipf=0;ipf<npf;ipf++)
         {
            if (pf[ipf]!=NULL)
            {
               sprintf(cnfg_file,"%s/%d/%d/%sn%d.pf%d_b%d",
                       block_dir,ib/n,ib%n,nbase,icnfg,k,ib);
               blk_export_sfld(bs,cnfg_file,eo[ipf],pf[ipf]);
               k+=1;
            }
         }
      }
   }

   if (types&0x4)
   {
      set_nio_streams(iodat[0].nio_streams);
      local_dir=iodat[0].local_dir;
      n=NPROC/iodat[0].nio_nodes;

      if (ioflg&0x1)
      {
         error_loc(name_size("%s/%d/%d/%sn%d_%d",local_dir,
                             my_rank/n,my_rank%n,nbase,icnfg,my_rank)
                   >=NAME_SIZE,1,"save_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%d/%d/%sn%d_%d",
                 local_dir,my_rank/n,my_rank%n,nbase,icnfg,my_rank);
         write_cnfg(cnfg_file);
      }

      if (ioflg&0x2)
      {
         error_loc(name_size("%s/%d/%d/%sn%d.mom_%d",local_dir,
                             my_rank/n,my_rank%n,nbase,icnfg,my_rank)
                   >=NAME_SIZE,1,"save_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%d/%d/%sn%d.mom_%d",
                 local_dir,my_rank/n,my_rank%n,nbase,icnfg,my_rank);
         write_mfld(cnfg_file);
      }

      if (ioflg&0x4)
      {
         error_loc(name_size("%s/%d/%d/%sn%d.pf%d_%d",local_dir,
                             my_rank/n,my_rank%n,nbase,icnfg,npf-1,my_rank)
                   >=NAME_SIZE,1,"save_flds [iodat.c]","File name is too long");
         k=0;

         for (ipf=0;ipf<npf;ipf++)
         {
            if (pf[ipf]!=NULL)
            {
               sprintf(cnfg_file,"%s/%d/%d/%sn%d.pf%d_%d",
                       local_dir,my_rank/n,my_rank%n,nbase,icnfg,k,my_rank);
               write_sfld(cnfg_file,eo[ipf],pf[ipf]);
               k+=1;
            }
         }
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
   wt2=MPI_Wtime();

   if (my_rank==0)
   {
      printf("Configuration no %d ",icnfg);

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


static int get_npf(void)
{
   int npf,ipf,n;
   mdflds_t *mdfs;
   spinor_dble **pf;

   mdfs=mdflds();
   npf=(*mdfs).npf;
   pf=(*mdfs).pf;
   n=0;

   for (ipf=0;ipf<npf;ipf++)
   {
      if (pf[ipf]!=NULL)
         n+=1;
   }

   return n;
}


void remove_flds(int icnfg,char *nbase,iodat_t *iodat,int ioflg)
{
   int my_rank,types,npf,ipf;
   int ib,n,m,i,ip,*bs;
   char cnfg_file[NAME_SIZE];
   char *cnfg_dir,*block_dir,*local_dir;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   types=iodat[0].types;
   npf=get_npf();

   if ((types&0x1)&&(my_rank==0))
   {
      cnfg_dir=iodat[0].cnfg_dir;

      if (ioflg&0x1)
      {
         error_loc(name_size("%s/%sn%d",cnfg_dir,nbase,icnfg)
                   >=NAME_SIZE,1,
                   "remove_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
         remove(cnfg_file);
      }

      if (ioflg&0x2)
      {
         error_loc(name_size("%s/%sn%d.mom",cnfg_dir,nbase,icnfg)
                   >=NAME_SIZE,1,
                   "remove_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%sn%d.mom",cnfg_dir,nbase,icnfg);
         remove(cnfg_file);
      }

      if (ioflg&0x4)
      {
         error_loc(name_size("%s/%sn%d.pf%d",cnfg_dir,nbase,icnfg,npf-1)
                   >=NAME_SIZE,1,
                   "remove_flds [iodat.c]","File name is too long");

         for (ipf=0;ipf<npf;ipf++)
         {
            sprintf(cnfg_file,"%s/%sn%d.pf%d",cnfg_dir,nbase,icnfg,ipf);
            remove(cnfg_file);
         }
      }

      if (ioflg&0x8)
      {
         error_loc(name_size("%s/%sn%d.fld",cnfg_dir,nbase,icnfg)
                   >=NAME_SIZE,1,
                   "remove_flds [iodat.c]","File name is too long");
         sprintf(cnfg_file,"%s/%sn%d.fld",cnfg_dir,nbase,icnfg);
         remove(cnfg_file);
      }
   }

   if (types&0x2)
   {
      block_dir=iodat[0].block_dir;
      bs=iodat[0].bs;
      ib=iodat[0].ib;
      n=iodat[0].nb/iodat[0].nio_nodes;
      m=iodat[0].nb/iodat[0].nio_streams;

      ip=((((cpr[0]*L0)%bs[0])==0)&&(((cpr[1]*L1)%bs[1])==0)&&
          (((cpr[2]*L2)%bs[2])==0)&&(((cpr[3]*L3)%bs[3])==0));

      for (i=0;i<m;i++)
      {
         MPI_Barrier(MPI_COMM_WORLD);

         if ((ip)&&((ib%m)==i))
         {
            if (ioflg&0x1)
            {
               error_loc(name_size("%s/%d/%d/%sn%d_b%d",
                                   block_dir,ib/n,ib%n,nbase,icnfg,ib)
                         >=NAME_SIZE,1,
                         "remove_flds [iodat.c]","File name is too long");
               sprintf(cnfg_file,"%s/%d/%d/%sn%d_b%d",
                       block_dir,ib/n,ib%n,nbase,icnfg,ib);
               remove(cnfg_file);
            }

            if (ioflg&0x2)
            {
               error_loc(name_size("%s/%d/%d/%sn%d.mom_b%d",
                                   block_dir,ib/n,ib%n,nbase,icnfg,ib)
                         >=NAME_SIZE,1,
                         "remove_flds [iodat.c]","File name is too long");
               sprintf(cnfg_file,"%s/%d/%d/%sn%d.mom_b%d",
                       block_dir,ib/n,ib%n,nbase,icnfg,ib);
               remove(cnfg_file);
            }

            if (ioflg&0x4)
            {
               error_loc(name_size("%s/%d/%d/%sn%d.pf%d_b%d",
                                   block_dir,ib/n,ib%n,nbase,icnfg,npf-1,ib)
                         >=NAME_SIZE,1,
                         "remove_flds [iodat.c]","File name is too long");

               for (ipf=0;ipf<npf;ipf++)
               {
                  sprintf(cnfg_file,"%s/%d/%d/%sn%d.pf%d_b%d",
                          block_dir,ib/n,ib%n,nbase,icnfg,ipf,ib);
                  remove(cnfg_file);
               }
            }

            if (ioflg&0x8)
            {
               error_loc(name_size("%s/%d/%d/%sn%d.fld_b%d",
                                   block_dir,ib/n,ib%n,nbase,icnfg,ib)
                         >=NAME_SIZE,1,
                         "remove_flds [iodat.c]","File name is too long");
               sprintf(cnfg_file,"%s/%d/%d/%sn%d.fld_b%d",
                       block_dir,ib/n,ib%n,nbase,icnfg,ib);
               remove(cnfg_file);
            }
         }
      }
   }

   if (types&0x4)
   {
      local_dir=iodat[0].local_dir;
      n=NPROC/iodat[0].nio_nodes;
      m=NPROC/iodat[0].nio_streams;

      for (i=0;i<m;i++)
      {
         MPI_Barrier(MPI_COMM_WORLD);

         if ((my_rank%m)==i)
         {
            if (ioflg&0x1)
            {
               error_loc(name_size("%s/%d/%d/%sn%d_%d",local_dir,
                                   my_rank/n,my_rank%n,nbase,icnfg,my_rank)
                         >=NAME_SIZE,1,
                         "remove_flds [iodat.c]","File name is too long");
               sprintf(cnfg_file,"%s/%d/%d/%sn%d_%d",local_dir,
                       my_rank/n,my_rank%n,nbase,icnfg,my_rank);
               remove(cnfg_file);
            }

            if (ioflg&0x2)
            {
               error_loc(name_size("%s/%d/%d/%sn%d.mom_%d",local_dir,
                                   my_rank/n,my_rank%n,nbase,icnfg,my_rank)
                         >=NAME_SIZE,1,
                         "remove_flds [iodat.c]","File name is too long");
               sprintf(cnfg_file,"%s/%d/%d/%sn%d.mom_%d",local_dir,
                       my_rank/n,my_rank%n,nbase,icnfg,my_rank);
               remove(cnfg_file);
            }

            if (ioflg&0x4)
            {
               error_loc(name_size("%s/%d/%d/%sn%d.pf%d_%d",local_dir,
                                   my_rank/n,my_rank%n,nbase,icnfg,npf-1,my_rank)
                         >=NAME_SIZE,1,
                         "remove_flds [iodat.c]","File name is too long");

               for (ipf=0;ipf<npf;ipf++)
               {
                  sprintf(cnfg_file,"%s/%d/%d/%sn%d.pf%d_%d",local_dir,
                          my_rank/n,my_rank%n,nbase,icnfg,ipf,my_rank);
                  remove(cnfg_file);
               }
            }

            if (ioflg&0x8)
            {
               error_loc(name_size("%s/%d/%d/%sn%d.fld_%d",local_dir,
                                   my_rank/n,my_rank%n,nbase,icnfg,my_rank)
                         >=NAME_SIZE,1,
                         "remove_flds [iodat.c]","File name is too long");
               sprintf(cnfg_file,"%s/%d/%d/%sn%d.fld_%d",local_dir,
                       my_rank/n,my_rank%n,nbase,icnfg,my_rank);
               remove(cnfg_file);
            }
         }
      }
   }
}


static void check_iodat0(iodat_t *iodat,int ioflg,char *cnfg)
{
   int my_rank,types,nion,nios,npf,n,m,i;
   int nb,ib,ns[4],bs[4];
   char cnfg_file[NAME_SIZE];
   char *cnfg_dir,*block_dir,*local_dir;
   FILE *fdat;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   cnfg_file[0]='0';
   types=iodat[0].types;
   nion=iodat[0].nio_nodes;
   nios=iodat[0].nio_streams;
   npf=get_npf();

   if (types&0x1)
   {
      if (my_rank==0)
      {
         cnfg_dir=iodat[0].cnfg_dir;

         if (ioflg&0x4)
            m=name_size("%s/%s.pf%d",cnfg_dir,cnfg,npf-1);
         else if ((ioflg&0x2)||(ioflg&0x8))
            m=name_size("%s/%s.mom",cnfg_dir,cnfg);
         else
            m=name_size("%s/%s",cnfg_dir,cnfg);

         error_root(m>=NAME_SIZE,1,"check_iodat0 [iodat.c]",
                    "cnfg_dir name is too long");

         for (i=0x1;i<=0x8;i<<=1)
         {
            if (i&ioflg)
            {
               if (i==(ioflg&0x1))
                  sprintf(cnfg_file,"%s/%s",cnfg_dir,cnfg);
               else if (i==(ioflg&0x2))
                  sprintf(cnfg_file,"%s/%s.mom",cnfg_dir,cnfg);
               else if (i==(ioflg&0x4))
                  sprintf(cnfg_file,"%s/%s.pf0",cnfg_dir,cnfg);
               else if (i==(ioflg&0x8))
                  sprintf(cnfg_file,"%s/%s.fld",cnfg_dir,cnfg);

               fdat=fopen(cnfg_file,"rb");
               error_root(fdat==NULL,1,"check_iodat0 [iodat.c]",
                          "Unable to access configuration file %s",cnfg_file);
               fclose(fdat);
            }
         }
      }
   }
   else if (types&0x2)
   {
      block_dir=iodat[0].block_dir;

      if ((cpr[0]==0)&&(cpr[1]==0)&&(cpr[2]==0)&&(cpr[3]==0))
      {
         if ((ioflg&0x2)||(ioflg&0x4)||(ioflg&0x8))
            m=name_size("%s/0/0/%s.mom_b0",block_dir,cnfg);
         else
            m=name_size("%s/0/0/%s_b0",block_dir,cnfg);

         error_loc(m>=NAME_SIZE,1,"check_iodat0 [iodat.c]",
                   "block_dir name is too long");
      }

      if (ioflg&0x1)
         sprintf(cnfg_file,"%s/0/0/%s_b0",block_dir,cnfg);
      else if (ioflg&0x2)
         sprintf(cnfg_file,"%s/0/0/%s.mom_b0",block_dir,cnfg);
      else if (ioflg&0x4)
         sprintf(cnfg_file,"%s/0/0/%s.pf0_b0",block_dir,cnfg);
      else
         sprintf(cnfg_file,"%s/0/0/%s.fld_b0",block_dir,cnfg);

      blk_sizes(cnfg_file,ns,bs);
      ib=blk_index(ns,bs,&nb);

      iodat[0].nb=nb;
      iodat[0].ib=ib;
      iodat[0].bs[0]=bs[0];
      iodat[0].bs[1]=bs[1];
      iodat[0].bs[2]=bs[2];
      iodat[0].bs[3]=bs[3];

      if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
          ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0)&&(ib<nb))
      {
         error_loc((nion<1)||((nb%nion)!=0)||(nios<1)||((nb%nios)!=0),1,
                   "check_iodat0 [iodat.c]",
                   "Improper choice of nio_nodes or nio_streams");
         n=nb/nion;

         if (ioflg&0x4)
            m=name_size("%s/%d/%d/%s.pf%d_b%d",
                        block_dir,ib/n,ib%n,cnfg,npf-1,ib);
         else if ((ioflg&0x2)||(ioflg&0x8))
            m=name_size("%s/%d/%d/%s.mom_b%d",
                        block_dir,ib/n,ib%n,cnfg,ib);
         else
            m=name_size("%s/%d/%d/%s_b%d",
                        block_dir,ib/n,ib%n,cnfg,ib);

         error_loc(m>=NAME_SIZE,1,"check_iodat0 [iodat.c]",
                   "block_dir name is too long");

         for (i=0x1;i<=0x8;i<<=1)
         {
            if (i&ioflg)
            {
               if (i==(ioflg&0x1))
                  sprintf(cnfg_file,"%s/%d/%d/%s_b%d",
                          block_dir,ib/n,ib%n,cnfg,ib);
               else if (i==(ioflg&0x2))
                  sprintf(cnfg_file,"%s/%d/%d/%s.mom_b%d",
                          block_dir,ib/n,ib%n,cnfg,ib);
               else if (i==(ioflg&0x4))
                  sprintf(cnfg_file,"%s/%d/%d/%s.pf0_b%d",
                          block_dir,ib/n,ib%n,cnfg,ib);
               else if (i==(ioflg&0x8))
                  sprintf(cnfg_file,"%s/%d/%d/%s.fld_b%d",
                          block_dir,ib/n,ib%n,cnfg,ib);

               fdat=fopen(cnfg_file,"rb");
               error_loc(fdat==NULL,1,"check_iodat0 [iodat.c]",
                         "Unable to access configuration file %s",cnfg_file);
               fclose(fdat);
            }
         }
      }
   }
   else if (types&0x4)
   {
      local_dir=iodat[0].local_dir;
      n=NPROC/nion;
      error_loc((nion<1)||((NPROC%nion)!=0)||(nios<1)||((NPROC%nios)!=0),1,
                "check_iodat0 [iodat.c]",
                "Improper choice of nio_nodes or nio_streams");

      if (ioflg&0x4)
         m=name_size("%s/%d/%d/%s.pf%d_%d",
                     local_dir,my_rank/n,my_rank%n,cnfg,npf-1,my_rank);
      else if ((ioflg&0x2)||(ioflg&0x8))
         m=name_size("%s/%d/%d/%s.mom_%d",
                     local_dir,my_rank/n,my_rank%n,cnfg,my_rank);
      else
         m=name_size("%s/%d/%d/%s_%d",
                     local_dir,my_rank/n,my_rank%n,cnfg,my_rank);

      error_loc(m>=NAME_SIZE,1,"check_iodat0 [iodat.c]",
                "Local_dir name is too long");

      for (i=0x1;i<=0x8;i<<=1)
      {
         if (i&ioflg)
         {
            if (i==(ioflg&0x1))
               sprintf(cnfg_file,"%s/%d/%d/%s_%d",
                       local_dir,my_rank/n,my_rank%n,cnfg,my_rank);
            else if (i==(ioflg&0x2))
               sprintf(cnfg_file,"%s/%d/%d/%s.mom_%d",
                       local_dir,my_rank/n,my_rank%n,cnfg,my_rank);
            else if (i==(ioflg&0x4))
               sprintf(cnfg_file,"%s/%d/%d/%s.pf0_%d",
                       local_dir,my_rank/n,my_rank%n,cnfg,my_rank);
            else if (i==(ioflg&0x8))
               sprintf(cnfg_file,"%s/%d/%d/%s.fld_%d",
                       local_dir,my_rank/n,my_rank%n,cnfg,my_rank);

            fdat=fopen(cnfg_file,"rb");
            error_loc(fdat==NULL,1,"check_iodat0 [iodat.c]",
                      "Unable to access configuration file %s",cnfg_file);
            fclose(fdat);
         }
      }
   }
}


static void check_iodat1(iodat_t *iodat,int ioflg,char *cnfg)
{
   int my_rank,types,nion,nios;
   int npf,n,m,nl[4];
   int nb,ib,*bs;
   char dir_file[NAME_SIZE];
   char *cnfg_dir,*block_dir,*local_dir;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   types=iodat[0].types;
   nion=iodat[0].nio_nodes;
   nios=iodat[0].nio_streams;
   npf=get_npf();

   if (types&0x1)
   {
      if (my_rank==0)
      {
         cnfg_dir=iodat[0].cnfg_dir;

         if (ioflg&0x4)
            m=name_size("%s/%s.pf%d",cnfg_dir,cnfg,npf-1);
         else if ((ioflg&0x2)||(ioflg&0x8))
            m=name_size("%s/%s.mom",cnfg_dir,cnfg);
         else
            m=name_size("%s/%s",cnfg_dir,cnfg);

         error_root(m>=NAME_SIZE,1,"check_iodat1 [iodat.c]",
                    "cnfg_dir name is too long");

         check_dir(cnfg_dir);
      }
   }

   if (types&0x2)
   {
      block_dir=iodat[0].block_dir;
      bs=iodat[0].bs;

      nl[0]=NPROC0*L0;
      nl[1]=NPROC1*L1;
      nl[2]=NPROC2*L2;
      nl[3]=NPROC3*L3;
      ib=blk_index(nl,bs,&nb);

      iodat[0].nb=nb;
      iodat[0].ib=ib;

      if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
          ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0))
      {
         error_loc((nion<1)||((nb%nion)!=0)||(nios<1)||((nb%nios)!=0),1,
                   "check_iodat1 [iodat.c]",
                   "Improper choice of nio_nodes or nio_streams");
         n=nb/nion;

         if (ioflg&0x4)
            m=name_size("%s/%d/%d/%s.pf%d_b%d",
                        block_dir,ib/n,ib%n,cnfg,npf-1,ib);
         else if ((ioflg&0x2)||(ioflg&0x8))
            m=name_size("%s/%d/%d/%s.mom_b%d",
                        block_dir,ib/n,ib%n,cnfg,ib);
         else
            m=name_size("%s/%d/%d/%s_b%d",
                        block_dir,ib/n,ib%n,cnfg,ib);

         error_loc(m>=NAME_SIZE,1,"check_iodat1 [iodat.c]",
                   "block_dir name is too long");

         sprintf(dir_file,"%s/%d/%d",block_dir,ib/n,ib%n);
         check_dir(dir_file);
      }
   }

   if (types&0x4)
   {
      local_dir=iodat[0].local_dir;
      error_loc((nion<1)||((NPROC%nion)!=0)||(nios<1)||((NPROC%nios)!=0),1,
                "check_iodat1 [iodat.c]",
                "Improper choice of nio_nodes or nio_streams");
      n=NPROC/nion;

      if (ioflg&0x4)
         m=name_size("%s/%d/%d/%s.pf%d_%d",
                     local_dir,my_rank/n,my_rank%n,cnfg,npf-1,my_rank);
      else if ((ioflg&0x2)||(ioflg&0x8))
         m=name_size("%s/%d/%d/%s.mom_%d",
                     local_dir,my_rank/n,my_rank%n,cnfg,my_rank);
      else
         m=name_size("%s/%d/%d/%s_%d",
                     local_dir,my_rank/n,my_rank%n,cnfg,my_rank);

      error_loc(m>=NAME_SIZE,1,"check_iodat1 [iodat.c]",
                "Local_dir name is too long");

      sprintf(dir_file,"%s/%d/%d",local_dir,my_rank/n,my_rank%n);
      check_dir(dir_file);
   }
}


void check_iodat(iodat_t *iodat,char *io,int ioflg,char *cnfg)
{
   error_root(ipt==NULL,1,"check_iodat [iodat.c]",
              "Geometry arrays are not set");
   error_root((ioflg&0xf)==0,1,"check_iodat [iodat.c]",
              "Unexpected value of ioflg");

   if (strchr(io,'i')!=NULL)
      check_iodat0(iodat,ioflg,cnfg);
   else if (strchr(io,'o')!=NULL)
      check_iodat1(iodat,ioflg,cnfg);
}


void write_iodat_parms(FILE *fdat,iodat_t *iodat)
{
   int my_rank,endian,iw;
   stdint_t istd[6];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if (my_rank==0)
   {
      istd[0]=(stdint_t)(iodat[0].types);
      istd[1]=(stdint_t)(iodat[0].nio_nodes);
      istd[2]=(stdint_t)(iodat[0].bs[0]);
      istd[3]=(stdint_t)(iodat[0].bs[1]);
      istd[4]=(stdint_t)(iodat[0].bs[2]);
      istd[5]=(stdint_t)(iodat[0].bs[3]);

      if (endian==BIG_ENDIAN)
         bswap_int(6,istd);

      iw=fwrite(istd,sizeof(stdint_t),6,fdat);
      error_root(iw!=6,1,"write_iodat_parms [iodat.c]",
                 "Incorrect write count");
   }
}


void check_iodat_parms(FILE *fdat,iodat_t *iodat)
{
   int my_rank,endian,ir,ie;
   stdint_t istd[6];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if (my_rank==0)
   {
      ir=fread(istd,sizeof(stdint_t),6,fdat);
      error_root(ir!=6,1,"check_iodat_parms [iodat.c]",
                 "Incorrect read count");

      if (endian==BIG_ENDIAN)
         bswap_int(6,istd);

      ie=0;
      ie|=(istd[0]!=(stdint_t)(iodat[0].types));
      ie|=(istd[1]!=(stdint_t)(iodat[0].nio_nodes));
      ie|=(istd[2]!=(stdint_t)(iodat[0].bs[0]));
      ie|=(istd[3]!=(stdint_t)(iodat[0].bs[1]));
      ie|=(istd[4]!=(stdint_t)(iodat[0].bs[2]));
      ie|=(istd[5]!=(stdint_t)(iodat[0].bs[3]));

      error_root(ie!=0,1,"check_iodat_parms [iodat.c]",
                 "Parameters do not match");
   }
}
