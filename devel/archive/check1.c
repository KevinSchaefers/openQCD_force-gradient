
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2017, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Writing and reading gauge, momentum and spinor fields.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "sflds.h"
#include "su3fcts.h"
#include "linalg.h"
#include "archive.h"
#include "cmpflds.h"
#include "global.h"


int main(int argc,char *argv[])
{
   int my_rank,bc,ie;
   int nion,nios,ns,n,eo,c;
   double phi[2],phi_prime[2],theta[3];
   su3_alg_dble *mom,**msv;
   su3_dble *udb,**usv;
   spinor_dble **wsd;
   mdflds_t *mdfs;
   char loc_dir[NAME_SIZE],loc_sub_dir[NAME_SIZE],cnfg[NAME_SIZE];
   FILE *flog=NULL,*fin=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   check_machine();

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);
      fin=freopen("check1.in","r",stdin);

      printf("\n");
      printf("Writing and reading gauge, momentum and spinor fields\n");
      printf("-----------------------------------------------------\n\n");

      print_lattice_sizes();

      read_line("loc_dir","%s",loc_dir);
      read_line("nio_nodes","%d",&nion);
      read_line("nio_streams","%d",&nios);
      fclose(fin);

      error_root((nion<1)||(nios<1)||((NPROC%nion)!=0)||((NPROC%nios)!=0),1,
                 "main [check1.c]","Improper nio_nodes or nio_streams");

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check1.c]",
                    "Syntax: check1 [-bc <type>]");

      ns=name_size("%s/%d/%d/test.ud_%d",loc_dir,NPROC-1,NPROC-1,NPROC);
      error_root(ns>=NAME_SIZE,1,"main [check1.c]","loc_dir name is too long");
   }

   MPI_Bcast(loc_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&nion,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nios,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);

   ns=(NPROC/nion);
   n=my_rank/ns;
   c=my_rank%ns;
   sprintf(loc_sub_dir,"%s/%d/%d",loc_dir,n,c);
   check_dir(loc_sub_dir);
   set_nio_streams(nios);

   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.5;
   theta[1]=1.0;
   theta[2]=-0.5;

   n=0;
   set_hmc_parms(1,&n,0,0,NULL,1,1.0);
   set_bc_parms(bc,1.0,1.0,1.0,1.0,phi,phi_prime,theta);
   print_bc_parms(3);

   start_ranlux(0,123456);
   geometry();
   alloc_wud(1);
   alloc_wfd(1);
   alloc_wsd(4);

   if (my_rank==0)
   {
      printf("Write random field configurations to disk. "
             "Then read the fields\n"
             "from there, compare with the saved fields and remove "
             "all files.\n\n");

      printf("nio_nodes=%d\n",nion);
      printf("nio_streams=%d\n\n",nios);
   }

   usv=reserve_wud(1);
   msv=reserve_wfd(1);
   wsd=reserve_wsd(4);
   udb=udfld();
   mdfs=mdflds();
   mom=(*mdfs).mom;

   random_ud();
   random_mom();
   random_sd(VOLUME_TRD,2,wsd[0],1.0);
   random_sd(VOLUME_TRD,2,wsd[1],1.0);

   assign_ud2ud(4*VOLUME_TRD,2,udb,usv[0]);
   assign_alg2alg(4*VOLUME_TRD,2,mom,msv[0]);
   assign_sd2sd(VOLUME_TRD,2,wsd[0],wsd[2]);
   assign_sd2sd(VOLUME_TRD,2,wsd[1],wsd[3]);

   print_flags();

   sprintf(cnfg,"%s/test.ud_%d",loc_sub_dir,my_rank);
   write_cnfg(cnfg);
   sprintf(cnfg,"%s/test.fd_%d",loc_sub_dir,my_rank);
   write_mfld(cnfg);

   for (eo=0;eo<2;eo++)
   {
      sprintf(cnfg,"%s/test.sd%d_%d",loc_sub_dir,eo,my_rank);
      write_sfld(cnfg,eo,wsd[eo]);
   }

   random_ud();
   random_mom();
   random_sd(VOLUME_TRD,2,wsd[0],1.0);
   random_sd(VOLUME_TRD/2,2,wsd[1],1.0);

   sprintf(cnfg,"%s/test.ud_%d",loc_sub_dir,my_rank);
   read_cnfg(cnfg);
   remove(cnfg);
   sprintf(cnfg,"%s/test.fd_%d",loc_sub_dir,my_rank);
   read_mfld(cnfg);
   remove(cnfg);

   for (eo=0;eo<2;eo++)
   {
      sprintf(cnfg,"%s/test.sd%d_%d",loc_sub_dir,eo,my_rank);
      read_sfld(cnfg,eo,wsd[eo]);
      remove(cnfg);
   }

   ie=(check_bc(0.0)^0x1);
   ie|=check_ud(4*VOLUME,udb,usv[0]);
   error(ie!=0,1,"main [check1.c]",
         "The gauge field is not properly restored");

   ie=check_fd(4*VOLUME,mom,msv[0]);
   error(ie!=0,1,"main [check1.c]",
         "The momentum field is not properly restored");

   for (eo=0;eo<2;eo++)
   {
      ie=check_sd(VOLUME,wsd[eo],wsd[eo+2]);
      error(ie!=0,1,"main [check1.c]",
            "The spinor field is not properly restored (eo=%d)",eo);
   }

   print_flags();

   if (my_rank==0)
   {
      printf("No errors detected --- the fields are correctly written\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
