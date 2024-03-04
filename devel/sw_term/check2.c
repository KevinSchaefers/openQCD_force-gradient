
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2005-2016, 2018, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the gauge covariance of the SW term.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "sflds.h"
#include "linalg.h"
#include "devfcts.h"
#include "sw_term.h"
#include "global.h"


static void transform_sd(spinor_dble *pk,spinor_dble *pl)
{
   int ix;
   su3_dble gx,*gd;
   spinor_dble r,s;

   gd=gtrans();

   for (ix=0;ix<VOLUME;ix++)
   {
      s=pk[ix];
      gx=gd[ix];

      _su3_multiply(r.c1,gx,s.c1);
      _su3_multiply(r.c2,gx,s.c2);
      _su3_multiply(r.c3,gx,s.c3);
      _su3_multiply(r.c4,gx,s.c4);

      pl[ix]=r;
   }
}


int main(int argc,char *argv[])
{
   int my_rank,bc,is,i;
   double phi[2],phi_prime[2],theta[3];
   double d;
   qflt rqsm;
   spinor_dble **psd;
   pauli_dble *sw;
   sw_parms_t swp;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);
      printf("\n");
      printf("Gauge covariance of the SW term (random fields)\n");
      printf("-----------------------------------------------\n\n");

      print_lattice_sizes();

      bc=find_opt(argc,argv,"-bc");
      is=find_opt(argc,argv,"-sw");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check2.c]",
                    "Syntax: check1 [-bc <type>] [-sw <type>]");

      if (is!=0)
         error_root(sscanf(argv[is+1],"%d",&is)!=1,1,"main [check2.c]",
                    "Syntax: check1 [-bc <type>] [-sw <type>]");
   }

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&is,1,MPI_INT,0,MPI_COMM_WORLD);

   check_machine();
   set_lat_parms(5.5,1.0,0,NULL,is,1.978);
   print_lat_parms(0x2);

   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;
   set_bc_parms(bc,1.0,1.0,0.9012,1.2034,phi,phi_prime,theta);
   print_bc_parms(0x2);

   start_ranlux(0,123456);
   geometry();
   alloc_wsd(4);
   psd=reserve_wsd(4);

   swp=set_sw_parms(-0.0123);

   if (my_rank==0)
   {
      printf("Parameters returned by sw_parms():\n");
      printf("isw=%d, m0=%.4e, csw=%.4e",swp.isw,swp.m0,swp.csw);
      if (is)
         printf(" (=> N=%d)\n",sw_order());
      else
         printf("\n");
      printf("cF=%.4e, cF'=%.4e\n\n",swp.cF[0],swp.cF[1]);
   }

   random_gtrans();
   random_ud();

   for (i=0;i<4;i++)
      random_sd(VOLUME_TRD,2,psd[i],1.0);

   (void)(sw_term(NO_PTS));
   sw=swdfld();
   apply_sw_dble(VOLUME,0.789,sw,psd[0],psd[1]);

   transform_sd(psd[0],psd[2]);
   apply_gtrans2ud();
   (void)(sw_term(NO_PTS));
   sw=swdfld();
   apply_sw_dble(VOLUME,0.789,sw,psd[2],psd[3]);
   transform_sd(psd[1],psd[2]);

   mulr_spinor_add_dble(VOLUME_TRD,2,psd[3],psd[2],-1.0);
   rqsm=norm_square_dble(VOLUME,1,psd[3]);
   d=rqsm.q[0];
   rqsm=norm_square_dble(VOLUME,1,psd[0]);
   d/=rqsm.q[0];

   if (my_rank==0)
   {
      printf("Maximal normalized difference = %.2e\n",sqrt(d));
      printf("(should be around 1.0e-15 or so)\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
