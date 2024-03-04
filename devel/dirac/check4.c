
/*******************************************************************************
*
* File check4.c
*
* Copyright (C) 2005-2016, 2018, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Gauge covariance of Dw_dble().
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
#include "sw_term.h"
#include "dirac.h"
#include "devfcts.h"
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
   int my_rank,bc,i;
   double phi[2],phi_prime[2],theta[3];
   double mu,d;
   qflt rqsm;
   spinor_dble **psd;
   sw_parms_t swp;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check4.log","w",stdout);
      printf("\n");
      printf("Gauge covariance of Dw_dble() (random fields)\n");
      printf("---------------------------------------------\n\n");

      print_lattice_sizes();

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check4.c]",
                    "Syntax: check1 [-bc <type>]");
   }

   check_machine();
   set_lat_parms(5.5,1.0,0,NULL,0,1.978);
   print_lat_parms(0x2);

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.35;
   theta[1]=-1.25;
   theta[2]=0.78;
   set_bc_parms(bc,0.55,0.78,0.9012,1.2034,phi,phi_prime,theta);
   print_bc_parms(2);

   start_ranlux(0,12345);
   geometry();
   alloc_wsd(5);
   psd=reserve_wsd(5);

   swp=set_sw_parms(-0.0123);
   mu=0.0376;

   if (my_rank==0)
      printf("m0 = %.4e, csw = %.4e, cF = %.4e, cF' = %.4e\n\n",
             swp.m0,swp.csw,swp.cF[0],swp.cF[1]);

   random_gtrans();
   random_ud();
   set_ud_phase();
   sw_term(NO_PTS);

   for (i=0;i<4;i++)
      random_sd(NSPIN,0,psd[i],1.0);

   assign_sd2sd(VOLUME_TRD,2,psd[0],psd[4]);
   bnd_sd2zero(ALL_PTS,psd[4]);
   Dw_dble(mu,psd[0],psd[1]);
   mulr_spinor_add_dble(VOLUME_TRD,2,psd[4],psd[0],-1.0);
   rqsm=norm_square_dble(VOLUME_TRD,3,psd[4]);
   error(rqsm.q[0]!=0.0,1,"main [check4.c]","Dw_dble() changes the input field");

   Dw_dble(mu,psd[0],psd[4]);
   mulr_spinor_add_dble(VOLUME_TRD,2,psd[4],psd[1],-1.0);
   rqsm=norm_square_dble(VOLUME_TRD,3,psd[4]);
   error(rqsm.q[0]!=0.0,1,"main [check4.c]","Action of Dw_dble() depends "
         "on the boundary values of the input field");

   assign_sd2sd(VOLUME_TRD,2,psd[1],psd[4]);
   bnd_sd2zero(ALL_PTS,psd[4]);
   mulr_spinor_add_dble(VOLUME_TRD,2,psd[4],psd[1],-1.0);
   rqsm=norm_square_dble(VOLUME_TRD,3,psd[4]);
   error(rqsm.q[0]!=0.0,1,"main [check4.c]",
         "Dw_dble() does not vanish at global time 0 and NPROC0*L0-1 ");

   apply_gtrans2ud();
   transform_sd(psd[0],psd[2]);
   sw_term(NO_PTS);
   Dw_dble(mu,psd[2],psd[3]);
   transform_sd(psd[1],psd[2]);

   mulr_spinor_add_dble(VOLUME_TRD,2,psd[3],psd[2],-1.0);
   rqsm=norm_square_dble(VOLUME_TRD,3,psd[3]);
   d=rqsm.q[0];
   rqsm=norm_square_dble(VOLUME_TRD,3,psd[0]);
   d/=rqsm.q[0];

   if (my_rank==0)
   {
      printf("Normalized difference = %.2e\n",sqrt(d));
      printf("(should be around 1*10^(-15) or so)\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
