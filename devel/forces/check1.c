
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2012-2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Gauge and translation invariance of the gauge action.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "forces.h"
#include "devfcts.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)


static double bnd_action(void)
{
   int bc,i,j;
   double c0,c1,*cG,*phi;
   double s[3],d0[2],d1[2],act;
   lat_parms_t lat;
   bc_parms_t bcp;

   bc=bc_type();

   if ((bc==1)||(bc==2))
   {
      lat=lat_parms();
      bcp=bc_parms();

      s[0]=(double)(N1);
      s[1]=(double)(N2);
      s[2]=(double)(N3);

      for (i=0;i<2;i++)
      {
         d0[i]=0.0;
         d1[i]=0.0;
         phi=bcp.phi[i];

         for (j=0;j<3;j++)
         {
            d0[i]-=(cos(phi[0]/s[j])+cos(phi[1]/s[j])+
                    cos(phi[2]/s[j])-3.0);
            d1[i]-=(cos(2.0*phi[0]/s[j])+cos(2.0*phi[1]/s[j])+
                    cos(2.0*phi[2]/s[j])-3.0);
         }
      }

      c0=lat.c0;
      c1=lat.c1;
      cG=bcp.cG;

      act=c0*cG[1]*d0[1]+c1*d0[1]+c1*1.5*d1[1];

      if (bc==1)
         act+=(c0*cG[0]*d0[0]+c1*d0[0]+c1*1.5*d1[0]);

      return (lat.beta/3.0)*(double)(N1*N2*N3)*act;
   }
   else
      return 0.0;
}


int main(int argc,char *argv[])
{
   int my_rank,bc,n,s[4];
   double phi[2],phi_prime[2],theta[3],act;
   qflt act0,act1;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);

      printf("\n");
      printf("Gauge and translation invariance of the gauge action\n");
      printf("----------------------------------------------------\n\n");

      print_lattice_sizes();

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check1.c]",
                    "Syntax: check1 [-bc <type>]");
   }

   check_machine();
   set_lat_parms(3.5,0.33,0,NULL,0,1.0);
   print_lat_parms(0x1);

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;
   set_bc_parms(bc,0.9012,1.2034,1.0,1.0,phi,phi_prime,theta);
   print_bc_parms(0x1);

   start_ranlux(0,12345);
   geometry();

   act0=action0(1);
   act=bnd_action();

   if (my_rank==0)
   {
      printf("Action after initialization = %.15e\n",act0.q[0]);
      printf("Expected value              = %.15e\n\n",act);
   }

   random_ud();
   act0=action0(1);

   random_gtrans();
   apply_gtrans2ud();
   act1=action0(1);
   act1.q[0]=-act1.q[0];
   act1.q[1]=-act1.q[1];
   add_qflt(act0.q,act1.q,act1.q);

   if (my_rank==0)
   {
      printf("Random gauge field:\n");
      printf("Action = %.12e\n",act0.q[0]);
      printf("Gauge invariance: relative difference = %.1e\n\n",
             fabs(act1.q[0]/act0.q[0]));
   }

   if (my_rank==0)
      printf("Translation invariance:\n");

   random_ud();
   act0=action0(1);

   for (n=0;n<8;n++)
   {
      random_shift(s);
      shift_ud(s);
      act1=action0(1);
      act1.q[0]=-act1.q[0];
      act1.q[1]=-act1.q[1];
      add_qflt(act0.q,act1.q,act1.q);

      if (my_rank==0)
      {
         printf("s=(% d, % d,% d,% d), ",s[0],s[1],s[2],s[3]);
         printf("relative deviation = %.1e\n",fabs(act1.q[0]/act0.q[0]));
      }
   }

   if (my_rank==0)
   {
      printf("\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
