
/*******************************************************************************
*
* File check4.c
*
* Copyright (C) 2005-2016, 2018, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the programs for the plaquette sums of the double-precision
* gauge field.
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
#include "devfcts.h"
#include "uflds.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)


int main(int argc,char *argv[])
{
   int my_rank,bc,n,t,s[4];
   double phi[2],phi_prime[2],theta[3];
   double act1,nplaq1,nplaq2,p1,p2;
   double d1,d2,d3;
   double asl1[N0],asl2[N0];
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check4.log","w",stdout);

      printf("\n");
      printf("Plaquette sums of the double-precision gauge field\n");
      printf("--------------------------------------------------\n\n");

      print_lattice_sizes();

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check4.c]",
                    "Syntax: check4 [-bc <type>]");
   }

   check_machine();
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;
   set_bc_parms(bc,1.0,1.0,1.0,1.0,phi,phi_prime,theta);
   print_bc_parms(0x0);

   start_ranlux(0,12345);
   geometry();

   p1=plaq_sum_dble(1);
   p2=plaq_wsum_dble(1);

   if (bc==0)
   {
      nplaq1=(double)((6*N0-3)*N1)*(double)(N2*N3);
      nplaq2=(double)((6*N0-6)*N1)*(double)(N2*N3);
   }
   else if (bc==3)
   {
      nplaq1=(double)(6*N0*N1)*(double)(N2*N3);
      nplaq2=nplaq1;
   }
   else
   {
      nplaq1=(double)((6*N0+3)*N1)*(double)(N2*N3);
      nplaq2=(double)(6*N0*N1)*(double)(N2*N3);
   }

   d1=0.0;
   d2=0.0;

   if (bc==1)
   {
      d1=cos(phi[0]/(double)(N1))+
         cos(phi[1]/(double)(N1))+
         cos((phi[0]+phi[1])/(double)(N1))+
         cos(phi[0]/(double)(N2))+
         cos(phi[1]/(double)(N2))+
         cos((phi[0]+phi[1])/(double)(N2))+
         cos(phi[0]/(double)(N3))+
         cos(phi[1]/(double)(N3))+
         cos((phi[0]+phi[1])/(double)(N3));

      d1=(d1-9.0)*(double)(N1*N2*N3);
   }

   if ((bc==1)||(bc==2))
   {
      d2=cos(phi_prime[0]/(double)(N1))+
         cos(phi_prime[1]/(double)(N1))+
         cos((phi_prime[0]+phi_prime[1])/(double)(N1))+
         cos(phi_prime[0]/(double)(N2))+
         cos(phi_prime[1]/(double)(N2))+
         cos((phi_prime[0]+phi_prime[1])/(double)(N2))+
         cos(phi_prime[0]/(double)(N3))+
         cos(phi_prime[1]/(double)(N3))+
         cos((phi_prime[0]+phi_prime[1])/(double)(N3));

      d2=(d2-9.0)*(double)(N1*N2*N3);
   }

   if (my_rank==0)
   {
      printf("After field initialization:\n");
      printf("Deviation from expected value (plaq_sum)  = %.1e\n",
             fabs(1.0-p1/(3.0*nplaq1+d1+d2)));
      printf("Deviation from expected value (plaq_wsum) = %.1e\n\n",
             fabs(1.0-p2/(3.0*nplaq2+d1+d2)));
   }

   print_flags();
   random_ud();

   p1=plaq_sum_dble(1);
   p2=plaq_wsum_dble(1);
   act1=plaq_action_slices(asl1);
   d1=act1;

   if ((bc==0)||(bc==3))
   {
      for (t=0;t<N0;t++)
         d1-=asl1[t];
   }

   if (my_rank==0)
   {
      printf("Comparison of plaq_wsum_dble() with plaq_action_slices():\n");
      printf("Action = %.3e, absolute difference = %.1e\n",
             act1,fabs(2.0*(3.0*nplaq2-p2)-act1));
      if ((bc==0)||(bc==3))
         printf("Absolute deviation from sum of action slices = %.1e\n\n",
                fabs(d1));
      else
         printf("\n");
   }

   random_gtrans();
   apply_gtrans2ud();
   d1=fabs(p1-plaq_sum_dble(1));
   d2=fabs(p2-plaq_wsum_dble(1));
   plaq_action_slices(asl2);
   d3=0.0;

   for (t=0;t<N0;t++)
      d3+=fabs(asl1[t]-asl2[t]);

   if (my_rank==0)
   {
      printf("Gauge invariance:\n");
      printf("|Sum| = %.3e\n",fabs(p1));
      printf("Absolute difference (plaq_sum_dble)  = %.1e\n",d1);
      printf("Absolute difference (plaq_wsum_dble) = %.1e\n",d2);
      printf("Absolute difference (action slices)  = %.1e\n\n",
             d3/(double)(N0));
   }

   random_ud();
   p1=plaq_sum_dble(1);
   p2=plaq_wsum_dble(1);
   plaq_action_slices(asl1);

   if (my_rank==0)
   {
      printf("Translation invariance:\n");
      printf("|Sum| = %.3e\n\n",p1);
   }

   for (n=0;n<8;n++)
   {
      random_shift(s);
      shift_ud(s);
      d1=fabs(p1-plaq_sum_dble(1));
      d2=fabs(p2-plaq_wsum_dble(1));
      plaq_action_slices(asl2);
      d3=0.0;

      for (t=0;t<N0;t++)
         d3+=fabs(asl1[safe_mod(t-s[0],N0)]-asl2[t]);

      for (t=0;t<N0;t++)
         asl1[t]=asl2[t];

      if (my_rank==0)
      {
         printf("s=(% 3d,% 3d,% 3d,% 3d):\n",s[0],s[1],s[2],s[3]);
         printf("Absolute deviation (plaq_sum_dble)  = %.1e\n",d1);
         printf("Absolute deviation (plaq_wsum_dble) = %.1e\n",d2);
         printf("Absolute deviation (action slices)  = %.1e\n\n",
                d3/(double)(N0));
      }
   }

   if (bc==1)
   {
      random_ud();
      p1=plaq_sum_dble(1);
      p2=plaq_wsum_dble(1);

      if (my_rank==0)
      {
         printf("\n");
         printf("Comparison of plaq_sum_dble() and plaq_wsum_dble():\n");
         printf("|Sum| = %.3e, Absolute deviation = %.1e\n\n",
                fabs(p1),fabs((p1-9.0*(double)(N1*N2*N3))-p2));
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
