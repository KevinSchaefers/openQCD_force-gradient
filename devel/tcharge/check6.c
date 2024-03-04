
/*******************************************************************************
*
* File check6.c
*
* Copyright (C) 2010, 2013, 2016, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the programs ym_action_slices() and ym_action_fld().
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "random.h"
#include "su3fcts.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "wflow.h"
#include "msfcts.h"
#include "tcharge.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int bc,n,dn;
static double eps,A1,A2,A[N0],A0[N0];


static double sum_fld(double *f)
{
   int k;
   double *fr,*fm;
   double sm,*qsm[1];
   qflt rqsm;

   rqsm.q[0]=0.0;
   rqsm.q[1]=0.0;

#pragma omp parallel private(k,fr,fm,sm) reduction(sum_qflt : rqsm)
   {
      k=omp_get_thread_num();

      fr=f+k*VOLUME_TRD;
      fm=fr+VOLUME_TRD;

      for (;fr<fm;fr+=4)
      {
         sm=fr[0]+fr[1]+fr[2]+fr[3];
         acc_qflt(sm,rqsm.q);
      }
   }

   if (NPROC>1)
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   return rqsm.q[0];
}


int main(int argc,char *argv[])
{
   int my_rank,iact,i,imax,t;
   double phi[2],phi_prime[2],theta[3];
   double dev,dmax,*f;
   FILE *fin=NULL,*flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check6.log","w",stdout);
      fin=freopen("check6.in","r",stdin);

      printf("\n");
      printf("Check of the programs ym_action_slices() and ym_action_fld()\n");
      printf("------------------------------------------------------------\n\n");

      print_lattice_sizes();

      read_line("n","%d\n",&n);
      read_line("dn","%d\n",&dn);
      read_line("eps","%lf",&eps);
      fclose(fin);

      printf("n = %d\n",n);
      printf("dn = %d\n",dn);
      printf("eps = %.2e\n\n",eps);

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check6.c]",
                    "Syntax: check6 [-bc <type>]");
   }

   check_machine();
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&dn,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&eps,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;

   iact=0;
   set_hmc_parms(1,&iact,0,0,NULL,1,1.0);
   set_bc_parms(bc,1.0,1.0,1.0,1.0,phi,phi_prime,theta);
   print_bc_parms(0x0);

   start_ranlux(0,123456);
   geometry();
   alloc_wfd(2);
   f=malloc(VOLUME*sizeof(*f));
   error(f==NULL,1,"main [check6.c]","Unable to allocate field array");

   random_ud();
   imax=n/dn;
   dmax=0.0;

   for (i=0;i<imax;i++)
   {
      fwd_euler(dn,eps);

      A1=ym_action();
      A2=ym_action_slices(A);
      dev=fabs(A1-A2)/A1;

      for (t=0;t<N0;t++)
      {
         A2-=A[t];
         A0[t]=A[t];
      }

      dev+=fabs(A2)/A1;

      if (my_rank==0)
      {
         printf("n=%3d, A=%12.6e, dev=%6.1e, A[0...%d]=%8.2e",
                (i+1)*dn,A1,dev,N0-1,A[0]);

         for (t=1;t<N0;t++)
            printf(", %8.2e",A[t]);

         printf("\n");
      }

      MPI_Bcast(A0,N0,MPI_DOUBLE,0,MPI_COMM_WORLD);

      for (t=0;t<N0;t++)
      {
         if ((A[t]-A0[t])!=0.0)
            break;
      }

      error(t!=N0,1,"main [check6.c]",
            "Action slices are not globally the same");

      ym_action_fld(f);
      A2=sum_fld(f);
      dev=fabs(A1-A2)/((double)(VOLUME)*(double)(NPROC));
      if (dev>dmax)
         dmax=dev;
   }

   if (my_rank==0)
   {
      printf("\n");
      printf("Check of ym_action_fld():\n"
             "Maximal absolute deviation of the action per point = %.1e\n\n",
             dmax);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
