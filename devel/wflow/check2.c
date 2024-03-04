
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2009-2013, 2016, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Gauge covariance of the Wilson flow.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "flags.h"
#include "su3fcts.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "update.h"
#include "devfcts.h"
#include "wflow.h"
#include "global.h"


static double cmp_ud(su3_dble *u,su3_dble *v)
{
   int i;
   double r[18],dev,dmax;

   r[ 0]=(*u).c11.re-(*v).c11.re;
   r[ 1]=(*u).c11.im-(*v).c11.im;
   r[ 2]=(*u).c12.re-(*v).c12.re;
   r[ 3]=(*u).c12.im-(*v).c12.im;
   r[ 4]=(*u).c13.re-(*v).c13.re;
   r[ 5]=(*u).c13.im-(*v).c13.im;

   r[ 6]=(*u).c21.re-(*v).c21.re;
   r[ 7]=(*u).c21.im-(*v).c21.im;
   r[ 8]=(*u).c22.re-(*v).c22.re;
   r[ 9]=(*u).c22.im-(*v).c22.im;
   r[10]=(*u).c23.re-(*v).c23.re;
   r[11]=(*u).c23.im-(*v).c23.im;

   r[12]=(*u).c31.re-(*v).c31.re;
   r[13]=(*u).c31.im-(*v).c31.im;
   r[14]=(*u).c32.re-(*v).c32.re;
   r[15]=(*u).c32.im-(*v).c32.im;
   r[16]=(*u).c33.re-(*v).c33.re;
   r[17]=(*u).c33.im-(*v).c33.im;

   dmax=0.0;

   for (i=0;i<18;i+=2)
   {
      dev=r[i]*r[i]+r[i+1]*r[i+1];
      if (dev>dmax)
         dmax=dev;
   }

   return dmax;
}


static double max_dev_ud(su3_dble *v)
{
   double d,dmax;
   su3_dble *u,*um;

   u=udfld();
   um=u+4*VOLUME;
   dmax=0.0;

   for (;u<um;u++)
   {
      d=cmp_ud(u,v);

      if (d>dmax)
         dmax=d;

      v+=1;
   }

   if (NPROC>1)
   {
      d=dmax;
      MPI_Reduce(&d,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }

   return sqrt(dmax);
}


int main(int argc,char *argv[])
{
   int my_rank,bc,iact,n;
   double phi[2],phi_prime[2],theta[3];
   double eps,dev;
   su3_dble *udb,**usv;
   FILE *flog=NULL,*fin=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);
      fin=freopen("check2.in","r",stdin);

      printf("\n");
      printf("Gauge covariance of the Wilson flow\n");
      printf("-----------------------------------\n\n");

      print_lattice_sizes();

      read_line("n","%d\n",&n);
      read_line("eps","%lf",&eps);
      fclose(fin);

      printf("n = %d\n",n);
      printf("eps = %.3e\n\n",eps);

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check2.c]",
                    "Syntax: check2 [-bc <type>]");
   }

   MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&eps,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);

   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;

   iact=0;
   set_hmc_parms(1,&iact,0,0,NULL,1,1.0);
   set_bc_parms(bc,0.973,1.127,1.0,1.0,phi,phi_prime,theta);
   print_bc_parms(0x0);

   start_ranlux(0,1234);
   geometry();
   alloc_wud(2);
   alloc_wfd(1);
   usv=reserve_wud(2);
   udb=udfld();

   random_ud();
   random_gtrans();
   cm3x3_assign(4*VOLUME,udb,usv[0]);
   fwd_euler(n,eps);
   apply_gtrans2ud();
   cm3x3_assign(4*VOLUME,udb,usv[1]);
   cm3x3_assign(4*VOLUME,usv[0],udb);
   set_flags(UPDATED_UD);
   apply_gtrans2ud();
   fwd_euler(n,eps);

   dev=max_dev_ud(usv[1]);

   if (my_rank==0)
   {
      printf("Maximal absolute deviation of U(x,mu) = %.1e\n\n",dev);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
