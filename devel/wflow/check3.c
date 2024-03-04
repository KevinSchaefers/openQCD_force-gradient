
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2009-2013, 2018, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Convergence of the numerical integration of the Wilson flow.
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
#include "su3fcts.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "archive.h"
#include "forces.h"
#include "tcharge.h"
#include "wflow.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int my_rank,first,last,step;
static int nstep,rule;
static double eps;
static iodat_t iodat[1];
static char nbase[NAME_SIZE],cnfg_file[NAME_SIZE];
static FILE *flog=NULL,*fin=NULL;


static void read_run_parms(void)
{
   int iact;

   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   read_iodat("Configurations","i",iodat);

   if (my_rank==0)
   {
      read_line("first","%d",&first);
      read_line("last","%d",&last);
      read_line("step","%d",&step);

      error_root((first<1)||(last<first)||(step<1)||((last-first)%step!=0),1,
                 "read_run_parms [check3.c]","Improper configuration range");
   }

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   read_bc_parms("Boundary conditions",0x1);

   if (my_rank==0)
   {
      find_section("Wilson flow");
      read_line("nstep","%d\n",&nstep);
      read_line("eps","%lf\n",&eps);
      read_line("rule","%d",&rule);
   }

   error_root((rule<0)||(rule>3),1,"read_run_parms [check3.c]",
              "rule must be 1,2 or 3");

   MPI_Bcast(&nstep,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&rule,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&eps,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   iact=0;
   (void)(set_hmc_parms(1,&iact,0,0,NULL,1,1.0));
}


static void check_files(void)
{
   error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
         "check_files [check3.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);
}


static void cmp_ud(su3_dble *u,su3_dble *v,double *dev)
{
   int i;
   double r[18],d;

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

   dev[0]=0.0;
   dev[1]=0.0;

   for (i=0;i<18;i+=2)
   {
      d=sqrt(r[i]*r[i]+r[i+1]*r[i+1]);

      if (d>dev[0])
         dev[0]=d;

      dev[1]+=d;
   }
}


static void dev_ud(su3_dble *v,double *dev)
{
   double d[2];
   su3_dble *u,*um;

   u=udfld();
   um=u+4*VOLUME;
   dev[0]=0.0;
   dev[1]=0.0;

   for (;u<um;u++)
   {
      cmp_ud(u,v,d);

      if (d[0]>dev[0])
         dev[0]=d[0];

      dev[1]+=d[1];
      v+=1;
   }

   if (NPROC>1)
   {
      d[0]=dev[0];
      d[1]=dev[1];
      MPI_Reduce(d,dev,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Reduce(d+1,dev+1,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(dev,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }

   dev[1]/=((double)(9*NPROC)*(double)(4*VOLUME));
}


int main(int argc,char *argv[])
{
   int icnfg,ncnfg;
   double dE[3],dQ[3],dU[2];
   double act[2],qtop[2],dev[2],nplaq;
   double wt1,wt2,wtavg;
   su3_dble *udb,**usv;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      fin=freopen("check3.in","r",stdin);

      printf("\n");
      printf("Convergence of the numerical integration of the Wilson flow\n");
      printf("-----------------------------------------------------------\n\n");

      print_lattice_sizes();
   }

   read_run_parms();

   if (my_rank==0)
      fclose(fin);

   check_machine();
   print_bc_parms(0x0);

   if (my_rank==0)
   {
      printf("Wilson-flow parameters:\n");
      printf("nstep = %d\n",nstep);
      printf("eps = %.3e\n",eps);

      if (rule==1)
         printf("Using the Euler integrator\n\n");
      else if (rule==2)
         printf("Using the 2nd order RK integrator\n\n");
      else
         printf("Using the 3rd order RK integrator\n\n");

      printf("Comparison of the integrated fields at fixed t=n*eps=%.2e\n",
             (double)(nstep)*eps);
      printf("with a precise integration using 5x the input value of n\n\n");

      printf("The deviation |U_ij-U'_ij| is calculated component by\n");
      printf("component on all links of the lattice\n\n");

      print_iodat("i",iodat);

      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   start_ranlux(0,1234);
   geometry();
   check_files();

   alloc_wud(2);
   alloc_wfd(1);
   usv=reserve_wud(2);
   udb=udfld();

   if (bc_type()==0)
      nplaq=(double)(6*N0-6)*(double)(N1*N2*N3);
   else
      nplaq=(double)(6*N0)*(double)(N1*N2*N3);

   dE[0]=0.0;
   dE[1]=0.0;
   dE[2]=0.0;
   dQ[0]=0.0;
   dQ[1]=0.0;
   dQ[2]=0.0;
   dU[0]=0.0;
   dU[1]=0.0;
   wtavg=0.0;

   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      if (my_rank==0)
      {
         printf("Configuration no %d\n\n",icnfg);
         fflush(flog);
      }

      sprintf(cnfg_file,"%sn%d",nbase,icnfg);
      read_flds(iodat,cnfg_file,0x0,0x1);
      cm3x3_assign(4*VOLUME,udb,usv[0]);

      if (rule==1)
         fwd_euler(10*nstep,eps/10.0);
      else if (rule==2)
         fwd_rk2(4*nstep,eps/4.0);
      else
         fwd_rk3(3*nstep,eps/3.0);

      cm3x3_assign(4*VOLUME,udb,usv[1]);
      act[0]=2.0*(3.0*nplaq-plaq_wsum_dble(1));
      qtop[0]=tcharge();

      cm3x3_assign(4*VOLUME,usv[0],udb);
      set_flags(UPDATED_UD);

      if (rule==1)
         fwd_euler(nstep,eps);
      else if (rule==2)
         fwd_rk2(nstep,eps);
      else
         fwd_rk3(nstep,eps);

      act[1]=2.0*(3.0*nplaq-plaq_wsum_dble(1));
      qtop[1]=tcharge();

      dev[0]=fabs(act[1]-act[0]);
      if (dev[0]>dE[0])
         dE[0]=dev[0];
      dE[1]+=dev[0];
      dE[2]+=act[0];

      dev[0]=fabs(qtop[1]-qtop[0]);
      if (dev[0]>dQ[0])
         dQ[0]=dev[0];
      dQ[1]+=dev[0];
      dQ[2]+=fabs(qtop[0]);

      dev_ud(usv[1],dev);
      if (dev[0]>dU[0])
         dU[0]=dev[0];
      dU[1]+=dev[1];

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wtavg+=(wt2-wt1);

      if (my_rank==0)
      {
         printf("dE/E = %.1e, dQ = %.1e, max|dU| = %.1e, avg|dU| = %.1e\n\n",
                fabs(1.0-act[0]/act[1]),fabs(qtop[1]-qtop[0]),dev[0],dev[1]);
         printf("Configuration no %d fully processed in %.2e sec ",
                icnfg,wt2-wt1);
         printf("(average = %.2e sec)\n\n",
                wtavg/(double)((icnfg-first)/step+1));
         fflush(flog);
      }
   }

   ncnfg=(last-first)/step+1;
   dE[1]/=(double)(ncnfg);
   dE[2]/=(double)(ncnfg);
   dQ[1]/=(double)(ncnfg);
   dQ[2]/=(double)(ncnfg);
   dU[1]/=(double)(ncnfg);

   if (my_rank==0)
   {
      printf("\n");
      printf("Test summary\n");
      printf("------------\n\n");

      printf("Processed %d configurations\n\n",ncnfg);
      printf("max|dE|/E = %.1e, avg|dE|/E = %.1e\n",
             dE[0]/dE[2],dE[1]/dE[2]);
      printf("max|dQ| =   %.1e, avg|dQ| =   %.1e, avg|Q| = %.2e\n",
             dQ[0],dQ[1],dQ[2]);
      printf("max|dU| =   %.1e, avg|dU| =   %.1e\n\n",
             dU[0],dU[1]);

      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
