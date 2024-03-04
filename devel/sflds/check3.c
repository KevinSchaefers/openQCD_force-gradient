
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the programs in the module unorm.c.
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
#include "utils.h"
#include "lattice.h"
#include "sflds.h"
#include "global.h"

#define NFLDS 4

typedef union
{
   spinor s;
   float r[24];
} spin_t;

typedef union
{
   spinor_dble s;
   double r[24];
} spin_dble_t;


static float loc_nrm(spinor *s)
{
   int i;
   float sm;
   spin_t *sp;

   sm=0.0f;
   sp=(spin_t*)(s);

   for (i=0;i<24;i++)
      sm+=(*sp).r[i]*(*sp).r[i];

   return (float)(sqrt((double)(sm)));
}


static double loc_nrm_dble(spinor_dble *sd)
{
   int i;
   double sm;
   spin_dble_t *sp;

   sm=0.0;
   sp=(spin_dble_t*)(sd);

   for (i=0;i<24;i++)
      sm+=(*sp).r[i]*(*sp).r[i];

   return sqrt(sm);
}


static int chk_nrm(int vol,int icom,spinor *s)
{
   float nrm,ns,dist,dmax,tol;
   spinor *sm;

   nrm=unorm(vol,icom,s);
   dist=nrm;
   dmax=0.0f;

   if (icom&0x2)
      sm=s+vol*NTHREAD;
   else
      sm=s+vol;

   for (;s<sm;s++)
   {
      ns=loc_nrm(s);

      if (ns<=nrm)
      {
         ns=nrm-ns;
         if (ns<dist)
            dist=ns;
      }
      else
      {
         ns=ns-nrm;
         if (ns<dist)
            dist=ns;
         if (ns>dmax)
            dmax=ns;
      }
   }

   tol=16.0f*nrm*FLT_EPSILON;

   if ((NPROC>1)&&(icom&0x1))
   {
      ns=dist;
      MPI_Reduce(&ns,&dist,1,MPI_FLOAT,MPI_MIN,0,MPI_COMM_WORLD);
      MPI_Bcast(&dist,1,MPI_FLOAT,0,MPI_COMM_WORLD);

      ns=dmax;
      MPI_Reduce(&ns,&dmax,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmax,1,MPI_FLOAT,0,MPI_COMM_WORLD);
   }

   return ((dist<=tol)&&(dmax<=tol));
}


static int chk_nrm_dble(int vol,int icom,spinor_dble *sd)
{
   double nrm,ns,dist,dmax,tol;
   spinor_dble *sm;

   nrm=unorm_dble(vol,icom,sd);
   dist=nrm;
   dmax=0.0;

   if (icom&0x2)
      sm=sd+vol*NTHREAD;
   else
      sm=sd+vol;

   for (;sd<sm;sd++)
   {
      ns=loc_nrm_dble(sd);

      if (ns<=nrm)
      {
         ns=nrm-ns;
         if (ns<dist)
            dist=ns;
      }
      else
      {
         ns=ns-nrm;
         if (ns<dist)
            dist=ns;
         if (ns>dmax)
            dmax=ns;
      }
   }

   tol=16.0*nrm*DBL_EPSILON;

   if ((NPROC>1)&&(icom&0x1))
   {
      ns=dist;
      MPI_Reduce(&ns,&dist,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
      MPI_Bcast(&dist,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      ns=dmax;
      MPI_Reduce(&ns,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }

   return ((dist<=tol)&&(dmax<=tol));
}


int main(int argc,char *argv[])
{
   int my_rank,k,ie;
   int vol,icom;
   spinor **ps;
   spinor_dble **psd;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      printf("\n");
      printf("Check of the programs in the module unorm.c\n");
      printf("-------------------------------------------\n\n");

      print_lattice_sizes();
   }

   start_ranlux(0,12345);
   geometry();
   alloc_ws(NFLDS);
   alloc_wsd(NFLDS);
   ps=reserve_ws(NFLDS);
   psd=reserve_wsd(NFLDS);

   for (icom=0;icom<4;icom++)
   {
      if (icom&0x2)
         vol=VOLUME_TRD;
      else
         vol=VOLUME;

      for (k=0;k<NFLDS;k++)
      {
         random_s(vol,icom,ps[k],1.0f+(float)(k));
         random_sd(vol,icom,psd[k],1.0+(double)(k));

         ie=chk_nrm(vol,icom,ps[k]);
         error(ie!=1,1,"main [check3.c]",
               "Unexpected result of unorm() (icom=%d)",k&0x1);

         ie=chk_nrm_dble(vol,icom,psd[k]);
         error(ie!=1,1,"main [check3.c]",
               "Unexpected result of unorm_dble() (icom=%d)",k&0x1);
      }

      if (my_rank==0)
         printf("No errors discovered (icom=%d)\n",icom);

      MPI_Barrier(MPI_COMM_WORLD);
   }

   if (my_rank==0)
   {
      printf("\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
