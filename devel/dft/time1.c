
/*******************************************************************************
*
* File time1.c
*
* Copyright (C) 2015, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of the 4d parallel Fourier transform.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "utils.h"
#include "random.h"
#include "lattice.h"
#include "dft.h"
#include "global.h"

static complex_dble **f=NULL;


static void alloc_arrays(int nflds,int csize,int *nx)
{
   int n,i;
   complex_dble **ww,*w;

   if (f!=NULL)
   {
      afree(f[0]);
      free(f);
   }

   n=csize*nx[0]*nx[1]*nx[2]*nx[3];
   ww=malloc(nflds*sizeof(*ww));
   w=amalloc(nflds*n*sizeof(*w),4);

   error((ww==NULL)||(w==NULL),1,"alloc_arrays [time1.c]",
         "Unable to allocate field arrays");

   gauss_dble((double*)(w),2*nflds*n);

   for (i=0;i<nflds;i++)
   {
      ww[i]=w;
      w+=n;
   }

   f=ww;
}


static double wt_scalar_dft4d(void)
{
   int my_rank,nmax,n,i,ib;
   int idp[4],nx[4],csize,id,nflds;
   double wt1,wt2,wdt,wtav;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   idp[0]=set_dft_parms(EXP,NPROC0*L0,0,0);
   idp[1]=set_dft_parms(EXP,NPROC1*L1,0,0);
   idp[2]=set_dft_parms(EXP,NPROC2*L2,0,0);
   idp[3]=set_dft_parms(EXP,NPROC3*L3,0,1);

   nx[0]=L0;
   nx[1]=L1;
   nx[2]=L2;
   nx[3]=L3;
   csize=1;
   id=set_dft4d_parms(idp,nx,csize);

   nflds=(int)((4*1024*1024)/(VOLUME_TRD*sizeof(complex_dble)))+1;
   if ((nflds%2)==1)
      nflds+=1;

   alloc_arrays(nflds,csize,nx);
   dft4d(id,f[0],f[1]);
   nmax=1;

   for (ib=0;ib<1;nmax*=2)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      for (n=0;n<nmax;n++)
      {
         for (i=0;i<nflds;i+=2)
            dft4d(id,f[i],f[i+1]);
      }

      wt2=MPI_Wtime();
      wdt=wt2-wt1;

      MPI_Reduce(&wdt,&wtav,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

      if (my_rank==0)
      {
         wtav/=(double)(NPROC);

         if (wtav>2.0)
            ib=1;

         wtav/=(double)((nmax*nflds)/2);
      }

      MPI_Bcast(&ib,1,MPI_INT,0,MPI_COMM_WORLD);
   }

   MPI_Bcast(&wtav,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   return wtav;
}


static double wt_gluon_dft4d(void)
{
   int my_rank,nmax,n,i,ib;
   int idp[4],nx[4],csize,id,nflds;
   double wt1,wt2,wdt,wtav;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   idp[0]=set_dft_parms(SIN,NPROC0*L0,1,0);
   idp[1]=set_dft_parms(EXP,NPROC1*L1,0,0);
   idp[2]=set_dft_parms(EXP,NPROC2*L2,0,0);
   idp[3]=set_dft_parms(EXP,NPROC3*L3,0,0);

   if (cpr[0]==(NPROC0-1))
      nx[0]=L0+1;
   else
      nx[0]=L0;
   nx[1]=L1;
   nx[2]=L2;
   nx[3]=L3;
   csize=8;
   id=set_dft4d_parms(idp,nx,csize);

   nflds=(int)((4*1024*1024)/(8*VOLUME_TRD*sizeof(complex_dble)))+1;
   if ((nflds%2)==1)
      nflds+=1;

   alloc_arrays(nflds,csize,nx);
   dft4d(id,f[0],f[1]);
   nmax=1;

   for (ib=0;ib<1;nmax*=2)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      for (n=0;n<nmax;n++)
      {
         for (i=0;i<nflds;i+=2)
            dft4d(id,f[i],f[i+1]);
      }

      wt2=MPI_Wtime();
      wdt=wt2-wt1;

      MPI_Reduce(&wdt,&wtav,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

      if (my_rank==0)
      {
         wtav/=(double)(NPROC);

         if (wtav>2.0)
            ib=1;

         wtav/=(double)((nmax*nflds)/2);
      }

      MPI_Bcast(&ib,1,MPI_INT,0,MPI_COMM_WORLD);
   }

   MPI_Bcast(&wtav,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   return 4.0*wtav;
}


int main(int argc,char *argv[])
{
   int my_rank;
   double wdt;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("time1.log","w",stdout);

      printf("\n");
      printf("Timing of the dft4d() program\n");
      printf("-----------------------------\n\n");

      print_lattice_sizes();

      if ((2*VOLUME*sizeof(double))<(4096*1024))
         printf("The local size of a complex scalar field is %d KB\n",
                (int)((2*VOLUME*sizeof(double))/(1024)));
      else
         printf("The local size of a complex scalar field is %d MB\n",
                (int)((2*VOLUME*sizeof(double))/(1024*1024)));

      if ((64*VOLUME*sizeof(double))<(4096*1024))
         printf("The local size of a complex gauge potential is %d KB\n",
                (int)((64*VOLUME*sizeof(double))/(1024)));
      else
         printf("The local size of a complex gauge potential is %d MB\n",
                (int)((64*VOLUME*sizeof(double))/(1024*1024)));

#if (defined x64)
#if (defined AVX)
      printf("Using AVX instructions\n");
#else
      printf("Using SSE3 instructions and 16 xmm registers\n");
#endif
#if (defined P3)
      printf("Assuming SSE prefetch instructions fetch 32 bytes\n");
#elif (defined PM)
      printf("Assuming SSE prefetch instructions fetch 64 bytes\n");
#elif (defined P4)
      printf("Assuming SSE prefetch instructions fetch 128 bytes\n");
#else
      printf("SSE prefetch instructions are not used\n");
#endif
#endif
      printf("\n");
   }

   start_ranlux(0,12345);
   geometry();

   wdt=wt_scalar_dft4d();

   if (my_rank==0)
   {
      printf("Fourier transform of a scalar field:\n");
      printf("Total time: %.1e sec\n",wdt);
      printf("Time per thread & lattice point: %4.3f usec\n\n",
             1.0e6*wdt/(double)(VOLUME_TRD));
   }

   wdt=wt_gluon_dft4d();

   if (my_rank==0)
   {
      printf("Fourier transform of a gluon field:\n");
      printf("Total time: %.1e sec\n",wdt);
      printf("Time per thread & lattice point: %4.3f usec\n\n",
             1.0e6*wdt/(double)(VOLUME_TRD));
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
