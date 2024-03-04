
/*******************************************************************************
*
* File time2.c
*
* Copyright (C) 2005-2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of the salg_dble routines.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "sflds.h"
#include "linalg.h"
#include "global.h"

static spinor_dble **psd;


static double wt_random_sd(int nflds,int vol,int icom)
{
   int my_rank,nmax,n,i,ib;
   double wt1,wt2,wdt,wtav;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   for (i=0;i<nflds;i++)
      random_sd(vol,icom,psd[i],1.0);

   nmax=1;

   for (ib=0;ib<1;nmax*=2)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      for (n=0;n<nmax;n++)
      {
         for (i=0;i<nflds;i++)
            random_sd(vol,icom,psd[i],1.0);
      }

      wt2=MPI_Wtime();
      wdt=wt2-wt1;

      MPI_Reduce(&wdt,&wtav,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

      if (my_rank==0)
      {
         wtav/=(double)(NPROC);

         if (wtav>2.0)
            ib=1;

         wtav/=(double)(nmax*nflds);
      }

      MPI_Bcast(&ib,1,MPI_INT,0,MPI_COMM_WORLD);
   }

   MPI_Bcast(&wtav,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   return wtav;
}


static double wt_spinor_prod_dble(int nflds,int vol,int icom)
{
   int my_rank,nmax,n,i,ib;
   double wt1,wt2,wdt,wtav;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   for (i=0;i<nflds;i++)
      random_sd(vol,icom,psd[i],1.0);

   nmax=1;

   for (ib=0;ib<1;nmax*=2)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      for (n=0;n<nmax;n++)
      {
         for (i=0;i<nflds;i+=2)
            (void)(spinor_prod_dble(vol,icom,psd[i],psd[i+1]));
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


static double wt_norm_square_dble(int nflds,int vol,int icom)
{
   int my_rank,nmax,n,i,ib;
   double wt1,wt2,wdt,wtav;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   nmax=1;

   for (ib=0;ib<1;nmax*=2)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      for (n=0;n<nmax;n++)
      {
         for (i=0;i<nflds;i++)
            (void)(norm_square_dble(vol,icom,psd[i]));
      }

      wt2=MPI_Wtime();
      wdt=wt2-wt1;

      MPI_Reduce(&wdt,&wtav,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

      if (my_rank==0)
      {
         wtav/=(double)(NPROC);

         if (wtav>2.0)
            ib=1;

         wtav/=(double)(nmax*nflds);
      }

      MPI_Bcast(&ib,1,MPI_INT,0,MPI_COMM_WORLD);
   }

   MPI_Bcast(&wtav,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   return wtav;
}


static double wt_normalize_dble(int nflds,int vol,int icom)
{
   int my_rank,nmax,n,i,ib;
   double wt1,wt2,wdt,wtav;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   nmax=1;

   for (ib=0;ib<1;nmax*=2)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      for (n=0;n<nmax;n++)
      {
         for (i=0;i<nflds;i++)
            (void)(normalize_dble(vol,icom,psd[i]));
      }

      wt2=MPI_Wtime();
      wdt=wt2-wt1;

      MPI_Reduce(&wdt,&wtav,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

      if (my_rank==0)
      {
         wtav/=(double)(NPROC);

         if (wtav>2.0)
            ib=1;

         wtav/=(double)(nmax*nflds);
      }

      MPI_Bcast(&ib,1,MPI_INT,0,MPI_COMM_WORLD);
   }

   MPI_Bcast(&wtav,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   return wtav;
}


static double wt_mulc_spinor_add_dble(int nflds,int vol,int icom)
{
   int my_rank,nmax,n,i,ib;
   complex_dble z;
   double wt1,wt2,wdt,wtav;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   z.re=0.123;
   z.im=0.456;
   nmax=1;

   for (ib=0;ib<1;nmax*=2)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      for (n=0;n<nmax;n++)
      {
         for (i=0;i<nflds;i+=2)
            mulc_spinor_add_dble(vol,icom,psd[i],psd[i+1],z);

         z.re-=z.re;
         z.im-=z.im;
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


static double wt_project_dble(int nflds,int vol,int icom)
{
   int my_rank,nmax,n,i,ib;
   double wt1,wt2,wdt,wtav;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   for (i=0;i<nflds;i++)
   {
      random_sd(vol,icom,psd[i],1.0);
      (void)(normalize_dble(vol,icom,psd[i]));
   }

   nmax=1;

   for (ib=0;ib<1;nmax*=2)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      for (n=0;n<nmax;n++)
      {
         for (i=0;i<nflds;i+=2)
            project_dble(vol,icom,psd[i],psd[i+1]);
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


int main(int argc,char *argv[])
{
   int my_rank,vol,icom,nflds;
   double wdt;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("time2.log","w",stdout);

      printf("\n");
      printf("Timing of the salg_dble routines\n");
      printf("--------------------------------\n\n");

      print_lattice_sizes();

      if ((VOLUME*sizeof(double))<(64*1024))
         printf("The local size of a quark field is %d KB\n",
                (int)((24*VOLUME*sizeof(double))/(1024)));
      else
         printf("The local size of a quark field is %d MB\n",
                (int)((24*VOLUME*sizeof(double))/(1024*1024)));

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

   check_machine();
   start_ranlux(0,12345);
   geometry();

   nflds=(int)((4*1024*1024)/(VOLUME_TRD*sizeof(double)))+1;
   if ((nflds%2)==1)
      nflds+=1;
   if (nflds<10)
      nflds=10;
   alloc_wsd(nflds);
   psd=reserve_wsd(nflds);

   for (icom=0;icom<4;icom++)
   {
      if (((icom&0x1)==0)||(NPROC>1))
      {
         if (icom&0x2)
            vol=VOLUME_TRD;
         else
            vol=VOLUME;

         if (my_rank==0)
         {
            if (icom==0)
            {
               printf("Local w/o threading\n");
               printf("===================\n\n");
            }
            else if (icom==1)
            {
               printf("Global w/o threading\n");
               printf("====================\n\n");
            }
            else if (icom==2)
            {
               printf("Local with threading\n");
               printf("====================\n\n");
            }
            else
            {
               printf("Global with threading\n");
               printf("=====================\n\n");
            }
         }

         if ((icom&0x1)==0)
         {
            wdt=1.0e6*wt_random_sd(nflds,vol,icom)/(double)(vol);

            if (my_rank==0)
            {
               printf("Function random_sd:\n");
               printf("Time per lattice point: %4.3f micro sec\n\n",wdt);
            }
         }

         wdt=1.0e6*wt_spinor_prod_dble(nflds,vol,icom)/(double)(vol);

         if (my_rank==0)
         {
            printf("Function spinor_prod_dble:\n");
            printf("Time per lattice point: %4.3f micro sec\n",wdt);
            printf("%d Mflops [%d bit arithmetic]\n\n",
                   (int)(96.0/wdt),(int)(sizeof(spinor_dble))/3);
         }

         wdt=1.0e6*wt_norm_square_dble(nflds,vol,icom)/(double)(vol);

         if (my_rank==0)
         {
            printf("Function norm_square_dble:\n");
            printf("Time per lattice point: %4.3f micro sec\n",wdt);
            printf("%d Mflops [%d bit arithmetic]\n\n",
                   (int)(48.0/wdt),(int)(sizeof(spinor_dble))/3);
         }

         wdt=1.0e6*wt_normalize_dble(nflds,vol,icom)/(double)(vol);

         if (my_rank==0)
         {
            printf("Function normalize_dble:\n");
            printf("Time per lattice point: %4.3f micro sec\n",wdt);
            printf("%d Mflops [%d bit arithmetic]\n\n",
                   (int)(72.0/wdt),(int)(sizeof(spinor_dble))/3);
         }

         if ((icom&0x1)==0)
         {
            wdt=1.0e6*wt_mulc_spinor_add_dble(nflds,vol,icom)/(double)(vol);

            if (my_rank==0)
            {
               printf("Function mulc_spinor_add_dble:\n");
               printf("Time per lattice point: %4.3f micro sec\n",wdt);
               printf("%d Mflops [%d bit arithmetic]\n\n",
                      (int)(96.0/wdt),(int)(sizeof(spinor_dble))/3);
            }
         }

         wdt=1.0e6*wt_project_dble(nflds,vol,icom)/(double)(vol);

         if (my_rank==0)
         {
            printf("Function project_dble:\n");
            printf("Time per lattice point: %4.3f micro sec\n",wdt);
            printf("%d Mflops [%d bit arithmetic]\n\n",
                   (int)(192.0/wdt),(int)(sizeof(spinor_dble))/3);
         }
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
