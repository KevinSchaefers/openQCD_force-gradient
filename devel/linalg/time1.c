
/*******************************************************************************
*
* File time1.c
*
* Copyright (C) 2005, 2008, 2011, 2013, 2016, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of the salg routines.
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

static spinor **ps;


static double wt_spinor_prod(int nflds,int vol,int icom)
{
   int my_rank,nmax,n,i,ib;
   double wt1,wt2,wdt,wtav;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   for (i=0;i<nflds;i++)
      random_s(vol,icom,ps[i],1.0f);

   nmax=1;

   for (ib=0;ib<1;nmax*=2)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      for (n=0;n<nmax;n++)
      {
         for (i=0;i<nflds;i+=2)
            (void)(spinor_prod(vol,icom,ps[i],ps[i+1]));
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


static double wt_norm_square(int nflds,int vol,int icom)
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
            (void)(norm_square(vol,icom,ps[i]));
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


static double wt_normalize(int nflds,int vol,int icom)
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
            (void)(normalize(vol,icom,ps[i]));
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


static double wt_mulc_spinor_add(int nflds,int vol,int icom)
{
   int my_rank,nmax,n,i,ib;
   complex z;
   double wt1,wt2,wdt,wtav;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   z.re=0.123f;
   z.im=0.456f;
   nmax=1;

   for (ib=0;ib<1;nmax*=2)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      for (n=0;n<nmax;n++)
      {
         for (i=0;i<nflds;i+=2)
            mulc_spinor_add(vol,icom,ps[i],ps[i+1],z);

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


static double wt_project(int nflds,int vol,int icom)
{
   int my_rank,nmax,n,i,ib;
   double wt1,wt2,wdt,wtav;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   for (i=0;i<nflds;i++)
   {
      random_s(vol,icom,ps[i],1.0f);
      (void)(normalize(vol,icom,ps[i]));
   }

   nmax=1;

   for (ib=0;ib<1;nmax*=2)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      for (n=0;n<nmax;n++)
      {
         for (i=0;i<nflds;i+=2)
            project(vol,icom,ps[i],ps[i+1]);
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
      flog=freopen("time1.log","w",stdout);

      printf("\n");
      printf("Timing of the salg routines\n");
      printf("---------------------------\n\n");

      print_lattice_sizes();

      if ((VOLUME*sizeof(float))<(64*1024))
         printf("The local size of a quark field is %d KB\n",
                (int)((24*VOLUME*sizeof(float))/(1024)));
      else
         printf("The local size of a quark field is %d MB\n",
                (int)((24*VOLUME*sizeof(float))/(1024*1024)));

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

   nflds=(int)((4*1024*1024)/(VOLUME_TRD*sizeof(float)))+1;
   if ((nflds%2)==1)
      nflds+=1;
   if (nflds<10)
      nflds=10;
   alloc_ws(nflds);
   ps=reserve_ws(nflds);

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

         wdt=1.0e6*wt_spinor_prod(nflds,vol,icom)/(double)(vol);

         if (my_rank==0)
         {
            printf("Function spinor_prod:\n");
            printf("Time per lattice point: %4.3f micro sec\n",wdt);
            printf("%d Mflops [%d bit arithmetic]\n\n",
                   (int)(96.0/wdt),(int)(sizeof(spinor))/3);
         }

         wdt=1.0e6*wt_norm_square(nflds,vol,icom)/(double)(vol);

         if (my_rank==0)
         {
            printf("Function norm_square:\n");
            printf("Time per lattice point: %4.3f micro sec\n",wdt);
            printf("%d Mflops [%d bit arithmetic]\n\n",
                   (int)(48.0/wdt),(int)(sizeof(spinor))/3);
         }

         wdt=1.0e6*wt_normalize(nflds,vol,icom)/(double)(vol);

         if (my_rank==0)
         {
            printf("Function normalize:\n");
            printf("Time per lattice point: %4.3f micro sec\n",wdt);
            printf("%d Mflops [%d bit arithmetic]\n\n",
                   (int)(72.0/wdt),(int)(sizeof(spinor))/3);
         }

         if ((icom&0x1)==0)
         {
            wdt=1.0e6*wt_mulc_spinor_add(nflds,vol,icom)/(double)(vol);

            if (my_rank==0)
            {
               printf("Function mulc_spinor_add:\n");
               printf("Time per lattice point: %4.3f micro sec\n",wdt);
               printf("%d Mflops [%d bit arithmetic]\n\n",
                      (int)(96.0/wdt),(int)(sizeof(spinor))/3);
            }
         }

         wdt=1.0e6*wt_project(nflds,vol,icom)/(double)(vol);

         if (my_rank==0)
         {
            printf("Function project:\n");
            printf("Time per lattice point: %4.3f micro sec\n",wdt);
            printf("%d Mflops [%d bit arithmetic]\n\n",
                   (int)(192.0/wdt),(int)(sizeof(spinor))/3);
         }
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
