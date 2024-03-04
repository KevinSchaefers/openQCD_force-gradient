
/*******************************************************************************
*
* File check5.c
*
* Copyright (C) 2015, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the programs dft4d() and inv_dft4d() (boundary values).
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "utils.h"
#include "random.h"
#include "lattice.h"
#include "dft.h"
#include "global.h"

#define NTEST 8

extern int dpgen(void);
extern void print_dft4d_parms(int id);

static const int nproc[4]={NPROC0,NPROC1,NPROC2,NPROC3};
static int cnt;
static complex_dble *fs=NULL,*fts;


static void set_fs(int id)
{
   int csize,m,nx[4];
   dft4d_parms_t *dp4d;

   dp4d=dft4d_parms(id);
   csize=(*dp4d).csize;
   nx[0]=(*dp4d).nx[0][cpr[0]];
   nx[1]=(*dp4d).nx[1][cpr[1]];
   nx[2]=(*dp4d).nx[2][cpr[2]];
   nx[3]=(*dp4d).nx[3][cpr[3]];

   if (fs!=NULL)
      afree(fs);

   m=csize*nx[0]*nx[1]*nx[2]*nx[3];
   fs=amalloc(2*m*sizeof(*fs),4);
   error(fs==NULL,1,"set_f [check5.c]","Unable to allocate field array");

   fts=fs+m;
   gauss_dble((double*)(fs),4*m);
}


static double cmp0(int n,complex_dble *f0)
{
   int i;
   double dev,dmx;

   dmx=0.0;

   for (i=0;i<n;i++)
   {
      dev=fabs(f0[i].re)+fabs(f0[i].im);
      if (dev>dmx)
         dmx=dev;
   }

   cnt+=1;

   return dmx;
}


static double cmp1(int n,complex_dble *f0,complex_dble *f1)
{
   int i;
   double dev,dmx;

   dmx=0.0;

   for (i=0;i<n;i++)
   {
      dev=fabs(f0[i].re-f1[i].re)+fabs(f0[i].im-f1[i].im);
      if (dev>dmx)
         dmx=dev;
   }

   cnt+=1;

   return dmx;
}


static double cmp2(int n,complex_dble *f0,complex_dble *f1)
{
   int i;
   double dev,dmx;

   dmx=0.0;

   for (i=0;i<n;i++)
   {
      dev=fabs(f0[i].re+f1[i].re)+fabs(f0[i].im+f1[i].im);
      if (dev>dmx)
         dmx=dev;
   }

   cnt+=1;

   return dmx;
}


static double chk_bnd0(int id)
{
   int csize,m,vol,nx[4];
   int i,ix,iy,mu,b,c,d,x[4];
   double dev,nrm,dmx,nmx;
   dft4d_parms_t *dp4d;
   dft_parms_t *dp;

   dp4d=dft4d_parms(id);
   csize=(*dp4d).csize;
   nx[0]=(*dp4d).nx[0][cpr[0]];
   nx[1]=(*dp4d).nx[1][cpr[1]];
   nx[2]=(*dp4d).nx[2][cpr[2]];
   nx[3]=(*dp4d).nx[3][cpr[3]];
   vol=nx[0]*nx[1]*nx[2]*nx[3];
   m=csize*vol;
   nmx=0.0;

   for (i=0;i<m;i++)
      nmx+=fabs(fts[i].re)+fabs(fts[i].im);

   nmx/=(double)(m);

   if (NPROC>1)
   {
      nrm=nmx;
      MPI_Reduce(&nrm,&nmx,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&nmx,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      nmx/=(double)(NPROC);
   }

   dmx=0.0;

   for (mu=0;mu<4;mu++)
   {
      dp=(*dp4d).dp[mu];

      if ((*dp).type!=EXP)
      {
         b=(*dp).b;
         c=(*dp).c;
         d=((*dp).type==SIN);

         for (ix=0;ix<vol;ix++)
         {
            iy=ix;
            x[3]=iy%nx[3];
            iy/=nx[3];
            x[2]=iy%nx[2];
            iy/=nx[2];
            x[1]=iy%nx[1];
            x[0]=iy/nx[1];

            if ((b==1)&&(cpr[mu]==(nproc[mu]-1))&&
                (x[mu]==(nx[mu]-1)))
            {
               if (mu==0)
                  iy=ix-nx[1]*nx[2]*nx[3];
               else if (mu==1)
                  iy=ix-nx[2]*nx[3];
               else if (mu==2)
                  iy=ix-nx[3];
               else
                  iy=ix-1;

               if ((c+d)==1)
                  dev=cmp2(csize,fts+csize*ix,fts+csize*iy);
               else
                  dev=cmp1(csize,fts+csize*ix,fts+csize*iy);

               if (dev>dmx)
                  dmx=dev;
            }

            if ((b==0)&&(d==1)&&(cpr[mu]==0)&&
                (x[mu]==0))
            {
               dev=cmp0(csize,fts+csize*ix);

               if (dev>dmx)
                  dmx=dev;
            }

            if ((b==0)&&((c+d)==1)&&(cpr[mu]==(nproc[mu]-1))&&
                (x[mu]==(nx[mu]-1)))
            {
               dev=cmp0(csize,fts+csize*ix);

               if (dev>dmx)
                  dmx=dev;
            }
         }
      }
   }

   if (NPROC>1)
   {
      dev=dmx;
      MPI_Reduce(&dev,&dmx,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmx,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      i=cnt;
      MPI_Reduce(&i,&cnt,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&cnt,1,MPI_INT,0,MPI_COMM_WORLD);
   }

   return dmx/nmx;
}


static double chk_bnd1(int id)
{
   int ix,csize,vol,nx[4];
   double nrm,nmx,dev,dmx;
   dft4d_parms_t *dp4d;

   dp4d=dft4d_parms(id);
   csize=(*dp4d).csize;
   nx[0]=(*dp4d).nx[0][cpr[0]];
   nx[1]=(*dp4d).nx[1][cpr[1]];
   nx[2]=(*dp4d).nx[2][cpr[2]];
   nx[3]=(*dp4d).nx[3][cpr[3]];
   vol=nx[0]*nx[1]*nx[2]*nx[3];
   nmx=0.0;
   dmx=0.0;

   for (ix=0;ix<vol;ix++)
   {
      nmx+=cmp0(csize,fts+csize*ix);
      dev=cmp1(csize,fs+csize*ix,fts+csize*ix);
      if (dev>dmx)
         dmx=dev;
   }

   nmx/=(double)(vol);

   if (NPROC>1)
   {
      nrm=nmx;
      MPI_Reduce(&nrm,&nmx,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&nmx,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      nmx/=(double)(NPROC);

      dev=dmx;
      MPI_Reduce(&dev,&dmx,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmx,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }

   return dmx/nmx;
}


static void random_bnd(int id)
{
   int csize,vol,nx[4];
   int ix,iy,mu,b,c,d,x[4];
   dft4d_parms_t *dp4d;
   dft_parms_t *dp;

   dp4d=dft4d_parms(id);
   csize=(*dp4d).csize;
   nx[0]=(*dp4d).nx[0][cpr[0]];
   nx[1]=(*dp4d).nx[1][cpr[1]];
   nx[2]=(*dp4d).nx[2][cpr[2]];
   nx[3]=(*dp4d).nx[3][cpr[3]];
   vol=nx[0]*nx[1]*nx[2]*nx[3];

   for (mu=0;mu<4;mu++)
   {
      dp=(*dp4d).dp[mu];

      if ((*dp).type!=EXP)
      {
         b=(*dp).b;
         c=(*dp).c;
         d=((*dp).type==SIN);

         for (ix=0;ix<vol;ix++)
         {
            iy=ix;
            x[3]=iy%nx[3];
            iy/=nx[3];
            x[2]=iy%nx[2];
            iy/=nx[2];
            x[1]=iy%nx[1];
            x[0]=iy/nx[1];

            if ((b==1)&&(cpr[mu]==(nproc[mu]-1))&&
                (x[mu]==(nx[mu]-1)))
            {
               gauss_dble((double*)(fts+csize*ix),2*csize);
            }

            if ((b==0)&&(d==1)&&(cpr[mu]==0)&&
                (x[mu]==0))
            {
               gauss_dble((double*)(fts+csize*ix),2*csize);
            }

            if ((b==0)&&((c+d)==1)&&(cpr[mu]==(nproc[mu]-1))&&
                (x[mu]==(nx[mu]-1)))
            {
               gauss_dble((double*)(fts+csize*ix),2*csize);
            }
         }
      }
   }
}


int main(int argc,char *argv[])
{
   int my_rank;
   int it,id;
   double dmax[2],dall[2];
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check5.log","w",stdout);

      printf("\n");
      printf("Check of the programs dft4d() and inv_dft4d()"
             " (boundary values)\n");
      printf("---------------------------------------------"
             "------------------\n\n");

      print_lattice_sizes();
      fflush(flog);
   }

   start_ranlux(0,12345);
   geometry();
   dall[0]=0.0;
   dall[1]=0.0;

   for (it=0;it<NTEST;it++)
   {
      cnt=0;
      id=dpgen();
      print_dft4d_parms(id);

      set_fs(id);
      dft4d(id,fs,fts);
      dmax[0]=chk_bnd0(id);

      inv_dft4d(id,fts,fs);
      random_bnd(id);
      inv_dft4d(id,fts,fts);
      dmax[1]=chk_bnd1(id);

      if (my_rank==0)
      {
         printf("Deviation (ft bnd) = %.1e (bnd count = %d)\n",dmax[0],cnt);
         printf("Deviation (f bnd)  = %.1e\n\n",dmax[1]);
      }

      if (dmax[0]>dall[0])
         dall[0]=dmax[0];

      if (dmax[1]>dall[1])
         dall[1]=dmax[1];
   }

   if (my_rank==0)
   {
      printf("Maximal deviation (ft bnd) = %.1e\n",dall[0]);
      printf("Maximal deviation (f bnd)  = %.1e\n",dall[1]);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
