
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2017, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the convolution and shift programs.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "random.h"
#include "lattice.h"
#include "linalg.h"
#include "msfcts.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int ls[4]={L0,L1,L2,L3};
static int ns[4]={N0,N1,N2,N3};
static double zero[3]={0.0,0.0,0.0};
static double ps[4],ds[4];
static double **f;
static complex_dble **rf;


static void alloc_msfld(int nf,int nrf)
{
   int i;
   double *f1,**f2;
   complex_dble *rf1,**rf2;

   f1=malloc(nf*VOLUME*sizeof(*f1));
   f2=malloc(nf*sizeof(*f2));

   rf1=malloc(nrf*VOLUME*sizeof(*rf1));
   rf2=malloc(nrf*sizeof(*rf2));

   error((f1==NULL)||(f2==NULL)||(rf1==NULL)||(rf2==NULL),1,
         "alloc_msfld [check1.c]","Unable to allocate fields");

   f=f2;
   rf=rf2;

   for (i=0;i<nf;i++)
   {
      f2[0]=f1;
      f1+=VOLUME;
      f2+=1;
   }

   for (i=0;i<nrf;i++)
   {
      rf2[0]=rf1;
      rf1+=VOLUME;
      rf2+=1;
   }
}


static void random_s(int *s)
{
   int mu;
   double r[4];

   ranlxd(r,4);

   for (mu=0;mu<4;mu++)
      s[mu]=(int)(r[mu]*(double)(ns[mu]));

   MPI_Bcast(s,4,MPI_INT,0,MPI_COMM_WORLD);
}


static void random_ps(void)
{
   int mu,k;
   double pi;

   pi=4.0*atan(1.0);
   ranlxd(ps,4);
   ranlxd(ds,4);

   for (mu=0;mu<4;mu++)
   {
      k=(int)(ps[mu]*(double)(ns[mu]));

      if ((2*k)>ns[mu])
         k-=ns[mu];

      ps[mu]=(double)(k)/(2.0*pi);

      ds[mu]-=0.5;
      ds[mu]*=(2*pi);
   }

   MPI_Bcast(ps,4,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(ds,4,MPI_DOUBLE,0,MPI_COMM_WORLD);
}


static double fval(int *x)
{
   int mu,z;
   double r;

   r=0.0;

   for (mu=0;mu<4;mu++)
   {
      z=x[mu]%ns[mu];
      if ((2*z)>ns[mu])
         z-=ns[mu];
      r+=cos(ps[mu]*(double)(z)+ds[mu]);
   }

   return exp(0.25*r);
}


static void set_fs(int *s,double *fs)
{
   int ix,iy,x[4];

   for (ix=0;ix<VOLUME;ix++)
   {
      iy=ix;
      x[3]=(iy%L3)+cpr[3]*ls[3]+s[3];
      iy/=L3;
      x[2]=(iy%L2)+cpr[2]*ls[2]+s[2];
      iy/=L2;
      x[1]=(iy%L1)+cpr[1]*ls[1]+s[1];
      iy/=L1;
      x[0]=iy+cpr[0]*ls[0]+s[0];

      fs[ipt[ix]]=fval(x);
   }
}


static double scalar_prod(double *fs,double *gs)
{
   complex_qflt cqsm;

   cqsm=vprod_dble(VOLUME_TRD/2,3,(complex_dble*)(fs),(complex_dble*)(gs));

   return cqsm.re.q[0];
}


static double max_dev(double *fs,double *gs)
{
   int ix;
   double r,ravg,d,dmax;

   ravg=0.0;
   dmax=0.0;

   for (ix=0;ix<VOLUME;ix++)
   {
      r=fabs(fs[ix]);
      ravg+=r;

      d=fabs(fs[ix]-gs[ix]);
      if (d>dmax)
         dmax=d;
   }

   if (NPROC>1)
   {
      r=ravg;
      d=dmax;

      MPI_Reduce(&r,&ravg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&ravg,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Reduce(&d,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }

   ravg/=((double)(N0*N1)*(double)(N2*N3));

   return dmax/ravg;
}


int main(int argc,char *argv[])
{
   int my_rank,i,j;
   int ip,ix,s[4],r[4];
   double pi,r1,r2,d,dall,v;
   double qs[4],es[4];
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);

      printf("\n");
      printf("Check of the convolution and shift programs\n");
      printf("-------------------------------------------\n\n");

      print_lattice_sizes();
   }

   check_machine();
   set_bc_parms(3,1.0,1.0,1.0,1.0,zero,zero,zero);

   start_ranlux(0,12345);
   geometry();

   alloc_msfld(3,2);
   pi=4.0*atan(1.0);
   v=(double)(N0*N1)*(double)(N2*N3);
   dall=0.0;

   if (my_rank==0)
   {
      printf("Self-correlation of a single field\n");
      printf("----------------------------------\n\n");
   }

   for (i=0;i<4;i++)
   {
      random_ps();

      if (my_rank==0)
      {
         printf("Random momentum = (%+3d,%+3d,%+3d,%+3d)\n",
                (int)(2*pi*ps[0]+0.1),(int)(2*pi*ps[1]+0.1),
                (int)(2*pi*ps[2]+0.1),(int)(2*pi*ps[3]+0.1));
      }

      s[0]=0;
      s[1]=0;
      s[2]=0;
      s[3]=0;
      set_fs(s,f[1]);

      if (i==0)
      {
         r[0]=0;
         r[1]=0;
         r[2]=0;
         r[3]=0;
         convolute_msfld(NULL,f[1],f[1],rf[0],rf[0],f[2]);
      }
      else
      {
         random_s(r);
         convolute_msfld(r,f[1],f[1],rf[0],rf[0],f[2]);
      }

      for (j=0;j<4;j++)
      {
         random_s(s);

         if (my_rank==0)
         {
            if (j==0)
               printf("r=(%2d,%2d,%2d,%2d), s=(%2d,%2d,%2d,%2d), ",
                      r[0],r[1],r[2],r[3],s[0],s[1],s[2],s[3]);
            else
               printf("                 s=(%2d,%2d,%2d,%2d), ",
                      s[0],s[1],s[2],s[3]);
         }

         ipt_global(s,&ip,&ix);
         r1=f[2][ix]/v;
         MPI_Bcast(&r1,1,MPI_DOUBLE,ip,MPI_COMM_WORLD);

         s[0]+=r[0];
         s[1]+=r[1];
         s[2]+=r[2];
         s[3]+=r[3];

         set_fs(s,f[0]);
         r2=scalar_prod(f[0],f[1])/v;

         if (my_rank==0)
         {
            printf("r1=% .16e, r2=% .16e\n",r1,r2);

            d=fabs(r1/r2-1.0);
            if (d>dall)
               dall=d;
         }
      }

      if (my_rank==0)
         printf("\n");
   }

   if (my_rank==0)
   {
      printf("Correlation of two different fields\n");
      printf("-----------------------------------\n\n");
   }

   for (i=0;i<4;i++)
   {
      s[0]=0;
      s[1]=0;
      s[2]=0;
      s[3]=0;

      random_ps();
      set_fs(s,f[0]);

      for (j=0;j<4;j++)
      {
         qs[j]=ps[j];
         es[j]=ds[j];
      }

      random_ps();
      set_fs(s,f[1]);

      if (i==0)
      {
         r[0]=0;
         r[1]=0;
         r[2]=0;
         r[3]=0;
         convolute_msfld(NULL,f[0],f[1],rf[0],rf[1],f[2]);
      }
      else
      {
         random_s(r);
         convolute_msfld(r,f[0],f[1],rf[0],rf[1],f[2]);
      }

      if (my_rank==0)
      {
         printf("Random momenta = (%+3d,%+3d,%+3d,%+3d), ",
                (int)(2*pi*qs[0]+0.1),(int)(2*pi*qs[1]+0.1),
                (int)(2*pi*qs[2]+0.1),(int)(2*pi*qs[3]+0.1));

         printf("(%+3d,%+3d,%+3d,%+3d)\n",
                (int)(2*pi*ps[0]+0.1),(int)(2*pi*ps[1]+0.1),
                (int)(2*pi*ps[2]+0.1),(int)(2*pi*ps[3]+0.1));
      }

      for (j=0;j<4;j++)
      {
         ps[j]=qs[j];
         ds[j]=es[j];
      }

      for (j=0;j<4;j++)
      {
         random_s(s);

         if (my_rank==0)
         {
            if (j==0)
               printf("r=(%2d,%2d,%2d,%2d), s=(%2d,%2d,%2d,%2d), ",
                      r[0],r[1],r[2],r[3],s[0],s[1],s[2],s[3]);
            else
               printf("                 s=(%2d,%2d,%2d,%2d), ",
                      s[0],s[1],s[2],s[3]);
         }

         ipt_global(s,&ip,&ix);
         r1=f[2][ix]/v;
         MPI_Bcast(&r1,1,MPI_DOUBLE,ip,MPI_COMM_WORLD);

         s[0]+=r[0];
         s[1]+=r[1];
         s[2]+=r[2];
         s[3]+=r[3];

         set_fs(s,f[0]);
         r2=scalar_prod(f[0],f[1])/v;

         if (my_rank==0)
         {
            printf("r1=% .16e, r2=% .16e\n",r1,r2);

            d=fabs(r1/r2-1.0);
            if (d>dall)
               dall=d;
         }
      }

      if (my_rank==0)
         printf("\n");
   }

   if (my_rank==0)
   {
      printf("Field translations\n");
      printf("------------------\n\n");
   }

   for (i=0;i<4;i++)
   {
      random_ps();

      if (my_rank==0)
      {
         printf("Random momentum = (%+3d,%+3d,%+3d,%+3d)\n",
                (int)(2*pi*ps[0]+0.1),(int)(2*pi*ps[1]+0.1),
                (int)(2*pi*ps[2]+0.1),(int)(2*pi*ps[3]+0.1));
      }

      s[0]=0;
      s[1]=0;
      s[2]=0;
      s[3]=0;
      set_fs(s,f[0]);

      for (j=0;j<4;j++)
      {
         random_s(s);
         set_fs(s,f[1]);
         shift_msfld(s,f[0],rf[0],f[2]);
         d=max_dev(f[1],f[2]);

         if (my_rank==0)
         {
            printf("s=(%2d,%2d,%2d,%2d), ",s[0],s[1],s[2],s[3]);
            printf("deviation=%.1e\n",d);

            if (d>dall)
               dall=d;
         }
      }

      if (my_rank==0)
         printf("\n");
   }

   if (my_rank==0)
   {
      printf("Maximal relative deviation = %.1e\n\n",dall);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
