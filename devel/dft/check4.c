
/*******************************************************************************
*
* File check4.c
*
* Copyright (C) 2015, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the programs dft4d() and inv_dft4d().
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

static double *phat[4]={NULL,NULL,NULL,NULL};
static complex_dble *f=NULL,*ft,*ftt;


static void set_f(int id)
{
   int csize,nx[4];
   int m,b,c,d,i0,i1,i;
   dft4d_parms_t *dp;
   dft_parms_t *dpf;

   dp=dft4d_parms(id);
   csize=(*dp).csize;
   nx[0]=(*dp).nx[0][cpr[0]];
   nx[1]=(*dp).nx[1][cpr[1]];
   nx[2]=(*dp).nx[2][cpr[2]];
   nx[3]=(*dp).nx[3][cpr[3]];

   if (f!=NULL)
      afree(f);

   m=csize*nx[0]*nx[1]*nx[2]*nx[3];
   f=amalloc(3*m*sizeof(*f),4);
   error(f==NULL,1,"set_f [check4.c]","Unable to allocate field arrays");

   ft=f+m;
   ftt=ft+m;

   gauss_dble((double*)(f),6*m);
   dpf=(*dp).dp[0];

   if ((*dpf).type!=EXP)
   {
      b=(*dpf).b;
      c=(*dpf).c;

      if ((*dpf).type==COS)
         d=0;
      else
         d=1;

      if ((cpr[0]==0)&&(c==0)&&(d==1))
      {
         for (i=0;i<(csize*nx[1]*nx[2]*nx[3]);i++)
         {
            f[i].re=0.0;
            f[i].im=0.0;
         }
      }

      if ((cpr[0]==(NPROC0-1))&&(c==0)&&((b+d)==1))
      {
         i0=csize*(nx[0]-1)*nx[1]*nx[2]*nx[3];

         for (i=0;i<(csize*nx[1]*nx[2]*nx[3]);i++)
         {
            f[i0+i].re=0.0;
            f[i0+i].im=0.0;
         }
      }

      if ((cpr[0]==(NPROC0-1))&&(c==1))
      {
         i0=csize*(nx[0]-1)*nx[1]*nx[2]*nx[3];
         i1=csize*(nx[0]-2)*nx[1]*nx[2]*nx[3];

         for (i=0;i<(csize*nx[1]*nx[2]*nx[3]);i++)
         {
            if ((b+d)==1)
            {
               f[i0+i].re=-f[i1+i].re;
               f[i0+i].im=-f[i1+i].im;
            }
            else
            {
               f[i0+i].re=f[i1+i].re;
               f[i0+i].im=f[i1+i].im;
            }
         }
      }
   }
}


static double cmp0(int id)
{
   int i,m,csize,nx[4];
   double d,r,dmx,rmx;
   dft4d_parms_t *dp;

   dp=dft4d_parms(id);
   csize=(*dp).csize;
   nx[0]=(*dp).nx[0][cpr[0]];
   nx[1]=(*dp).nx[1][cpr[1]];
   nx[2]=(*dp).nx[2][cpr[2]];
   nx[3]=(*dp).nx[3][cpr[3]];

   m=csize*nx[0]*nx[1]*nx[2]*nx[3];
   dmx=0.0;
   rmx=0.0;

   for (i=0;i<m;i++)
   {
      r=fabs(f[i].re)+fabs(f[i].im);
      if (r>rmx)
         rmx=r;

      d=fabs(f[i].re-ftt[i].re)+fabs(f[i].im-ftt[i].im);
      if (d>dmx)
         dmx=d;
   }

   d=dmx/rmx;

   MPI_Reduce(&d,&dmx,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Bcast(&dmx,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   return dmx;
}


static void set_phat(int id)
{
   int mu,k,n[4],b[4];
   double pi,r0,r1;
   dft4d_parms_t *dp;

   dp=dft4d_parms(id);

   for (mu=0;mu<4;mu++)
   {
      n[mu]=(*dp).dp[mu][0].n;
      b[mu]=(*dp).dp[mu][0].b;
   }

   if (phat[0]!=NULL)
      free(phat[0]);

   k=n[0]+n[1]+n[2]+n[3]+4;
   phat[0]=malloc(k*sizeof(*(phat[0])));
   error(phat[0]==NULL,1,"set_phat [check4.c]",
         "Unable to allocate auxiliary array");
   phat[1]=phat[0]+n[0]+1;
   phat[2]=phat[1]+n[1]+1;
   phat[3]=phat[2]+n[2]+1;

   pi=4.0*atan(1.0);

   for (mu=0;mu<4;mu++)
   {
      if ((*dp).dp[mu][0].type==EXP)
         r0=2.0*pi/(double)(n[mu]);
      else
         r0=pi/(double)(n[mu]);

      for (k=0;k<=n[mu];k++)
      {
         r1=2.0*sin(0.25*r0*(double)(2*k+b[mu]));
         phat[mu][k]=r1*r1;
      }
   }
}


static void apply_Delta(int id)
{
   int csize,nk[4];
   int k0,k1,k2,k3,s,ik;
   double p0,p1,p2,p3,psq;
   dft4d_parms_t *dp;

   set_phat(id);
   dp=dft4d_parms(id);
   csize=(*dp).csize;
   nk[0]=(*dp).nx[0][cpr[0]];
   nk[1]=(*dp).nx[1][cpr[1]];
   nk[2]=(*dp).nx[2][cpr[2]];
   nk[3]=(*dp).nx[3][cpr[3]];

   for (k0=0;k0<nk[0];k0++)
   {
      for (k1=0;k1<nk[1];k1++)
      {
         for (k2=0;k2<nk[2];k2++)
         {
            for (k3=0;k3<nk[3];k3++)
            {
               p0=phat[0][k0+cpr[0]*L0];
               p1=phat[1][k1+cpr[1]*L1];
               p2=phat[2][k2+cpr[2]*L2];
               p3=phat[3][k3+cpr[3]*L3];

               ik=k3+nk[3]*k2+nk[2]*nk[3]*k1+nk[1]*nk[2]*nk[3]*k0;
               psq=p0+p1+p2+p3;

               for (s=0;s<csize;s++)
               {
                  ft[s+csize*ik].re*=psq;
                  ft[s+csize*ik].im*=psq;
               }
            }
         }
      }
   }
}


static double cmp1(int id)
{
   int csize,nx[4];
   int x0,x1,x2,x3,s;
   int ix,ixp[4],ixm[4];
   double d,r,dmx,rmx;
   complex_dble z;
   dft4d_parms_t *dp;

   dp=dft4d_parms(id);
   csize=(*dp).csize;
   nx[0]=(*dp).nx[0][cpr[0]];
   nx[1]=(*dp).nx[1][cpr[1]];
   nx[2]=(*dp).nx[2][cpr[2]];
   nx[3]=(*dp).nx[3][cpr[3]];

   dmx=0.0;
   rmx=0.0;

   for (x0=1;x0<(nx[0]-1);x0++)
   {
      for (x1=1;x1<(nx[1]-1);x1++)
      {
         for (x2=1;x2<(nx[2]-1);x2++)
         {
            for (x3=1;x3<(nx[3]-1);x3++)
            {
               ix=x3+nx[3]*x2+nx[2]*nx[3]*x1+nx[1]*nx[2]*nx[3]*x0;

               ixp[0]=ix+nx[1]*nx[2]*nx[3];
               ixm[0]=ix-nx[1]*nx[2]*nx[3];

               ixp[1]=ix+nx[2]*nx[3];
               ixm[1]=ix-nx[2]*nx[3];

               ixp[2]=ix+nx[3];
               ixm[2]=ix-nx[3];

               ixp[3]=ix+1;
               ixm[3]=ix-1;

               for (s=0;s<csize;s++)
               {
                  z.re=
                     f[s+csize*ixp[0]].re+f[s+csize*ixm[0]].re+
                     f[s+csize*ixp[1]].re+f[s+csize*ixm[1]].re+
                     f[s+csize*ixp[2]].re+f[s+csize*ixm[2]].re+
                     f[s+csize*ixp[3]].re+f[s+csize*ixm[3]].re-
                     8.0*f[s+csize*ix].re;

                  z.im=
                     f[s+csize*ixp[0]].im+f[s+csize*ixm[0]].im+
                     f[s+csize*ixp[1]].im+f[s+csize*ixm[1]].im+
                     f[s+csize*ixp[2]].im+f[s+csize*ixm[2]].im+
                     f[s+csize*ixp[3]].im+f[s+csize*ixm[3]].im-
                     8.0*f[s+csize*ix].im;

                  r=fabs(ftt[s+csize*ix].re)+fabs(ftt[s+csize*ix].im);
                  d=fabs(ftt[s+csize*ix].re+z.re)+fabs(ftt[s+csize*ix].im+z.im);

                  if (r>rmx)
                     rmx=r;
                  if (d>dmx)
                     dmx=d;
               }
            }
         }
      }
   }

   d=dmx/rmx;

   MPI_Reduce(&d,&dmx,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Bcast(&dmx,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   return dmx;
}


int main(int argc,char *argv[])
{
   int my_rank;
   int t,b,c;
   int idp[4],nx[4],csize,id;
   double dmax,dall;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check4.log","w",stdout);

      printf("\n");
      printf("Check of the programs dft4d() and inv_dft4d()\n");
      printf("---------------------------------------------\n\n");

      print_lattice_sizes();
      fflush(flog);
   }

   start_ranlux(0,12345);
   geometry();
   dall=0.0;

   for (t=0;t<(int)(DFT_TYPES);t++)
   {
      if (my_rank==0)
      {
         if ((dft_type_t)(t)==EXP)
            printf("Type = EXP\n");
         else if ((dft_type_t)(t)==SIN)
            printf("Type = SIN\n");
         else if ((dft_type_t)(t)==COS)
            printf("Type = COS\n");
         else
            error_root(1,1,"main [check4.c]",
                       "Unknown transformation type");
      }

      for (b=0;b<2;b++)
      {
         for (c=0;c<2;c++)
         {
            idp[0]=set_dft_parms((dft_type_t)(t),NPROC0*L0,b,c);
            idp[1]=set_dft_parms(EXP,NPROC1*L1,b^0x1,c);
            idp[2]=set_dft_parms(EXP,NPROC2*L2,b,c^0x1);
            idp[3]=set_dft_parms(EXP,NPROC3*L3,b^0x1,c^0x1);

            if ((cpr[0]==(NPROC0-1))&&((dft_type_t)(t)!=EXP))
               nx[0]=L0+1;
            else
               nx[0]=L0;
            nx[1]=L1;
            nx[2]=L2;
            nx[3]=L3;
            csize=3;

            id=set_dft4d_parms(idp,nx,csize);
            set_f(id);
            dft4d(id,f,ft);
            inv_dft4d(id,ft,ftt);
            dmax=cmp0(id);
            if (dmax>dall)
               dall=dmax;

            if (my_rank==0)
               printf("Maximal deviation (dft4d + inv_dft4d) "
                      "(b=%d,c=%d) = %.1e\n",b,c,dmax);

            set_f(id);
            dft4d(id,f,ft);
            apply_Delta(id);
            inv_dft4d(id,ft,ftt);
            dmax=cmp1(id);
            if (dmax>dall)
               dall=dmax;

            if (my_rank==0)
               printf("Maximal deviation (dft4d + Delta + inv_dft4d) "
                      "  = %.1e\n",dmax);

            dft4d(id,f,ftt);
            dft4d(id,f,f);
            dmax=cmp0(id);
            if (dmax>dall)
               dall=dmax;

            if (my_rank==0)
               printf("Maximal deviation (in place dft4d)            "
                      "  = %.1e\n",dmax);
         }
      }

      if (my_rank==0)
         printf("\n");
   }

   if (my_rank==0)
   {
      printf("Maximal deviation = %.1e\n\n",dall);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
