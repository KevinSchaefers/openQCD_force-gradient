
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2015, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the programs fft() and inv_fft().
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
#include "dft.h"
#include "global.h"

static int ns[6]={10,16,24,36,40,144},nfs[6]={5,37,1,3,8,4};
static complex_dble **w=NULL;
static complex_dble **f=NULL,**ft,**ftt;


static void set_wn(dft_parms_t *dp)
{
   int n,b,c,i,j;
   double pi,r,s;
   complex_dble *w0;

   n=(*dp).n;
   b=(*dp).b;
   c=(*dp).c;

   if (w!=NULL)
   {
      afree(w[0]);
      free(w);
   }

   w=malloc((n+1)*sizeof(*w));
   w0=amalloc((n+1)*(n+1)*sizeof(*w0),4);
   error((w==NULL)||(w0==NULL),1,"set_wn [check2.c]",
         "Unable to allocate auxiliary array");
   w[0]=w0;

   for (i=1;i<=n;i++)
      w[i]=w[i-1]+n+1;

   pi=4.0*atan(1.0);

   if ((*dp).type==EXP)
      r=2.0*pi/(double)(n);
   else
      r=pi/(double)(n);

   for (i=0;i<=n;i++)
   {
      for (j=0;j<=n;j++)
      {
         s=r*0.25*(double)(2*i+b)*(double)(2*j+c);
         w[i][j].re=cos(s);
         w[i][j].im=sin(s);
      }
   }
}


static void set_fn(int n,int nfc)
{
   int i;
   complex_dble *f0;

   if (f!=NULL)
   {
      afree(f[0]);
      free(f);
   }

   f=malloc(3*(n+1)*sizeof(*f));
   f0=amalloc(3*(n+1)*nfc*sizeof(*f0),4);
   error((f==NULL)||(f0==NULL),1,"set_fn [check2.c]",
         "Unable to allocate function arrays");

   gauss_dble((double*)(f0),6*(n+1)*nfc);

   for (i=0;i<(3*(n+1));i++)
      f[i]=f0+i*nfc;

   ft=f+n+1;
   ftt=ft+n+1;
}


static double check_fft(dft_parms_t *dp,int nfc)
{
   int n,b,c;
   int i,k,x;
   double d,dmx,r,rmx;
   complex_dble sm,*z;

   n=(*dp).n;
   b=(*dp).b;
   c=(*dp).c;
   dmx=0.0;
   rmx=0.0;

   for (i=0;i<nfc;i++)
   {
      r=0.0;

      if ((*dp).type==EXP)
      {
         for (k=0;k<n;k++)
         {
            r+=fabs(f[k][i].re)+fabs(f[k][i].im);
            sm.re=0.0;
            sm.im=0.0;

            for (x=0;x<n;x++)
            {
               z=f[x]+i;
               sm.re+=(w[k][x].re*(*z).re-w[k][x].im*(*z).im);
               sm.im+=(w[k][x].re*(*z).im+w[k][x].im*(*z).re);
            }

            d=fabs(sm.re-ft[k][i].re)+fabs(sm.im-ft[k][i].im);

            if (d>dmx)
               dmx=d;
         }
      }
      else if ((*dp).type==COS)
      {
         if (c==0)
         {
            for (k=0;k<=n;k++)
            {
               if ((k<n)||(b==0))
                  r+=fabs(f[k][i].re)+fabs(f[k][i].im);

               sm.re=0.0;
               sm.im=0.0;

               for (x=0;x<=n;x++)
               {
                  if ((x<n)||(b==0))
                  {
                     z=f[x]+i;

                     if ((x==0)||(x==n))
                     {
                        sm.re+=w[k][x].re*(*z).re;
                        sm.im+=w[k][x].re*(*z).im;
                     }
                     else
                     {
                        sm.re+=2.0*w[k][x].re*(*z).re;
                        sm.im+=2.0*w[k][x].re*(*z).im;
                     }
                  }
               }

               d=fabs(sm.re-ft[k][i].re)+fabs(sm.im-ft[k][i].im);

               if (d>dmx)
                  dmx=d;
            }
         }
         else
         {
            for (k=0;k<=n;k++)
            {
               if (k<n)
                  r+=fabs(f[k][i].re)+fabs(f[k][i].im);

               sm.re=0.0;
               sm.im=0.0;

               for (x=0;x<n;x++)
               {
                  z=f[x]+i;
                  sm.re+=2.0*w[k][x].re*(*z).re;
                  sm.im+=2.0*w[k][x].re*(*z).im;
               }

               d=fabs(sm.re-ft[k][i].re)+fabs(sm.im-ft[k][i].im);

               if (d>dmx)
                  dmx=d;
            }
         }
      }
      else
      {
         if (c==0)
         {
            for (k=0;k<=n;k++)
            {
               if ((k>0)&&((k<n)||(b==1)))
                  r+=fabs(f[k][i].re)+fabs(f[k][i].im);

               sm.re=0.0;
               sm.im=0.0;

               for (x=0;x<=n;x++)
               {
                  if ((x>0)&&((x<n)||(b==1)))
                  {
                     z=f[x]+i;

                     if ((x==0)||(x==n))
                     {
                        sm.re-=w[k][x].im*(*z).im;
                        sm.im+=w[k][x].im*(*z).re;
                     }
                     else
                     {
                        sm.re-=2.0*w[k][x].im*(*z).im;
                        sm.im+=2.0*w[k][x].im*(*z).re;
                     }
                  }
               }

               d=fabs(sm.re-ft[k][i].re)+fabs(sm.im-ft[k][i].im);

               if (d>dmx)
                  dmx=d;
            }
         }
         else
         {
            for (k=0;k<=n;k++)
            {
               if (k<n)
                  r+=fabs(f[k][i].re)+fabs(f[k][i].im);

               sm.re=0.0;
               sm.im=0.0;

               for (x=0;x<n;x++)
               {
                  z=f[x]+i;
                  sm.re-=2.0*w[k][x].im*(*z).im;
                  sm.im+=2.0*w[k][x].im*(*z).re;
               }

               d=fabs(sm.re-ft[k][i].re)+fabs(sm.im-ft[k][i].im);

               if (d>dmx)
                  dmx=d;
            }
         }

      }

      if (r>rmx)
         rmx=r;
   }

   d=dmx/rmx;
   MPI_Reduce(&d,&dmx,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Bcast(&dmx,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   return dmx;
}


static double cmp_fn(dft_parms_t *dp,int nfc)
{
   int n,b,c,i,j;
   double d,dmx,rmx;
   complex_dble z1,z2;

   n=(*dp).n;
   b=(*dp).b;
   c=(*dp).c;

   dmx=0.0;
   rmx=0.0;

   if ((*dp).type==EXP)
   {
      for (i=0;i<nfc;i++)
      {
         for (j=0;j<n;j++)
         {
            z1=f[j][i];
            z2=ftt[j][i];

            rmx+=(fabs(z1.re)+fabs(z1.im));
            d=fabs(z1.re-z2.re)+fabs(z1.im-z2.im);
            if (d>dmx)
               dmx=d;
         }
      }
   }
   else if ((*dp).type==COS)
   {
      for (i=0;i<nfc;i++)
      {
         for (j=0;j<=n;j++)
         {
            z1=f[j][i];
            z2=ftt[j][i];

            if (j==n)
            {
               if ((b==1)&&(c==0))
               {
                  z1.re=0.0;
                  z1.im=0.0;
               }
               else if (c==1)
               {
                  z1=f[j-1][i];

                  if (b==1)
                  {
                     z1.re=-z1.re;
                     z1.im=-z1.im;
                  }
               }
            }

            rmx+=(fabs(z1.re)+fabs(z1.im));
            d=fabs(z1.re-z2.re)+fabs(z1.im-z2.im);
            if (d>dmx)
               dmx=d;
         }
      }
   }
   else
   {
      for (i=0;i<nfc;i++)
      {
         for (j=0;j<=n;j++)
         {
            z1=f[j][i];
            z2=ftt[j][i];

            if (j==0)
            {
               if (c==0)
               {
                  z1.re=0.0;
                  z1.im=0.0;
               }
            }
            else if (j==n)
            {
               if ((b==0)&&(c==0))
               {
                  z1.re=0.0;
                  z1.im=0.0;
               }
               else if (c==1)
               {
                  z1=f[j-1][i];

                  if (b==0)
                  {
                     z1.re=-z1.re;
                     z1.im=-z1.im;
                  }
               }
            }

            rmx+=(fabs(z1.re)+fabs(z1.im));
            d=fabs(z1.re-z2.re)+fabs(z1.im-z2.im);
            if (d>dmx)
               dmx=d;
         }
      }
   }

   rmx/=(double)(n*nfc);
   d=dmx/rmx;
   MPI_Reduce(&d,&dmx,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Bcast(&dmx,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   return dmx;
}


int main(int argc,char *argv[])
{
   int my_rank,np;
   int t,in,n,b,c,nfc,id;
   double d,dmx;
   dft_parms_t *dp;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   MPI_Comm_size(MPI_COMM_WORLD,&np);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);

      printf("\n");
      printf("Check of the programs fft() and inv_fft()\n");
      printf("-----------------------------------------\n\n");

      printf("Number of processes = %d\n\n",np);
      fflush(flog);
   }

   start_ranlux(0,1234);
   dmx=0.0;

   for (t=0;t<(int)(DFT_TYPES);t++)
   {
      for (in=0;in<6;in++)
      {
         for (b=0;b<2;b++)
         {
            for (c=0;c<2;c++)
            {
               n=ns[in];
               nfc=nfs[in];
               id=set_dft_parms((dft_type_t)(t),n,b,c);
               dp=dft_parms(id);

               set_wn(dp);
               set_fn(n,nfc);
               error_loc(fft(dp,nfc,f,ft)!=0,1,"main [check2.c]",
                         "fft() failed");
               d=check_fft(dp,nfc);
               if (d>dmx)
                  dmx=d;

               if (my_rank==0)
               {
                  if ((*dp).type==EXP)
                     printf("Type = EXP\n");
                  else if ((*dp).type==SIN)
                     printf("Type = SIN\n");
                  else if ((*dp).type==COS)
                     printf("Type = COS\n");
                  else
                     error_root(1,1,"main [check2.c]",
                                "Unknown transformation type");

                  printf("n,b,c,nfc = %d,%d,%d,%d\n",n,b,c,nfc);
                  printf("Maximal relative deviation (fft)         = %.1e\n",
                         d);
               }

               error_loc(inv_fft(dp,nfc,ft,ftt)!=0,1,"main [check2.c]",
                         "inv_fft() failed");

               d=cmp_fn(dp,nfc);
               if (d>dmx)
                  dmx=d;

               if (my_rank==0)
               {
                  printf("Maximal relative deviation (fft+inv_fft) = %.1e\n\n",
                         d);
               }
            }
         }
      }
   }

   if (my_rank==0)
   {
      printf("Maximal deviation (all tests) = %.1e\n\n",dmx);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
