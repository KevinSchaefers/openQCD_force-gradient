
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2015 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the programs dft_gather() and dft_scatter().
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

static int nproc[4]={NPROC0,NPROC1,NPROC2,NPROC3};
static int nfs,nxs[NPROC],mfs[NPROC];
static complex_dble **lf=NULL,**lff,**f;


static void set_nxs(int mu)
{
   int i,np,ip0,nu,cp[4];
   int m,r;
   int nx[NPROC],nf[NPROC];
   float rn;

   np=nproc[mu];

   for (i=0;i<NPROC;i++)
   {
      ranlxs(&rn,1);
      nx[i]=1+(int)(18.0f*rn);
      ranlxs(&rn,1);
      nf[i]=np+(int)(224.0f*rn);
   }

   MPI_Bcast(nx,NPROC,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(nf,NPROC,MPI_INT,0,MPI_COMM_WORLD);

   for (nu=0;nu<4;nu++)
   {
      if (nu!=mu)
         cp[nu]=cpr[nu];
      else
         cp[nu]=0;
   }

   ip0=ipr_global(cp);
   nfs=nf[ip0];
   m=nfs/np;
   r=nfs%np;

   for (i=0;i<np;i++)
   {
      cp[mu]=i;
      nxs[i]=nx[ipr_global(cp)];
      mfs[i]=m+(i<r);
   }
}


static void alloc_arrays(int mu)
{
   int np,nx,nf,mf,snx,i;
   complex_dble *p;

   if (lf!=NULL)
   {
      afree(lf[0]);
      free(lf);
   }

   np=nproc[mu];
   nx=nxs[cpr[mu]];
   snx=0;

   for (i=0;i<np;i++)
      snx+=nxs[i];

   nf=nfs;
   mf=mfs[cpr[mu]];

   lf=malloc((2*nx+snx)*sizeof(*lf));
   p=amalloc((2*nx*nf+snx*mf)*sizeof(*p),4);

   error((lf==NULL)||(p==NULL),1,"alloc_arrays [check3.c]",
         "Unable to allocate data arrays");

   lff=lf+nx;
   f=lff+nx;
   gauss_dble((double*)(p),2*(2*nx*nf+snx*mf));

   for (i=0;i<nx;i++)
   {
      lf[i]=p;
      p+=nf;
   }

   for (i=0;i<nx;i++)
   {
      lff[i]=p;
      p+=nf;
   }

   for (i=0;i<snx;i++)
   {
      f[i]=p;
      p+=mf;
   }
}


static void set_lf(int mu)
{
   int i,j,nf,nx,snx;

   nf=nfs;
   nx=nxs[cpr[mu]];
   snx=0;

   for (i=0;i<cpr[mu];i++)
      snx+=nxs[i];

   for (i=0;i<nx;i++)
   {
      for (j=0;j<nf;j++)
      {
         lf[i][j].re=(double)(snx+i);
         lf[i][j].im=(double)(j);
      }
   }
}


static void chk_f(int mu)
{
   int np,mf,snx,smf;
   int i,j,ie;
   char s[NAME_SIZE];

   np=nproc[mu];
   mf=mfs[cpr[mu]];
   snx=0;
   smf=0;

   for (i=0;i<np;i++)
      snx+=nxs[i];

   for (i=0;i<cpr[mu];i++)
      smf+=mfs[i];

   ie=0;

   for (i=0;i<snx;i++)
   {
      for (j=0;j<mf;j++)
      {
         ie|=(f[i][j].re!=(double)(i));
         ie|=(f[i][j].im!=(double)(smf+j));

         if (ie!=0)
         {
            sprintf(s,"smf,i,j,f = %d,%d,%d,(%d,%d)",
                    smf,i,j,(int)(f[i][j].re),(int)(f[i][j].im));
            error_loc(1,1,"chk_f [check3.c]",s);
            return;
         }
      }
   }
}


static void chk_lff(int mu)
{
   int nf,nx,snx;
   int i,j,ie;
   char s[NAME_SIZE];

   nx=nxs[cpr[mu]];
   nf=nfs;
   snx=0;

   for (i=0;i<cpr[mu];i++)
      snx+=nxs[i];

   ie=0;

   for (i=0;i<nx;i++)
   {
      for (j=0;j<nf;j++)
      {
         ie|=(lff[i][j].re!=(double)(i+snx));
         ie|=(lff[i][j].im!=(double)(j));

         if (ie!=0)
         {
            sprintf(s,"snx,i,j,lff = %d,%d,%d,(%d,%d)",
                    snx,i,j,(int)(lff[i][j].re),(int)(lff[i][j].im));
            error_loc(1,1,"chk_lff [check3.c]",s);
            return;
         }
      }
   }
}


int main(int argc,char *argv[])
{
   int my_rank,mu;
   int nx,nf;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);

      printf("\n");
      printf("Check of the programs dft_gather() and dft_scatter()\n");
      printf("----------------------------------------------------\n\n");

      print_lattice_sizes();
      fflush(flog);
   }

   start_ranlux(0,12345);
   geometry();

   for (mu=0;mu<4;mu++)
   {
      if (my_rank==0)
         printf("mu = %d, ",mu);

      set_nxs(mu);
      nx=nxs[cpr[mu]];
      nf=nfs;

      if (my_rank==0)
         printf("nx = %d, nf = %d\n",nx,nf);

      alloc_arrays(mu);
      set_lf(mu);

      dft_gather(mu,nxs,mfs,lf,f);
      chk_f(mu);

      dft_scatter(mu,nxs,mfs,f,lff);
      chk_lff(mu);
   }

   if (my_rank==0)
   {
      printf("\n");
      printf("No errors discovered\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
