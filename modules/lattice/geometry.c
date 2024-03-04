
/*******************************************************************************
*
* File geometry.c
*
* Copyright (C) 2005-2017, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs related to the global lattice geometry.
*
*   void geometry(void)
*     Allocates and initializes all global index arrays that describe the
*     lattice geometry.
*
*   void ipt_global(int *x,int *ip,int *ix)
*     Given the Cartesian coordinates x[0],..,x[3] of a point on the full
*     lattice, this program finds the local lattice containing x. On exit
*     the rank of the associated MPI process is assigned to ip and the
*     local index of the point to ix.
*
*   int global_time(int ix)
*     Returns the (global) time coordinate of the lattice point with local
*     index ix. If ix is out of range, NPROC0*L0 is returned.
*
* See main/README.global for a description of the lattice geometry.
*
* The program geometry() is assumed to be called by the OpenMP master thread
* on all MPI processes simultaneously. All other programs are thread-safe and
* can be locally called, but assume that the index arrays have been set up by
* geometry().
*
*******************************************************************************/

#define GEOMETRY_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "utils.h"
#include "lattice.h"
#include "global.h"

static int *tms=NULL;


static void cache_block_size(int *bs,int *cbs)
{
   int mu;

   cbs[0]=bs[0];

   for (mu=1;mu<4;mu++)
   {
      if ((bs[mu]%4)==0)
         cbs[mu]=4;
      else if ((bs[mu]%3)==0)
         cbs[mu]=3;
      else
         cbs[mu]=2;
   }
}


static void subblock_sizes(int (*sbs)[4])
{
   sbs[0][0]=L0_TRD/2+((L0_TRD/2)%2);
   sbs[1][0]=L0_TRD-sbs[0][0];
   sbs[0][1]=L1_TRD/2+((L1_TRD/2)%2);
   sbs[1][1]=L1_TRD-sbs[0][1];
   sbs[0][2]=L2_TRD/2+((L2_TRD/2)%2);
   sbs[1][2]=L2_TRD-sbs[0][2];
   sbs[0][3]=L3_TRD/2+((L3_TRD/2)%2);
   sbs[1][3]=L3_TRD-sbs[0][3];
}


static void update_ipt0(int *bs,int *bo,int *ofs)
{
   int x0,x1,x2,x3,ix,ieo;
   int y0,y1,y2,y3;

   for (x0=0;x0<bs[0];x0++)
   {
      for (x1=0;x1<bs[1];x1++)
      {
         for (x2=0;x2<bs[2];x2++)
         {
            for (x3=0;x3<bs[3];x3++)
            {
               y0=x0+bo[0];
               y1=x1+bo[1];
               y2=x2+bo[2];
               y3=x3+bo[3];

               ix=y3+y2*L3+y1*L2*L3+y0*L1*L2*L3;
               ieo=((y0+y1+y2+y3)&0x1);
               ipt[ix]=ofs[ieo];
               ofs[ieo]+=1;
            }
         }
      }
   }
}


static void update_ipt1(int *bs,int *bo,int *ofs)
{
   int i0,i1,i2,i3;
   int cbs[4],cbo[4],nbs[4];

   cache_block_size(bs,cbs);

   nbs[0]=bs[0]/cbs[0];
   nbs[1]=bs[1]/cbs[1];
   nbs[2]=bs[2]/cbs[2];
   nbs[3]=bs[3]/cbs[3];

   for (i0=0;i0<nbs[0];i0++)
   {
      for (i1=0;i1<nbs[1];i1++)
      {
         for (i2=0;i2<nbs[2];i2++)
         {
            for (i3=0;i3<nbs[3];i3++)
            {
               cbo[0]=bo[0]+i0*cbs[0];
               cbo[1]=bo[1]+i1*cbs[1];
               cbo[2]=bo[2]+i2*cbs[2];
               cbo[3]=bo[3]+i3*cbs[3];

               update_ipt0(cbs,cbo,ofs);
            }
         }
      }
   }
}


static void update_ipt2(int *ofs)
{
   int i0,i1,i2,i3;
   int sbs[2][4],bs[4],bo[4];

   subblock_sizes(sbs);

   for (i0=0;i0<2;i0++)
   {
      for (i1=0;i1<2;i1++)
      {
         for (i2=0;i2<2;i2++)
         {
            for (i3=0;i3<2;i3++)
            {
               bs[0]=sbs[i0][0];
               bs[1]=sbs[i1][1];
               bs[2]=sbs[i2][2];
               bs[3]=sbs[i3][3];

               bo[0]=i0*sbs[0][0];
               bo[1]=i1*sbs[0][1];
               bo[2]=i2*sbs[0][2];
               bo[3]=i3*sbs[0][3];

               update_ipt1(bs,bo,ofs);
            }
         }
      }
   }
}


static void alloc_ipt(void)
{
   ipt=malloc(VOLUME*sizeof(*ipt));

   error(ipt==NULL,1,"alloc_ipt [geometry.c]",
         "Unable to allocate index array");
}


static void set_ipt(void)
{
   int k,n,n0,n1,n2,n3,is,ix;
   int nt1,nt2,nt3,ofs[2];

   alloc_ipt();

   ofs[0]=0;
   ofs[1]=(VOLUME/2);
   update_ipt2(ofs);

   nt1=(L1/L1_TRD);
   nt2=(L2/L2_TRD);
   nt3=(L3/L3_TRD);

#pragma omp parallel private(k,n,n0,n1,n2,n3,is,ix)
   {
      k=omp_get_thread_num();

      if (k>0)
      {
         n=k;
         n3=n%nt3;
         n/=nt3;
         n2=n%nt2;
         n/=nt2;
         n1=n%nt1;
         n/=nt1;
         n0=n;

         n0*=L0_TRD;
         n1*=L1_TRD;
         n2*=L2_TRD;
         n3*=L3_TRD;

         is=n3+n2*L3+n1*L2*L3+n0*L1*L2*L3;

         for (n0=0;n0<L0_TRD;n0++)
         {
            for (n1=0;n1<L1_TRD;n1++)
            {
               for (n2=0;n2<L2_TRD;n2++)
               {
                  for (n3=0;n3<L3_TRD;n3++)
                  {
                     ix=n3+n2*L3+n1*L2*L3+n0*L1*L2*L3;
                     ipt[ix+is]=ipt[ix]+k*(VOLUME_TRD/2);
                  }
               }
            }
         }
      }
   }
}


static void alloc_tms(void)
{
   tms=malloc(VOLUME*sizeof(*tms));

   error(tms==NULL,1,"alloc_tms [geometry.c]",
         "Unable to allocate time array");
}


static void set_tms(void)
{
   int k,ix,iy,x0;

   alloc_tms();

#pragma omp parallel private(k,ix,iy,x0)
   {
      k=omp_get_thread_num();

      for (iy=(k*VOLUME_TRD);iy<((k+1)*VOLUME_TRD);iy++)
      {
         x0=iy/(L1*L2*L3);
         ix=ipt[iy];

         tms[ix]=x0+cpr[0]*L0;
      }
   }
}


void geometry(void)
{
   if (ipt==NULL)
   {
      set_cpr();
      set_sbofs();
      set_ipt();
      set_iupdn();
      set_map();
      set_tms();
   }
}


void ipt_global(int *x,int *ip,int *ix)
{
   int x0,x1,x2,x3;
   int n[4];

   x0=safe_mod(x[0],NPROC0*L0);
   x1=safe_mod(x[1],NPROC1*L1);
   x2=safe_mod(x[2],NPROC2*L2);
   x3=safe_mod(x[3],NPROC3*L3);

   n[0]=x0/L0;
   n[1]=x1/L1;
   n[2]=x2/L2;
   n[3]=x3/L3;

   (*ip)=ipr_global(n);

   x0=x0%L0;
   x1=x1%L1;
   x2=x2%L2;
   x3=x3%L3;

   (*ix)=ipt[x3+x2*L3+x1*L2*L3+x0*L1*L2*L3];
}


int global_time(int ix)
{
   if ((ix>=0)&&(ix<VOLUME))
      return tms[ix];
   else
      return NPROC0*L0;
}
