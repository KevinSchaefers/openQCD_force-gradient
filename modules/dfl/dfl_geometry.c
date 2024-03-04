
/*******************************************************************************
*
* File dfl_geometry.c
*
* Copyright (C) 2007, 2011, 2013, 2017, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Geometry of the DFL_BLOCKS block grid.
*
*   void set_dfl_geometry(void)
*     Allocates and initializes the geometry arrays describing the
*     DFL_BLOCKS block grid.
*
*   dfl_grid_t *dfl_geometry(void)
*     Returns the pointer to a structure containing the index arrays that
*     describe the geometry of the DFL_BLOCKS block grid.
*
* The blocks in the DFL_BLOCKS grid form a hypercubic lattice whose geometry
* is described by a structure of type dfl_grid_t. The elements of this
* structure are:
*
*  nb              Number of blocks in the local lattice.
*
*  nbb             Number of exterior boundary blocks of the local
*                  block lattice.
*
*  inn[ix][ifc]    Index of the nearest neighbour block in direction ifc
*                  of the block with index ix (ix=0,..,nb-1, ifc=0,..,7).
*                  The ordering of the directions ifc is -0,+0,..,-3,+3.
*
*  idx[ix]         Position of the block with index ix in the array of
*                  blocks returned by blk_list(). Note that ix=idx[ib]
*                  if ib=idx[ix].
*
*  ipp[ix]         Index of the nearest neighbour (partner block) in the
*                  local lattice of the block on the exterior boundary
*                  with index nb+ix (ix=0,..,nbb-1).
*
*  nbbe[ifc]       Number of even (odd) blocks on the exterior boundary
*  nbbo[ifc]       in direction ifc.
*
*  obbe[ifc]       Offset of the index of the first even (odd) block on
*  obbo[ifc]       the exterior boundary in direction ifc. The offsets
*                  are given relative to the first block on the boundary.
*
* The program set_dfl_geometry() obtains the block size from the parameter
* data base (see flags/dfl_parms.c).
*
* The blocks in the local lattice are ordered according to their Cartesian
* coordinates (n0,n1,n2,n3) in the global block lattice. First come all even
* blocks (those with (n0+n1+n2+n3)=0 mod 2) and then the odd ones. Within
* each of these two groups of blocks, the ordering is lexicographic, i.e.
* the block with coordinates n comes before the block with coordinates m if
*
*   (n0<m0) or ((n0=m0)&&(n1<m1)) or ...
*
* The blocks on the exterior boundaries of the local lattice are ordered in
* the same way and thus exactly like the blocks on the corresponding interior
* boundary of the neighbouring local lattices.
*
* This ordering coincides with the one of the blocks in the array returned
* by blk_list(DFL_BLOCKS,&nb,&isw) if isw=0, while if isw=1 the even and odd
* blocks are swapped.
*
* The program set_dfl_geometry() is assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously. When the deflation block grid
* is allocated, the geometry arrays are automatically set up and there is
* then no need to call set_dfl_geometry() from anywhere else.
*
* The program dfl_geometry() is thread-safe, but assumes set_dfl_geometry()
* has previously been called.
*
*******************************************************************************/

#define DFL_GEOMETRY_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "flags.h"
#include "utils.h"
#include "dfl.h"
#include "global.h"

static dfl_grid_t *dfl_grid=NULL;


static void set_grid_sizes(int *isw,int *nbl)
{
   int mu,ifc,nbb[4];
   int *bs,*nbbe,*nbbo,*obbe,*obbo;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   bs=dfl.bs;

   error_root(dfl.Ns==0,1,"set_grid_sizes [dfl_geometry.c]",
              "Deflation subspace parameters are not set");

   nbl[0]=L0/bs[0];
   nbl[1]=L1/bs[1];
   nbl[2]=L2/bs[2];
   nbl[3]=L3/bs[3];

   nbb[0]=(NPROC0>1)*nbl[1]*nbl[2]*nbl[3];
   nbb[1]=(NPROC1>1)*nbl[2]*nbl[3]*nbl[0];
   nbb[2]=(NPROC2>1)*nbl[3]*nbl[0]*nbl[1];
   nbb[3]=(NPROC3>1)*nbl[0]*nbl[1]*nbl[2];

   (*isw)=(nbl[0]*cpr[0]+nbl[1]*cpr[1]+
        nbl[2]*cpr[2]+nbl[3]*cpr[3])&0x1;

   (*dfl_grid).nb=nbl[0]*nbl[1]*nbl[2]*nbl[3];
   (*dfl_grid).nbb=2*(nbb[0]+nbb[1]+nbb[2]+nbb[3]);

   nbbe=(*dfl_grid).nbbe;
   nbbo=(*dfl_grid).nbbo;
   obbe=(*dfl_grid).obbe;
   obbo=(*dfl_grid).obbo;

   for (mu=0;mu<4;mu++)
   {
      if (*isw)
      {
         nbbe[2*mu]=(nbb[mu]+1)/2;
         nbbo[2*mu]=nbb[mu]-nbbe[2*mu];
      }
      else
      {
         nbbo[2*mu]=(nbb[mu]+1)/2;
         nbbe[2*mu]=nbb[mu]-nbbo[2*mu];
      }

      if (nbl[mu]&0x1)
      {
         nbbe[2*mu+1]=nbbe[2*mu];
         nbbo[2*mu+1]=nbbo[2*mu];
      }
      else
      {
         nbbe[2*mu+1]=nbbo[2*mu];
         nbbo[2*mu+1]=nbbe[2*mu];
      }
   }

   obbe[0]=0;

   for (ifc=1;ifc<8;ifc++)
      obbe[ifc]=obbe[ifc-1]+nbbe[ifc-1];

   obbo[0]=obbe[7]+nbbe[7];

   for (ifc=1;ifc<8;ifc++)
      obbo[ifc]=obbo[ifc-1]+nbbo[ifc-1];
}


static void alloc_arrays(void)
{
   int nb,nbb;
   int (*inn)[8],*idx;

   nb=(*dfl_grid).nb;
   nbb=(*dfl_grid).nbb;
   inn=malloc(nb*sizeof(*inn));
   idx=malloc((nb+nbb)*sizeof(*idx));

   error((inn==NULL)||(idx==NULL),1,"alloc_arrays [dfl_geometry.c]",
         "Unable to allocate index arrays");

   (*dfl_grid).inn=inn;
   (*dfl_grid).idx=idx;
   (*dfl_grid).ipp=idx+nb;
}


static void set_index(int isw,int *nbl)
{
   int n0,n1,n2,n3;
   int in,ic[2],*idx;

   in=0;
   ic[0]=0;
   ic[1]=(*dfl_grid).nb/2;
   idx=(*dfl_grid).idx;

   for (n0=0;n0<nbl[0];n0++)
   {
      for (n1=0;n1<nbl[1];n1++)
      {
         for (n2=0;n2<nbl[2];n2++)
         {
            for (n3=0;n3<nbl[3];n3++)
            {
               if (((n0+n1+n2+n3)&0x1)==isw)
               {
                  idx[in]=ic[0];
                  ic[0]+=1;
               }
               else
               {
                  idx[in]=ic[1];
                  ic[1]+=1;
               }

               in+=1;
            }
         }
      }
   }
}


static int index(int n0,int n1,int n2,int n3,int *nbl)
{
   int ib;

   n0=safe_mod(n0,nbl[0]);
   n1=safe_mod(n1,nbl[1]);
   n2=safe_mod(n2,nbl[2]);
   n3=safe_mod(n3,nbl[3]);

   ib=n3+nbl[3]*n2+nbl[2]*nbl[3]*n1+nbl[1]*nbl[2]*nbl[3]*n0;

   return (*dfl_grid).idx[ib];
}


static void set_inn(int *nbl)
{
   int n0,n1,n2,n3,nb,nbh;
   int in,ifc,ic[8];
   int (*inn)[8],*obbe,*obbo;

   nb=(*dfl_grid).nb;
   nbh=nb/2;
   inn=(*dfl_grid).inn;

   for (n0=0;n0<nbl[0];n0++)
   {
      for (n1=0;n1<nbl[1];n1++)
      {
         for (n2=0;n2<nbl[2];n2++)
         {
            for (n3=0;n3<nbl[3];n3++)
            {
               in=index(n0,n1,n2,n3,nbl);
               inn[in][0]=index(n0-1,n1,n2,n3,nbl);
               inn[in][1]=index(n0+1,n1,n2,n3,nbl);
               inn[in][2]=index(n0,n1-1,n2,n3,nbl);
               inn[in][3]=index(n0,n1+1,n2,n3,nbl);
               inn[in][4]=index(n0,n1,n2-1,n3,nbl);
               inn[in][5]=index(n0,n1,n2+1,n3,nbl);
               inn[in][6]=index(n0,n1,n2,n3-1,nbl);
               inn[in][7]=index(n0,n1,n2,n3+1,nbl);

               if (NPROC0>1)
               {
                  if (n0==0)
                     inn[in][0]=nb;
                  if (n0==(nbl[0]-1))
                     inn[in][1]=nb;
               }
               if (NPROC1>1)
               {
                  if (n1==0)
                     inn[in][2]=nb;
                  if (n1==(nbl[1]-1))
                     inn[in][3]=nb;
               }
               if (NPROC2>1)
               {
                  if (n2==0)
                     inn[in][4]=nb;
                  if (n2==(nbl[2]-1))
                     inn[in][5]=nb;
               }
               if (NPROC3>1)
               {
                  if (n3==0)
                     inn[in][6]=nb;
                  if (n3==(nbl[3]-1))
                     inn[in][7]=nb;
               }
            }
         }
      }
   }

   obbe=(*dfl_grid).obbe;
   obbo=(*dfl_grid).obbo;

   for (ifc=0;ifc<8;ifc++)
      ic[ifc]=0;

   for (in=0;in<nbh;in++)
   {
      for (ifc=0;ifc<8;ifc++)
      {
         if (inn[in][ifc]==nb)
         {
            inn[in][ifc]+=obbo[ifc]+ic[ifc];
            ic[ifc]+=1;
         }
      }
   }

   for (ifc=0;ifc<8;ifc++)
      ic[ifc]=0;

   for (in=nbh;in<nb;in++)
   {
      for (ifc=0;ifc<8;ifc++)
      {
         if (inn[in][ifc]==nb)
         {
            inn[in][ifc]+=obbe[ifc]+ic[ifc];
            ic[ifc]+=1;
         }
      }
   }
}


static void set_ipp(void)
{
   int nb,in,im,ifc;
   int (*inn)[8],*ipp;

   nb=(*dfl_grid).nb;
   inn=(*dfl_grid).inn;
   ipp=(*dfl_grid).ipp;

   for (in=0;in<nb;in++)
   {
      for (ifc=0;ifc<8;ifc++)
      {
         im=inn[in][ifc];

         if (im>=nb)
            ipp[im-nb]=in;
      }
   }
}


static void set_idx(int isw)
{
   int nb,nbh,ix,*idx;

   idx=(*dfl_grid).idx;
   nb=(*dfl_grid).nb;
   nbh=nb/2;

   for (ix=0;ix<nb;ix++)
   {
      if (ix<nbh)
         idx[ix]=ix+isw*nbh;
      else
         idx[ix]=ix-isw*nbh;
   }
}


void set_dfl_geometry(void)
{
   int isw,nbl[4];

   if (dfl_grid==NULL)
   {
      dfl_grid=malloc(1*sizeof(*dfl_grid));
      error(dfl_grid==NULL,1,"set_dfl_geometry [dfl_geometry.c]",
            "Unable to allocate dfl_grid structure");

      set_grid_sizes(&isw,nbl);
      alloc_arrays();
      set_index(isw,nbl);
      set_inn(nbl);
      set_ipp();
      set_idx(isw);
   }
}


dfl_grid_t *dfl_geometry(void)
{
   return dfl_grid;
}
