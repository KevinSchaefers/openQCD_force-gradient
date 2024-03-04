
/*******************************************************************************
*
* File blk_geometry.c
*
* Copyright (C) 2005, 2008, 2011-2017, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs related to the block geometry.
*
*   void blk_geometry(block_t *b)
*     Computes the index arrays b.ipt,b.iup and b.idn that describe the
*     geometry of the block b.
*
*   void blk_imbed(block_t *b)
*     Computes the index arrays b.imb and b.ibp that describe the
*     embedding of the block b in the full lattice.
*
*   void bnd_geometry(block_t *b)
*     Computes the index arrays bb.ipp and bb.map that describe the
*     geometry of the exterior boundaries bb of the block b.
*
*   void bnd_imbed(block_t *b)
*     Computes the index arrays bb.imb that describe the embedding
*     of the exterior boundaries bb of the block b in the full lattice.
*
* See main/README.global for a description of the lattice geometry and
* block/README.block for explanations of the block structure.
*
* The programs in this module are thread-safe and can be locally called.
* They assume that the global geometry arrays have been initialized by
* geometry() and that the relevant arrays in the block structure are
* allocated.
*
* The dependencies are:
*
* blk_imbed():          blk_geometry()
* bnd_geometry():       blk_geometry()
* bnd_imbed():          blk_geometry(),blk_imbed(),bnd_geometry()
*
*******************************************************************************/

#define BLK_GEOMETRY_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "global.h"


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


static void update_ipt0(block_t *b,int *cbs,int *cbo,int *ofs)
{
   int x0,x1,x2,x3,ix,ieo;
   int y0,y1,y2,y3,*bs;

   bs=(*b).bs;

   for (x0=0;x0<cbs[0];x0++)
   {
      for (x1=0;x1<cbs[1];x1++)
      {
         for (x2=0;x2<cbs[2];x2++)
         {
            for (x3=0;x3<cbs[3];x3++)
            {
               y0=x0+cbo[0];
               y1=x1+cbo[1];
               y2=x2+cbo[2];
               y3=x3+cbo[3];

               ix=y3+y2*bs[3]+y1*bs[2]*bs[3]+y0*bs[1]*bs[2]*bs[3];
               ieo=((y0+y1+y2+y3)&0x1);
               (*b).ipt[ix]=ofs[ieo];
               ofs[ieo]+=1;
            }
         }
      }
   }
}


static void set_blk_ipt(block_t *b)
{
   int i0,i1,i2,i3,*bs;
   int cbs[4],cbo[4],nbs[4],ofs[2];

   ofs[0]=0;
   ofs[1]=(*b).vol/2;

   bs=(*b).bs;
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
               cbo[0]=i0*cbs[0];
               cbo[1]=i1*cbs[1];
               cbo[2]=i2*cbs[2];
               cbo[3]=i3*cbs[3];

               update_ipt0(b,cbs,cbo,ofs);
            }
         }
      }
   }

   (*b).ipt[(*b).vol]=(*b).ipt[0];
}


static int index(block_t *b,int x0,int x1,int x2,int x3)
{
   int *bs;

   bs=(*b).bs;

   if ((x0<0)||(x0>=bs[0])||(x1<0)||(x1>=bs[1])||
       (x2<0)||(x2>=bs[2])||(x3<0)||(x3>=bs[3]))
      return (*b).vol;
   else
      return (*b).ipt[x3+x2*bs[3]+x1*bs[2]*bs[3]+x0*bs[1]*bs[2]*bs[3]];
}


static void set_blk_iupdn(block_t *b)
{
   int *bs;
   int x0,x1,x2,x3,ix;

   bs=(*b).bs;

   for (x0=0;x0<bs[0];x0++)
   {
      for (x1=0;x1<bs[1];x1++)
      {
         for (x2=0;x2<bs[2];x2++)
         {
            for (x3=0;x3<bs[3];x3++)
            {
               ix=index(b,x0,x1,x2,x3);

               (*b).iup[ix][0]=index(b,x0+1,x1,x2,x3);
               (*b).idn[ix][0]=index(b,x0-1,x1,x2,x3);
               (*b).iup[ix][1]=index(b,x0,x1+1,x2,x3);
               (*b).idn[ix][1]=index(b,x0,x1-1,x2,x3);
               (*b).iup[ix][2]=index(b,x0,x1,x2+1,x3);
               (*b).idn[ix][2]=index(b,x0,x1,x2-1,x3);
               (*b).iup[ix][3]=index(b,x0,x1,x2,x3+1);
               (*b).idn[ix][3]=index(b,x0,x1,x2,x3-1);
            }
         }
      }
   }
}


void blk_geometry(block_t *b)
{
   set_blk_ipt(b);
   set_blk_iupdn(b);
}


void blk_imbed(block_t *b)
{
   int *bo,*bs;
   int x0,x1,x2,x3;
   int ix,iy,ibd,ibu,*ibp;

   bo=(*b).bo;
   bs=(*b).bs;

   for (x0=0;x0<bs[0];x0++)
   {
      for (x1=0;x1<bs[1];x1++)
      {
         for (x2=0;x2<bs[2];x2++)
         {
            for (x3=0;x3<bs[3];x3++)
            {
               iy=x3+x2*bs[3]+x1*bs[2]*bs[3]+x0*bs[1]*bs[2]*bs[3];
               ix=(*b).ipt[iy];

               iy=(bo[3]+x3)+(bo[2]+x2)*L3+(bo[1]+x1)*L2*L3+(bo[0]+x0)*L1*L2*L3;
               (*b).imb[ix]=ipt[iy];
            }
         }
      }
   }

   (*b).imb[(*b).vol]=(*b).imb[0];

   ibd=((cpr[0]==0)&&((*b).bo[0]==0)&&(bc_type()!=3));
   ibu=((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0)&&(bc_type()==0));
   ibp=(*b).ibp;

   for (ix=0;ix<(*b).vol;ix++)
   {
      if (((ibd)&&((*b).idn[ix][0]==(*b).vol))||
          ((ibu)&&((*b).iup[ix][0]==(*b).vol)))
      {
         (*ibp)=ix;
         ibp+=1;
      }
   }
}


void bnd_geometry(block_t *b)
{
   int ifc,mu,ix,iy,iw,iz;
   int vol,volh,*ipp[8],*mp[8];
   bndry_t *bb;

   vol=(*b).vol;
   volh=vol/2;
   bb=(*b).bb;

   for (ifc=0;ifc<8;ifc++)
   {
      ipp[ifc]=bb[ifc].ipp;
      mp[ifc]=bb[ifc].map;
   }

   for (ix=0;ix<vol;ix++)
   {
      if (ix<volh)
         iy=ix+volh;
      else
         iy=ix-volh;

      for (mu=0;mu<4;mu++)
      {
         if ((*b).iup[iy][mu]==vol)
         {
            ifc=2*mu+1;
            ipp[ifc][0]=iy;
            ipp[ifc]+=1;

            iw=iy;
            iz=iy;

            while (iw<vol)
            {
               iz=iw;
               iw=(*b).idn[iw][mu];
            }

            mp[ifc][0]=iz;
            mp[ifc]+=1;
         }

         if ((*b).idn[iy][mu]==vol)
         {
            ifc=2*mu;
            ipp[ifc][0]=iy;
            ipp[ifc]+=1;

            iw=iy;
            iz=iy;

            while (iw<vol)
            {
               iz=iw;
               iw=(*b).iup[iw][mu];
            }

            mp[ifc][0]=iz;
            mp[ifc]+=1;
         }
      }
   }

   for (ifc=0;ifc<8;ifc++)
   {
      vol=(*bb).vol;
      (*bb).ipp[vol]=(*bb).ipp[0];
      (*bb).map[vol]=(*bb).map[0];
      bb+=1;
   }
}


void bnd_imbed(block_t *b)
{
   int ifc,ix,iy;
   int vol,*ipp,*imb;
   bndry_t *bb;

   bb=(*b).bb;
   imb=(*b).imb;

   for (ifc=0;ifc<8;ifc++)
   {
      vol=(*bb).vol;
      ipp=(*bb).ipp;

      for (ix=0;ix<vol;ix++)
      {
         iy=imb[ipp[ix]];

         if (ifc&0x1)
            (*bb).imb[ix]=iup[iy][ifc/2];
         else
            (*bb).imb[ix]=idn[iy][ifc/2];
      }

      (*bb).imb[vol]=(*bb).imb[0];

      if ((*bb).imb[0]>=VOLUME)
         (*bb).ibn=1;
      else
         (*bb).ibn=0;

      bb+=1;
   }
}
