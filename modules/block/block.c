
/*******************************************************************************
*
* File block.c
*
* Copyright (C) 2005, 2011, 2013, 2017, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Basic allocation programs for blocks of lattice points.
*
*   int alloc_blk(block_t *b,int *bo,int *bs,
*                 int iu,int iud,int ns,int nsd)
*     Sets the offset and side-lengths of the block b to bo[4] and bs[4],
*     respectively, and allocates the block fields depending on the values
*     of the other parameters. The single-precision gauge and SW fields are
*     allocated if iu=1, the double-precision gauge and SW fields if iud=1,
*     while ns and nsd are the numbers of single- and double-precision Dirac
*     fields that are allocated. All elements of the block are properly
*     initialized and the share flag b.shf is set to 0x0.
*
*   int alloc_bnd(block_t *b,int iu,int iud,int nw,int nwd)
*     Allocates the boundary structures b.bb in the block b and the fields
*     in there depending on the parameters iu,iud,nw and nwd. The single-
*     and double-precision gauge fields are allocated if iu=1 and iud=1,
*     respectively, while nw and nwd are the numbers of single- and double-
*     precision Weyl fields that are allocated. All elements of the block
*     are then properly initialized (see the notes).
*
*   int clone_blk(block_t *b,int shf,int *bo,block_t *c)
*     Sets the offset of the block c to bo[4] and its side lengths to
*     b.bs[4]. The fields in c are then allocated depending on the bits
*     b1,b2,..,b8 (counting from the lowest) of the share flag shf. The
*     relevant bits are:
*
*       b2=1: b.ipt,b.iup and b.idn are shared,
*       b3=1: b.u, b.bb.u and b.sw are shared,
*       b4=1: b.ud, b.bb.ud and b.swd are shared,
*       b5=1: b.s is shared,
*       b6=1: b.sd is shared.
*       b7=1: b.bb.w is shared,
*       b8=1: b.bb.wd is shared.
*
*     Fields allocated on b are allocated on c too if the associated share
*     flag is not set. If the flag is set, the address of the field on c is
*     set to the one on b. On exit the share flag c.shf is set to shf.
*
*   int free_blk(block_t *b)
*     Frees the non-shared arrays in the block b and the boundaries b.bb.
*     The boundary structures are then freed too (if they were allocated)
*     and all entries in the block structure are set to 0 (or NULL). No
*     operation is performed and 1 is returned if the block is protected,
*     i.e. if the lowest bit of the share flag b.shf is equal to 1.
*
* The entries of the block and boundary structures are explained in the file
* README.block in this directory.
*
* It is currently not possible to allocate blocks that are not fully
* contained in the local lattice. Moreover, the block sizes must be even
* and not smaller than 4. The exterior boundaries of a block may, however,
* overlap with the lattices on the neighbouring MPI processes.
*
* All programs in this module are thread-safe. A non-zero value is returned
* if a program fails.
*
*******************************************************************************/

#define BLOCK_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "sflds.h"
#include "block.h"
#include "global.h"

static const su3 u0={{0.0f}};
static const su3_dble ud0={{0.0}};
static const pauli p0={{0.0f}};
static const pauli_dble pd0={{0.0}};
static const weyl w0={{{0.0f}}};
static const weyl_dble wd0={{{0.0}}};


static void set_u2unity(int vol,su3 *u)
{
   su3 unity,*um;

   unity=u0;
   unity.c11.re=1.0f;
   unity.c22.re=1.0f;
   unity.c33.re=1.0f;

   um=u+vol;

   for (;u<um;u++)
      (*u)=unity;
}


static void set_ud2unity(int vol,su3_dble *ud)
{
   su3_dble unity,*um;

   unity=ud0;
   unity.c11.re=1.0;
   unity.c22.re=1.0;
   unity.c33.re=1.0;

   um=ud+vol;

   for (;ud<um;ud++)
      (*ud)=unity;
}


static void set_sw2unity(int vol,pauli *p)
{
   pauli unity,*pm;

   unity=p0;
   unity.u[0]=1.0f;
   unity.u[1]=1.0f;
   unity.u[2]=1.0f;
   unity.u[3]=1.0f;
   unity.u[4]=1.0f;
   unity.u[5]=1.0f;

   pm=p+vol;

   for (;p<pm;p++)
      (*p)=unity;
}


static void set_swd2unity(int vol,pauli_dble *pd)
{
   pauli_dble unity,*pm;

   unity=pd0;
   unity.u[0]=1.0;
   unity.u[1]=1.0;
   unity.u[2]=1.0;
   unity.u[3]=1.0;
   unity.u[4]=1.0;
   unity.u[5]=1.0;

   pm=pd+vol;

   for (;pd<pm;pd++)
      (*pd)=unity;
}


static void set_w2zero(int vol,weyl *w)
{
   weyl *wm;

   wm=w+vol;

   for (;w<wm;w++)
      (*w)=w0;
}


static void set_wd2zero(int vol,weyl_dble *wd)
{
   weyl_dble *wm;

   wm=wd+vol;

   for (;wd<wm;wd++)
      (*wd)=wd0;
}


static int new_blk(block_t *b,int *bo,int *bs,
                   int iu,int iud,int ns,int nsd,int shf)
{
   int n,mu,ie;

   ie=((bo[0]<0)||((bo[0]+bs[0])>L0)||(bs[0]<4)||((bs[0]%2)!=0)||
       (bo[1]<0)||((bo[1]+bs[1])>L1)||(bs[1]<4)||((bs[1]%2)!=0)||
       (bo[2]<0)||((bo[2]+bs[2])>L2)||(bs[2]<4)||((bs[2]%2)!=0)||
       (bo[3]<0)||((bo[3]+bs[3])>L3)||(bs[3]<4)||((bs[3]%2)!=0));

   ie|=((ns<0)||(nsd<0));

   (*b).bo=malloc(8*sizeof(*(*b).bo));
   ie|=((*b).bo==NULL);

   if (ie==0)
   {
      (*b).bs=(*b).bo+4;

      for (mu=0;mu<4;mu++)
      {
         (*b).bo[mu]=bo[mu];
         (*b).bs[mu]=bs[mu];
      }

      (*b).vol=bs[0]*bs[1]*bs[2]*bs[3];
      (*b).vbb=2*(bs[0]*bs[1]*bs[2]+bs[1]*bs[2]*bs[3]+
                  bs[2]*bs[3]*bs[0]+bs[3]*bs[0]*bs[1]);
      (*b).nbp=0;

      if ((cpr[0]==0)&&(bo[0]==0)&&(bc_type()!=3))
         (*b).nbp+=bs[1]*bs[2]*bs[3];
      if ((cpr[0]==(NPROC0-1))&&((bo[0]+bs[0])==L0)&&(bc_type()==0))
         (*b).nbp+=bs[1]*bs[2]*bs[3];

      (*b).ns=ns;
      (*b).nsd=nsd;
      (*b).shf=shf;

      if (shf&0x2)
      {
         (*b).ipt=NULL;
         (*b).iup=NULL;
         (*b).idn=NULL;
      }
      else
      {
         (*b).ipt=malloc(((*b).vol+1)*sizeof(*(*b).ipt));
         (*b).iup=malloc(2*(*b).vol*sizeof(*(*b).iup));
         ie|=(((*b).ipt==NULL)||((*b).iup==NULL));

         (*b).idn=(*b).iup+(*b).vol;
      }

      (*b).imb=malloc((((*b).vol+1)+(*b).nbp)*sizeof(*(*b).imb));
      ie|=((*b).imb==NULL);

      (*b).ibp=(*b).imb+(*b).vol+1;

      if ((shf&0x4)||(iu!=1))
      {
         (*b).u=NULL;
         (*b).sw=NULL;
      }
      else
      {
         (*b).u=amalloc(4*(*b).vol*sizeof(*(*b).u),ALIGN);
         (*b).sw=amalloc(2*(*b).vol*sizeof(*(*b).sw),ALIGN);
         ie|=(((*b).u==NULL)||((*b).sw==NULL));

         if (ie==0)
         {
            set_u2unity(4*(*b).vol,(*b).u);
            set_sw2unity(2*(*b).vol,(*b).sw);
         }
      }

      if ((shf&0x8)||(iud!=1))
      {
         (*b).ud=NULL;
         (*b).swd=NULL;
      }
      else
      {
         (*b).ud=amalloc(4*(*b).vol*sizeof(*(*b).ud),ALIGN);
         (*b).swd=amalloc(2*(*b).vol*sizeof(*(*b).swd),ALIGN);
         ie|=(((*b).ud==NULL)||((*b).swd==NULL));

         if (ie==0)
         {
            set_ud2unity(4*(*b).vol,(*b).ud);
            set_swd2unity(2*(*b).vol,(*b).swd);
         }
      }

      if ((shf&0x10)||(ns==0))
         (*b).s=NULL;
      else
      {
         (*b).s=malloc(ns*sizeof(*(*b).s));
         ie|=((*b).s==NULL);

         if (ie==0)
         {
            (*b).s[0]=amalloc(ns*((*b).vol+1)*sizeof(*((*b).s[0])),ALIGN);
            ie|=((*b).s[0]==NULL);
         }

         if (ie==0)
         {
            for (n=1;n<ns;n++)
               (*b).s[n]=(*b).s[n-1]+(*b).vol+1;

            set_s2zero(ns*((*b).vol+1),0,(*b).s[0]);
         }
      }

      if ((shf&0x20)||(nsd==0))
         (*b).sd=NULL;
      else
      {
         (*b).sd=malloc(nsd*sizeof(*(*b).sd));
         ie|=((*b).sd==NULL);

         if (ie==0)
         {
            (*b).sd[0]=amalloc(nsd*((*b).vol+1)*sizeof(*((*b).sd[0])),ALIGN);
            ie|=((*b).sd[0]==NULL);
         }

         if (ie==0)
         {
            for (n=1;n<nsd;n++)
               (*b).sd[n]=(*b).sd[n-1]+(*b).vol+1;

            set_sd2zero(nsd*((*b).vol+1),0,(*b).sd[0]);
         }
      }

      (*b).bb=NULL;
   }

   return ie;
}


int alloc_blk(block_t *b,int *bo,int *bs,
              int iu,int iud,int ns,int nsd)
{
   int ie;

   ie=((ipt==NULL)||(new_blk(b,bo,bs,iu,iud,ns,nsd,0x0)!=0));

   if (ie==0)
   {
      blk_geometry(b);
      blk_imbed(b);
   }

   return ie;
}


static int new_bnd(block_t *b,int iu,int iud,int nw,int nwd,int shf)
{
   int vol,ifc,n,ie;
   int *bs,*ipp,*imb;
   su3 *u;
   su3_dble *ud;
   weyl **w,*wb;
   weyl_dble **wd,*wdb;
   bndry_t *bb;

   ie=((nw<0)||(nwd<0));

   bb=malloc(8*sizeof(*bb));
   (*b).bb=bb;
   ie|=(bb==NULL);

   vol=(*b).vol;
   bs=(*b).bs;

   if (ie==0)
   {
      for (ifc=0;ifc<8;ifc++)
      {
         bb[ifc].ifc=ifc;
         bb[ifc].vol=vol/bs[ifc/2];
         bb[ifc].nw=nw;
         bb[ifc].nwd=nwd;
      }

      vol=(*b).vbb;

      if (shf&0x2)
      {
         for (ifc=0;ifc<8;ifc++)
         {
            bb[ifc].ipp=NULL;
            bb[ifc].map=NULL;
         }
      }
      else
      {
         ipp=malloc(2*(vol+8)*sizeof(*ipp));
         ie|=(ipp==NULL);

         if (ie==0)
         {
            for (ifc=0;ifc<8;ifc++)
            {
               bb[ifc].ipp=ipp;
               ipp+=(bb[ifc].vol+1);
            }

            for (ifc=0;ifc<8;ifc++)
            {
               bb[ifc].map=ipp;
               ipp+=(bb[ifc].vol+1);
            }
         }
      }
   }

   imb=malloc((vol+8)*sizeof(*imb));
   ie=(imb==NULL);

   if (ie==0)
   {
      for (ifc=0;ifc<8;ifc++)
      {
         bb[ifc].imb=imb;
         imb+=(bb[ifc].vol+1);
      }

      if ((shf&0x4)||(iu!=1))
      {
         for (ifc=0;ifc<8;ifc++)
            bb[ifc].u=NULL;
      }
      else
      {
         u=amalloc(vol*sizeof(*u),ALIGN);
         ie|=(u==NULL);

         if (ie==0)
         {
            set_u2unity(vol,u);

            for (ifc=0;ifc<8;ifc++)
            {
               bb[ifc].u=u;
               u+=bb[ifc].vol;
            }
         }
      }

      if ((shf&0x8)||(iud!=1))
      {
         for (ifc=0;ifc<8;ifc++)
            bb[ifc].ud=NULL;
      }
      else
      {
         ud=amalloc(vol*sizeof(*ud),ALIGN);
         ie|=(ud==NULL);

         if (ie==0)
         {
            set_ud2unity(vol,ud);

            for (ifc=0;ifc<8;ifc++)
            {
               bb[ifc].ud=ud;
               ud+=bb[ifc].vol;
            }
         }
      }

      if ((shf&0x40)||(nw==0))
      {
         for (ifc=0;ifc<8;ifc++)
            bb[ifc].w=NULL;
      }
      else
      {
         w=malloc(8*nw*sizeof(*w));
         wb=amalloc(nw*vol*sizeof(*wb),ALIGN);
         ie|=((w==NULL)||(wb==NULL));

         if (ie==0)
         {
            set_w2zero(nw*vol,wb);

            for (ifc=0;ifc<8;ifc++)
            {
               bb[ifc].w=w;

               for (n=0;n<nw;n++)
               {
                  (*w)=wb+n*vol;
                  w+=1;
               }

               wb+=bb[ifc].vol;
            }
         }
      }

      if ((shf&0x80)||(nwd==0))
      {
         for (ifc=0;ifc<8;ifc++)
            bb[ifc].wd=NULL;
      }
      else
      {
         wd=malloc(8*nwd*sizeof(*wd));
         wdb=amalloc(nwd*vol*sizeof(*wdb),ALIGN);
         ie|=((wd==NULL)||(wdb==NULL));

         if (ie==0)
         {
            set_wd2zero(nwd*vol,wdb);

            for (ifc=0;ifc<8;ifc++)
            {
               bb[ifc].wd=wd;

               for (n=0;n<nwd;n++)
               {
                  (*wd)=wdb+n*vol;
                  wd+=1;
               }

               wdb+=bb[ifc].vol;
            }
         }
      }
   }

   return ie;
}


int alloc_bnd(block_t *b,int iu,int iud,int nw,int nwd)
{
   int ie;

   ie=(((*b).shf&0x1)||(new_bnd(b,iu,iud,nw,nwd,0x0)!=0));

   if (ie==0)
   {
      bnd_geometry(b);
      bnd_imbed(b);
   }

   return ie;
}


int clone_blk(block_t *b,int shf,int *bo,block_t *c)
{
   int ie,*bbo,*bs;
   int iu,iud,ns,nsd,iub,iudb,nw,nwd;
   int ib,ifc;

   bbo=(*b).bo;
   bs=(*b).bs;
   iu=((*b).u!=NULL);
   iud=((*b).ud!=NULL);
   ns=(*b).ns;
   nsd=(*b).nsd;

   if ((*b).bb!=NULL)
   {
      iub=((*b).bb[0].u!=NULL);
      iudb=((*b).bb[0].ud!=NULL);
      nw=(*b).bb[0].nw;
      nwd=(*b).bb[0].nwd;
      ib=1;
   }
   else
   {
      iub=0;
      iudb=0;
      nw=0;
      nwd=0;
      ib=0;
   }

   ie=((bo[0]<0)||((bo[0]+bs[0])>L0)||((abs(bo[0]-bbo[0])%bs[0])!=0)||
       (bo[1]<0)||((bo[1]+bs[1])>L1)||((abs(bo[1]-bbo[1])%bs[1])!=0)||
       (bo[2]<0)||((bo[2]+bs[2])>L2)||((abs(bo[2]-bbo[2])%bs[2])!=0)||
       (bo[3]<0)||((bo[3]+bs[3])>L3)||((abs(bo[3]-bbo[3])%bs[3])!=0));

   ie=((ie!=0)||(new_blk(c,bo,bs,iu,iud,ns,nsd,shf)!=0));

   if (ie==0)
   {
      if (shf&0x2)
      {
         (*c).ipt=(*b).ipt;
         (*c).iup=(*b).iup;
         (*c).idn=(*b).idn;
      }

      if (shf&0x4)
      {
         (*c).u=(*b).u;
         (*c).sw=(*b).sw;
      }

      if (shf&0x8)
      {
         (*c).ud=(*b).ud;
         (*c).swd=(*b).swd;
      }

      if (shf&0x10)
         (*c).s=(*b).s;

      if (shf&0x20)
         (*c).sd=(*b).sd;

      if ((shf&0x2)==0x0)
         blk_geometry(c);
      blk_imbed(c);

      if (ib)
      {
         ie|=new_bnd(c,iub,iudb,nw,nwd,shf);

         if (ie==0)
         {
            for (ifc=0;ifc<8;ifc++)
            {
               if (shf&0x2)
               {
                  (*c).bb[ifc].ipp=(*b).bb[ifc].ipp;
                  (*c).bb[ifc].map=(*b).bb[ifc].map;
               }

               if (shf&0x4)
                  (*c).bb[ifc].u=(*b).bb[ifc].u;

               if (shf&0x8)
                  (*c).bb[ifc].ud=(*b).bb[ifc].ud;

               if (shf&0x40)
                  (*c).bb[ifc].w=(*b).bb[ifc].w;

               if (shf&0x80)
                  (*c).bb[ifc].wd=(*b).bb[ifc].wd;
            }

            if ((shf&0x2)==0x0)
               bnd_geometry(c);
            bnd_imbed(c);
         }
      }
   }

   return ie;
}


static void free_bnd(block_t *b)
{
   int shf;
   bndry_t *bb;

   bb=(*b).bb;

   if (bb!=NULL)
   {
      shf=(*b).shf;

      if (!(shf&0x2))
         free((*bb).ipp);

      free((*bb).imb);

      if ((!(shf&0x4))&&((*bb).u!=NULL))
         afree((*bb).u);

      if ((!(shf&0x8))&&((*bb).ud!=NULL))
         afree((*bb).ud);

      if ((!(shf&0x40))&&((*bb).nw>0))
      {
         afree((*bb).w[0]);
         free((*bb).w);
      }

      if ((!(shf&0x80))&&((*bb).nwd>0))
      {
         afree((*bb).wd[0]);
         free((*bb).wd);
      }

      free(bb);
      (*b).bb=NULL;
   }
}


int free_blk(block_t *b)
{
   int shf;

   shf=(*b).shf;

   if ((shf&0x1)==0x0)
   {
      free_bnd(b);

      free((*b).bo);
      (*b).bo=NULL;
      (*b).bs=NULL;

      if (!(shf&0x2))
      {
         free((*b).ipt);
         free((*b).iup);
      }

      free((*b).imb);

      (*b).vol=0;
      (*b).vbb=0;
      (*b).nbp=0;
      (*b).shf=0x0;
      (*b).ipt=NULL;
      (*b).imb=NULL;
      (*b).ibp=NULL;
      (*b).iup=NULL;
      (*b).idn=NULL;

      if ((!(shf&0x4))&&((*b).u!=NULL))
      {
         afree((*b).u);
         afree((*b).sw);
      }

      if ((!(shf&0x8))&&((*b).ud!=NULL))
      {
         afree((*b).ud);
         afree((*b).swd);
      }

      if ((!(shf&0x10))&&((*b).ns>0))
      {
         afree((*b).s[0]);
         free((*b).s);
      }

      if ((!(shf&0x20))&&((*b).nsd>0))
      {
         afree((*b).sd[0]);
         free((*b).sd);
      }

      (*b).ns=0;
      (*b).nsd=0;
      (*b).u=NULL;
      (*b).ud=NULL;
      (*b).sw=NULL;
      (*b).swd=NULL;
      (*b).s=NULL;
      (*b).sd=NULL;

      return 0;
   }
   else
      return 1;
}
