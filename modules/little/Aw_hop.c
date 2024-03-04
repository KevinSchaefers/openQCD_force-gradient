
/*******************************************************************************
*
* File Aw_hop.c
*
* Copyright (C) 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the hopping terms of the little Dirac operator.
*
*   void set_Awhop(void)
*     Computes the hopping terms of the double-precision little Dirac
*     operator.
*
* For a description of the little Dirac operator and the associated data
* structures see README.Aw. The computation of the hopping terms and the
* associated communication requirements are outlined in README.Aw_com.
*
* This program is called by the programs in Aw_ops.c and is not intended
* to be called from anywhere else (the prototype in little.h is masked
* accordingly). It is assumed to be called by the OpenMP master thread
* on all MPI processes simultaneously.
*
*******************************************************************************/

#define AW_HOP_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "linalg.h"
#include "uflds.h"
#include "sflds.h"
#include "block.h"
#include "dfl.h"
#include "little.h"
#include "global.h"

typedef struct
{
   int n,ifc,m,ibn,volh;
   spinor *se;
   spinor_dble *sde,*sdo;
} b2b_t;

static int Ns=0,nmu[8],(*inn)[8],*idx,*obbe,*obbo;
static complex_dble **Amats;
static su3_dble *udb;
static spinor *flds;
static b2b_t *b2bs;


static int alloc_b2b(b2b_t *b2b)
{
   int nb,isw,ifc,n,m;
   char *addr;
   block_t *b;

   b=blk_list(DFL_BLOCKS,&nb,&isw);
   n=0;

   for (ifc=0;ifc<8;ifc+=2)
   {
      m=(*b).bb[ifc].vol;

      if (n<m)
         n=m;
   }

   addr=amalloc(Ns*n*sizeof(spinor_dble),5);

   if (addr==NULL)
      return 1;
   else
   {
      (*b2b).se=(spinor*)(addr);
      (*b2b).sdo=(spinor_dble*)(addr);
      (*b2b).sde=(*b2b).sdo+Ns*(n/2);

      return 0;
   }
}


static void alloc_bufs(void)
{
   int nb,isw,n,m,ifc;
   int k,ie;
   complex_dble *w;
   block_t *b;
   Aw_dble_t Aw;
   dfl_grid_t *grd;

   udb=udfld();

   Aw=Awop_dble();
   grd=dfl_geometry();

   Ns=Aw.Ns;
   inn=(*grd).inn;
   idx=(*grd).idx;
   obbe=(*grd).obbe;
   obbo=(*grd).obbo;

   for (ifc=0;ifc<8;ifc++)
      nmu[ifc]=cpr[ifc/2]&0x1;

   b2bs=malloc(NTHREAD*sizeof(*b2bs));
   error(b2bs==NULL,1,"alloc_bufs [Aw_hop.c]",
         "Unable to allocate buffers");

   ie=0;

#pragma omp parallel private(k) reduction(| : ie)
   {
      k=omp_get_thread_num();
      ie=alloc_b2b(b2bs+k);
   }

   error(ie!=0,1,"alloc_bufs [Aw_hop.c]",
         "Unable to allocate b2b auxiliary arrays");

   if (NPROC>1)
   {
      n=0;

      for (ifc=0;ifc<8;ifc++)
      {
         m=(*grd).nbbe[ifc];

         if (n<m)
            n=m;
      }

      Amats=malloc(n*sizeof(*Amats));
      w=amalloc(n*Ns*Ns*sizeof(*w),5);
      error((Amats==NULL)||(w==NULL),2,"alloc_bufs [Aw_hop.c]",
            "Unable to allocate buffers");

      for (m=0;m<n;m++)
      {
         Amats[m]=w;
         w+=Ns*Ns;
      }

      b=blk_list(DFL_BLOCKS,&nb,&isw);
      n=0;

      for (ifc=0;ifc<8;ifc++)
      {
         m=(*grd).nbbe[ifc]*((*b).bb[ifc].vol/2);

         if (n<m)
            n=m;
      }

      flds=amalloc(n*Ns*sizeof(*flds),5);
      error(flds==NULL,3,"alloc_bufs [Aw_hop.c]",
            "Unable to allocate buffers");
   }
   else
   {
      Amats=NULL;
      flds=NULL;
   }
}


static void get_flds(int ieo,int ifc,int n,b2b_t *b2b)
{
   int nb,isw,ibn,m,volh;
   block_t *b;

   b=blk_list(DFL_BLOCKS,&nb,&isw);
   volh=(*b).bb[ifc].vol/2;
   m=inn[n][ifc];

   if (m<nb)
   {
      ibn=0;
      gather_se(Ns,ifc^0x1,b+idx[m],(*b2b).se);
      assign_s2sd(Ns*volh,0,(*b2b).se,(*b2b).sde);
   }
   else
   {
      ibn=1;

      if (ieo)
         m=m-nb-obbe[ifc];
      else
         m=m-nb-obbo[ifc];

      assign_s2sd(Ns*volh,0,flds+m*Ns*volh,(*b2b).sde);
   }

   gather_ud(ifc,udb,b+idx[n]);
   gather_so(Ns,ifc,b+idx[n],(*b2b).sdo);

   (*b2b).n=n;
   (*b2b).ifc=ifc;
   (*b2b).m=m;
   (*b2b).ibn=ibn;
   (*b2b).volh=volh;
}


static void add2Ahop(int ieo,b2b_t *b2b,complex_dble **Ahop)
{
   int k,l,n,ifc,m,ibn,volh;
   complex_dble z,w,sp[2];
   complex_dble *A,*B;
   spinor_dble *sde,*sdo;

   n=(*b2b).n;
   ifc=(*b2b).ifc;
   m=(*b2b).m;
   ibn=(*b2b).ibn;
   volh=(*b2b).volh;
   sde=(*b2b).sde;
   sdo=(*b2b).sdo;

   A=Ahop[8*n+ifc];

   if (ibn)
      B=Amats[m];
   else
      B=Ahop[8*m+(ifc^0x1)];

   for (k=0;k<Ns;k++)
   {
      for (l=0;l<Ns;l++)
      {
         spinor_prod_gamma[ifc/2](volh,sdo+k*volh,sde+l*volh,sp);

         if (ifc&0x1)
         {
            z.re=-0.5*(sp[0].re-sp[1].re);
            z.im=-0.5*(sp[0].im-sp[1].im);

            w.re=-0.5*(sp[0].re+sp[1].re);
            w.im= 0.5*(sp[0].im+sp[1].im);
         }
         else
         {
            z.re=-0.5*(sp[0].re+sp[1].re);
            z.im=-0.5*(sp[0].im+sp[1].im);

            w.re=-0.5*(sp[0].re-sp[1].re);
            w.im= 0.5*(sp[0].im-sp[1].im);
         }

         if ((ieo)&&(ibn==0))
         {
            A[k*Ns+l].re+=z.re;
            A[k*Ns+l].im+=z.im;

            B[l*Ns+k].re+=w.re;
            B[l*Ns+k].im+=w.im;
         }
         else
         {
            A[k*Ns+l].re=z.re;
            A[k*Ns+l].im=z.im;

            B[l*Ns+k].re=w.re;
            B[l*Ns+k].im=w.im;
         }
      }
   }
}


void set_Awhop(void)
{
   int nb,nbh,ieo,ifc;
   int k,n;
   complex_dble **Ahop;
   Aw_dble_t Aw;

   if (Ns==0)
      alloc_bufs();

   Aw=Awop_dble();
   Ahop=Aw.Ahop;
   nb=Aw.nb;
   nbh=nb/2;

   for (ieo=0;ieo<2;ieo++)
   {
      for (ifc=0;ifc<8;ifc++)
      {
         cpse_int_bnd(ieo,ifc,flds);

#pragma omp parallel private(k,n)
         {
            k=omp_get_thread_num();

            for (n=k;n<nbh;n+=NTHREAD)
            {
               get_flds(ieo^nmu[ifc],ifc^nmu[ifc],n+(ieo^nmu[ifc])*nbh,b2bs+k);
               add2Ahop(ieo,b2bs+k,Ahop);
            }
         }

         cpAhop_ext_bnd(ieo,ifc,Amats);
      }
   }
}
