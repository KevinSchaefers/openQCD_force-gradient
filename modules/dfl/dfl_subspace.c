
/*******************************************************************************
*
* File dfl_subspace.c
*
* Copyright (C) 2007-2013, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Basic programs related to the deflation subspace.
*
*   void dfl_s2vd(spinor *s,complex_dble *vd)
*     Assigns the components of the global spinor field s along the
*     deflation subspace to the vector field vd.
*
*   void dfl_vd2s(complex_dble *vd,spinor *s)
*     Assigns the element of the deflation subspace corresponding to
*     the vector field vd to the global spinor field s.
*
*   void dfl_subspace(spinor **mds)
*     Copies the global spinor fields mds[0],..,mds[Ns-1] to the fields
*     b.s[1],..,b.s[Ns] on the blocks b of the DFL_BLOCKS grid. The block
*     fields are then orthonormalized using the Gram-Schmidt process with
*     double-precision accuracy.
*      In this basis of fields, the modes mds[0],..,mds[Ns-1] are given by
*     fields vmds[0],..,vmds[Ns-1] of Ns*nb complex numbers, where nb is
*     the number of blocks in the block grid. These fields are assigned to
*     the last Ns single-precision vector fields of the array returned by
*     vflds() [vflds/vflds.c].
*
*   void dfl_renormalize_modes(spinor **mds)
*     Orthonormalizes the global spinor fields mds[0],..,mds[Ns-1].
*
*   void dfl_restore_modes(spinor **mds)
*     Restores the global spinor fields mds[0],..,mds[Ns-1] that generated
*     the current deflation subspace from the vector fields returned by
*     vflds().
*
* The deflation subspace is spanned by the fields (*b).s[1],..,(*b).s[Ns]
* on the blocks b of the DFL_BLOCKS grid. The number Ns of fields is set by
* the program dfl_set_parms() [flags/dfl_parms.c].
*
* Any spinor field in the deflation subspace is a linear combination of the
* basis elements on the blocks. The associated complex coefficients form a
* vector field of the type described in vflds/vflds.c. Such fields are thus
* in one-to-one correspondence with the deflation modes. In particular, the
* deflation subspace contains the global spinor fields from which it was
* created by the program dfl_subspace().
*
* The program dfl_subspace() allocates the DFL_BLOCKS block grid if it is
* not already allocated.
*
* The programs dfl_s2vd() and dfl_vd2s() are assumed to be called by the
* OpenMP master thread on any MPI process.
*
* The programs dfl_subspace(), dfl_renormalize_modes() and dfl_restore_modes()
* are assumed to be called by the OpenMP master thread on all MPI processes
* simultaneously.
*
* If SSE instructions are used, the Dirac spinors must be aligned to a 16
* byte boundary.
*
*******************************************************************************/

#define DFL_SUBSPACE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "sflds.h"
#include "vflds.h"
#include "linalg.h"
#include "block.h"
#include "dfl.h"
#include "global.h"


void dfl_s2vd(spinor *s,complex_dble *vd)
{
   int Ns,nb,nbh,isw,vol;
   int k,l,n,m;
   complex z;
   complex_dble *wd;
   block_t *b;
   spinor **sb;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   Ns=dfl.Ns;
   b=blk_list(DFL_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;

#pragma omp parallel private(k,l,n,m,z,wd,sb)
   {
      k=omp_get_thread_num();

      for (n=k;n<nb;n+=NTHREAD)
      {
         if (n<nbh)
            m=n+isw*nbh;
         else
            m=n-isw*nbh;

         assign_s2sblk(DFL_BLOCKS,m,ALL_PTS,s,0);
         sb=b[m].s;
         wd=vd+Ns*n;

         for (l=1;l<=Ns;l++)
         {
            z=spinor_prod(vol,0,sb[l],sb[0]);
            (*wd).re=(double)(z.re);
            (*wd).im=(double)(z.im);
            wd+=1;
         }
      }
   }
}


void dfl_vd2s(complex_dble *vd,spinor *s)
{
   int Ns,nb,nbh,isw,vol;
   int k,l,n,m;
   complex z;
   complex_dble *wd;
   block_t *b;
   spinor **sb;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   Ns=dfl.Ns;
   b=blk_list(DFL_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;

#pragma omp parallel private(k,l,n,m,z,wd,sb)
   {
      k=omp_get_thread_num();

      for (n=k;n<nb;n+=NTHREAD)
      {
         if (n<nbh)
            m=n+isw*nbh;
         else
            m=n-isw*nbh;

         sb=b[m].s;
         set_s2zero(vol,0,sb[0]);
         wd=vd+Ns*n;

         for (l=1;l<=Ns;l++)
         {
            z.re=(float)((*wd).re);
            z.im=(float)((*wd).im);
            mulc_spinor_add(vol,0,sb[0],sb[l],z);
            wd+=1;
         }

         assign_sblk2s(DFL_BLOCKS,m,ALL_PTS,0,s);
      }
   }
}


void dfl_subspace(spinor **mds)
{
   int Ns,nb,nbh,isw,vol;
   int k,l,j,n,m,ifail;
   complex **vs,*v;
   complex_dble z;
   complex_qflt cqsm;
   block_t *b;
   spinor **sb;
   spinor_dble **sdb;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   Ns=dfl.Ns;

   error_root(Ns==0,1,"dfl_subspace [dfl_subspace.c]",
              "Deflation subspace parameters are not set");

   b=blk_list(DFL_BLOCKS,&nb,&isw);

   if (nb==0)
   {
      alloc_bgr(DFL_BLOCKS);
      b=blk_list(DFL_BLOCKS,&nb,&isw);
   }

   nbh=nb/2;
   vol=(*b).vol;
   vs=vflds()+Ns;
   ifail=0;

#pragma omp parallel private(k,l,j,n,m,v,z,cqsm,sb,sdb) reduction(| : ifail)
   {
      k=omp_get_thread_num();

      for (n=k;n<nb;n+=NTHREAD)
      {
         if (n<nbh)
            m=n+isw*nbh;
         else
            m=n-isw*nbh;

         sb=b[m].s;
         sdb=b[m].sd;

         for (l=1;l<=Ns;l++)
         {
            assign_s2sdblk(DFL_BLOCKS,m,ALL_PTS,mds[l-1],0);
            v=vs[l-1]+Ns*n;

            for (j=1;j<l;j++)
            {
               assign_s2sd(vol,0,sb[j],sdb[1]);
               cqsm=spinor_prod_dble(vol,0,sdb[1],sdb[0]);
               z.re=cqsm.re.q[0];
               z.im=cqsm.im.q[0];

               (*v).re=(float)(z.re);
               (*v).im=(float)(z.im);
               v+=1;

               z.re=-(z).re;
               z.im=-(z).im;
               mulc_spinor_add_dble(vol,0,sdb[0],sdb[1],z);
            }

            (*v).re=(float)(normalize_dble(vol,0,sdb[0]));
            (*v).im=0.0f;
            ifail|=((*v).re==0.0f);
            v+=1;

            for (j=(l+1);j<=Ns;j++)
            {
               (*v).re=0.0f;
               (*v).im=0.0f;
               v+=1;
            }

            assign_sd2s(vol,0,sdb[0],sb[l]);
         }
      }
   }

   error(ifail,1,"dfl_subspace [dfl_subspace.c]","Degenerate block modes");

   set_flags(ERASED_AW);
   set_flags(ERASED_AWHAT);
}


void dfl_renormalize_modes(spinor **mds)
{
   int Ns,l,j;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   Ns=dfl.Ns;

   for (l=0;l<Ns;l++)
   {
      for (j=0;j<l;j++)
         project(VOLUME_TRD,3,mds[l],mds[j]);

      (void)(normalize(VOLUME_TRD,3,mds[l]));
   }
}


void dfl_restore_modes(spinor **mds)
{
   int Ns,nb,nbh,isw,vol;
   int k,l,j,n,m;
   complex *w,**vs;
   block_t *b;
   spinor **sb;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   Ns=dfl.Ns;
   b=blk_list(DFL_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;
   vs=vflds()+Ns;

#pragma omp parallel private(k,l,j,n,m,w,sb)
   {
      k=omp_get_thread_num();

      for (n=k;n<nb;n+=NTHREAD)
      {
         if (n<nbh)
            m=n+isw*nbh;
         else
            m=n-isw*nbh;

         sb=b[m].s;

         for (l=1;l<=Ns;l++)
         {
            set_s2zero(vol,0,sb[0]);
            w=vs[l-1]+Ns*n;

            for (j=1;j<=l;j++)
            {
               mulc_spinor_add(vol,0,sb[0],sb[j],*w);
               w+=1;
            }

            assign_sblk2s(DFL_BLOCKS,m,ALL_PTS,0,mds[l-1]);
         }
      }
   }
}
