
/*******************************************************************************
*
* File ftensor.c
*
* Copyright (C) 2010-2013, 2016, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the symmetric field tensor.
*
*   u3_alg_dble **ftensor(void)
*     Computes the symmetric field tensor of the global double-precision
*     gauge field and returns the pointers ft[0],..,ft[5] to the field
*     components with the Lorentz indices (0,1),(0,2),(0,3),(2,3),(3,1),
*     (1,2). The arrays are automatically allocated if needed. Along the
*     boundaries of the lattice (if any), the program sets the field to
*     zero.
*
* At all points x in the interior of the global lattice, the (mu,nu)-component
* of the field tensor is defined by
*
*  F_{mu,nu}(x) = (1/8)*[Q_{mu,nu}(x)-Q_{nu,mu}(x)],
*
* where
*
*  Q_{mu,nu}(x) = U(x,mu)*U(x+mu,nu)*U(x+nu,mu)^dag*U(x,nu)^dag + (3 more)
*
* denotes the sum of the four plaquette loops at x in the (mu,nu)-plane (the
* same as in the case of the SW term). Elsewhere the elements of the field
* arrays are set to zero. The interior points are those at global time x0
* in the range
*
*  0<x0<NPROC0*L0-1        (open bc),
*
*  0<x0<NPROC0*L0          (SF and open-SF bc),
*
*  0<=x0<NPROC0*L0         (periodic bc).
*
* Note that the field tensor calculated here is in the Lie algebra of U(3)
* not SU(3). The type u3_alg_dble is explained in the notes in the module
* su3fcts/su3prod.c.
*
* The program ftensor() acts globally and must be called simultaneously by
* the OpenMP master thread on all MPI processes.
*
*******************************************************************************/

#define FTENSOR_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "su3fcts.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "linalg.h"
#include "tcharge.h"
#include "global.h"

#define N0 (NPROC0*L0)

static int (*ofs_ft)[6][2];
static u3_alg_dble **fts=NULL,**ft;
static ftidx_t *idx;


static void alloc_fts(void)
{
   int n,nbf,k,*a,*b;
   u3_alg_dble **pp,*p;

   error_root(sizeof(u3_alg_dble)!=(9*sizeof(double)),1,
              "alloc_fts [ftensor.c]",
              "The u3_alg_dble structures are not properly packed");

   set_uidx();
   set_ftidx();
   idx=ftidx();
   nbf=0;

   for (n=0;n<6;n++)
      nbf+=idx[n].nft[0]+idx[n].nft[1];

   pp=malloc(12*sizeof(*pp));
   p=amalloc((6*VOLUME+nbf)*sizeof(*p),ALIGN);
   ofs_ft=malloc(NTHREAD*sizeof(*ofs_ft));
   a=malloc(2*NTHREAD*sizeof(*a));
   error((pp==NULL)||(p==NULL)||(ofs_ft==NULL)||(a==NULL),1,
         "alloc_fts [ftensor.c]","Unable to allocate field tensor arrays");

   fts=pp;
   ft=pp+6;
   b=a+NTHREAD;

   for (n=0;n<6;n++)
   {
      nbf=idx[n].nft[0]+idx[n].nft[1];

      (*pp)=p;
      pp+=1;
      p+=VOLUME+nbf;

      divide_range(VOLUME+nbf,NTHREAD,a,b);

      for (k=0;k<NTHREAD;k++)
      {
         ofs_ft[k][n][0]=a[k];
         ofs_ft[k][n][1]=b[k]-a[k];
      }
   }

   free(a);
}


static void add_X2ft(u3_alg_dble *X,u3_alg_dble *f)
{
   double r;

   r=0.125;
   (*f).c1+=r*(*X).c1;
   (*f).c2+=r*(*X).c2;
   (*f).c3+=r*(*X).c3;
   (*f).c4+=r*(*X).c4;
   (*f).c5+=r*(*X).c5;
   (*f).c6+=r*(*X).c6;
   (*f).c7+=r*(*X).c7;
   (*f).c8+=r*(*X).c8;
   (*f).c9+=r*(*X).c9;
}


static void build_fts_part(int n,int ofs,int vol)
{
   int bc,tmx;
   int ix,t,ip[4],ipf[4];
   u3_alg_dble X,*ftn;
   su3_dble w1 ALIGNED16;
   su3_dble w2 ALIGNED16;
   su3_dble *ub;

   bc=bc_type();

   tmx=N0;
   if (bc==0)
      tmx-=1;
   if (n<3)
      tmx-=1;

   ub=udfld();
   ftn=fts[n];

   for (ix=ofs;ix<(ofs+vol);ix++)
   {
      t=global_time(ix);
      plaq_uidx(n,ix,ip);
      plaq_ftidx(n,ix,ipf);

      if (((t>0)&&(t<tmx))||(bc==3))
      {
         su3xsu3(ub+ip[0],ub+ip[1],&w1);
         su3dagxsu3dag(ub+ip[3],ub+ip[2],&w2);
         prod2u3alg(&w1,&w2,&X);
         add_X2ft(&X,ftn+ipf[0]);
         prod2u3alg(&w2,&w1,&X);
         add_X2ft(&X,ftn+ipf[3]);

         su3dagxsu3(ub+ip[2],ub+ip[0],&w1);
         su3xsu3dag(ub+ip[1],ub+ip[3],&w2);
         prod2u3alg(&w1,&w2,&X);
         add_X2ft(&X,ftn+ipf[2]);
         prod2u3alg(&w2,&w1,&X);
         add_X2ft(&X,ftn+ipf[1]);
      }
      else if ((t==0)&&(n<3))
      {
         su3xsu3(ub+ip[0],ub+ip[1],&w1);
         su3dagxsu3dag(ub+ip[3],ub+ip[2],&w2);
         prod2u3alg(&w2,&w1,&X);
         add_X2ft(&X,ftn+ipf[3]);

         su3dagxsu3(ub+ip[2],ub+ip[0],&w1);
         su3xsu3dag(ub+ip[1],ub+ip[3],&w2);
         prod2u3alg(&w2,&w1,&X);
         add_X2ft(&X,ftn+ipf[1]);
      }
      else if ((t==tmx)&&(n<3))
      {
         su3xsu3(ub+ip[0],ub+ip[1],&w1);
         su3dagxsu3dag(ub+ip[3],ub+ip[2],&w2);
         prod2u3alg(&w1,&w2,&X);
         add_X2ft(&X,ftn+ipf[0]);

         su3dagxsu3(ub+ip[2],ub+ip[0],&w1);
         su3xsu3dag(ub+ip[1],ub+ip[3],&w2);
         prod2u3alg(&w1,&w2,&X);
         add_X2ft(&X,ftn+ipf[2]);
      }
   }
}


static void build_fts(void)
{
   int n;
   int k,isb,ofs,vol;
   u3_alg_dble *ftn;

   for (n=0;n<6;n++)
   {
      ftn=fts[n];

#pragma omp parallel private(k,isb,ofs,vol)
      {
         k=omp_get_thread_num();
         set_ualg2zero(ofs_ft[k][n][1],0,ftn+ofs_ft[k][n][0]);

         for (isb=0;isb<16;isb++)
         {
            ofs=k*(VOLUME_TRD/2)+sbofs[isb]/2;
            vol=sbvol[isb]/2;

#pragma omp barrier
            build_fts_part(n,ofs,vol);
            build_fts_part(n,ofs+(VOLUME/2),vol);
         }
      }

      add_bnd_ft(n,ftn);
   }
}


u3_alg_dble **ftensor(void)
{
   int n;

   if (query_flags(FTS_UP2DATE)!=1)
   {
      if (fts==NULL)
         alloc_fts();

      if (query_flags(UDBUF_UP2DATE)!=1)
         copy_bnd_ud();

      build_fts();
      set_flags(COMPUTED_FTS);
   }

   for (n=0;n<6;n++)
      ft[n]=fts[n];

   return ft;
}
