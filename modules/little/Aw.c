
/*******************************************************************************
*
* File Aw.c
*
* Copyright (C) 2007, 2011, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Application of the single-precision little Wilson-Dirac operator Aw.
*
*   void Aw(complex *v,complex *w)
*     Applies the little Dirac operator to the field v and assigns the
*     result to the field w.
*
*   void Aweeinv(complex *v,complex *w)
*     Applies the inverse of the even-even part of the little Dirac operator
*     to the field v and assigns the result to the field w on the even
*     blocks. On the odd blocks, w is unchanged.
*
*   void Awooinv(complex *v,complex *w)
*     Applies the inverse of the odd-odd part of the little Dirac operator
*     to the field v and assigns the result to the field w on the odd blocks.
*     On the even blocks, w is unchanged.
*
*   void Awoe(complex *v,complex *w)
*     Applies the odd-even part of the little Dirac operator to the field v
*     and assigns the result to the field w on the odd blocks. On the even
*     blocks, w is unchanged.
*
*   void Aweo(complex *v,complex *w)
*     Applies the even-odd part of the little Dirac operator to the field v
*     and *subtracts* the result from the field w on the even blocks. On the
*     odd blocks, w is unchanged.
*
*   void Awhat(complex *v,complex *w)
*     Applies the even-odd preconditioned little Dirac operator to the field
*     v and assigns the result to the field w on the even blocks. On the odd
*     blocks, w is unchanged.
*
* The little Dirac operator and the associated data structures are described
* in the file README.Aw.
*
* The programs Aw(), Awoe() and Aweo() take it for granted that the little
* Dirac operator is up-to-date, while the other programs, Aweeinv(), Awooinv()
* and Awhat(), assume the even-odd preconditioned operator to be up-to-date
* (see Aw_ops.c).
*
* The input fields are assumed to be of the same size (or larger) than the
* fields made available through the workspace facility (see utils/wspace.c)
* Input and output fields must be different.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define AW_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "vflds.h"
#include "linalg.h"
#include "dfl.h"
#include "little.h"
#include "global.h"

static int Ns=0,nbh,nbt;
static int (*inne)[8],(*inno)[8];
static complex *vo;


static void alloc_arrays(void)
{
   int nb,nbb,n,ifc,(*inn)[8];
   dfl_parms_t dfl;
   dfl_grid_t *grd;

   dfl=dfl_parms();
   Ns=dfl.Ns;

   error_root(Ns==0,1,"alloc_arrays [Aw.c]",
              "Deflation parameters are not set");

   set_dfl_geometry();
   grd=dfl_geometry();

   nb=(*grd).nb;
   nbb=(*grd).nbb;
   inn=(*grd).inn;
   nbh=nb/2;
   nbt=nbh/NTHREAD;

   inne=malloc(nb*sizeof(*inne));
   vo=amalloc(Ns*(nbh+nbb/2)*sizeof(*vo),4);
   error((inne==NULL)||(vo==NULL),1,"alloc_arrays [Aw.c]",
         "Unable to allocate auxiliary arrays");
   inno=inne+nbh;
   vo-=Ns*nbh;

   for (n=0;n<nbh;n++)
   {
      for (ifc=0;ifc<8;ifc++)
      {
         if (inn[n][ifc]<nb)
            inne[n][ifc]=Ns*inn[n][ifc];
         else
            inne[n][ifc]=Ns*(inn[n][ifc]-(nbb/2));

         inno[n][ifc]=Ns*inn[n+nbh][ifc];
      }
   }
}


void Aw(complex *v,complex *w)
{
   int k,n,ifc,(*inn)[8];
   complex *rv,*rw,**A;
   Aw_t Aw;

   if (Ns==0)
      alloc_arrays();

   Aw=Awop();

#pragma omp parallel private(k,n,rv,rw,A)
   {
      k=omp_get_thread_num();

      n=2*k*nbt;
      rv=v+Ns*n;
      rw=w+Ns*n;
      A=Aw.Ablk+n;

      for (n=0;n<(2*nbt);n++)
      {
         cmat_vec(Ns,*A,rv,rw);

         A+=1;
         rv+=Ns;
         rw+=Ns;
      }
   }

   cpv_int_bnd(1,v);

#pragma omp parallel private(k,n,ifc,inn,rw,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      inn=inne+n;
      rw=w+Ns*n;
      A=Aw.Ahop+8*n;

      for (n=0;n<nbt;n++)
      {
         for (ifc=0;ifc<8;ifc++)
         {
            cmat_vec_assign(Ns,*A,v+inn[0][ifc],rw);
            A+=1;
         }

         inn+=1;
         rw+=Ns;
      }
   }

   cpv_int_bnd(0,v);

#pragma omp parallel private(k,n,ifc,inn,rw,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      inn=inno+n;
      rw=w+Ns*(n+nbh);
      A=Aw.Ahop+8*(n+nbh);

      for (n=0;n<nbt;n++)
      {
         for (ifc=0;ifc<8;ifc++)
         {
            cmat_vec_assign(Ns,*A,v+inn[0][ifc],rw);
            A+=1;
         }

         inn+=1;
         rw+=Ns;
      }
   }
}


void Aweeinv(complex *v,complex *w)
{
   int k,n;
   complex *rv,*rw,**A;
   Aw_t Aw;

   if (Ns==0)
      alloc_arrays();

   Aw=Awophat();

#pragma omp parallel private(k,n,rv,rw,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      rv=v+Ns*n;
      rw=w+Ns*n;
      A=Aw.Ablk+n;

      for (n=0;n<nbt;n++)
      {
         cmat_vec(Ns,*A,rv,rw);

         A+=1;
         rv+=Ns;
         rw+=Ns;
      }
   }
}


void Awooinv(complex *v,complex *w)
{
   int k,n;
   complex *rv,*rw,**A;
   Aw_t Aw;

   if (Ns==0)
      alloc_arrays();

   Aw=Awophat();

#pragma omp parallel private(k,n,rv,rw,A)
   {
      k=omp_get_thread_num();

      n=nbh+k*nbt;
      A=Aw.Ablk+n;
      rv=v+Ns*n;
      rw=w+Ns*n;

      for (n=0;n<nbt;n++)
      {
         cmat_vec(Ns,*A,rv,rw);

         A+=1;
         rv+=Ns;
         rw+=Ns;
      }
   }
}


void Awoe(complex *v,complex *w)
{
   int k,n,ifc,(*inn)[8];
   complex *rw,**A;
   Aw_t Aw;

   if (Ns==0)
      alloc_arrays();

   cpv_int_bnd(0,v);

   Aw=Awop();

#pragma omp parallel private(k,n,ifc,inn,rw,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      inn=inno+n;
      rw=w+Ns*(nbh+n);
      A=Aw.Ahop+8*(nbh+n);

      for (n=0;n<nbt;n++)
      {
         cmat_vec(Ns,*A,v+inn[0][0],rw);
         A+=1;

         for (ifc=1;ifc<8;ifc++)
         {
            cmat_vec_assign(Ns,*A,v+inn[0][ifc],rw);
            A+=1;
         }

         inn+=1;
         rw+=Ns;
      }
   }
}


void Aweo(complex *v,complex *w)
{
   int k,n,ifc,(*inn)[8];
   complex *ro,*rv,*rw,**A;
   Aw_t Aw;

   if (Ns==0)
      alloc_arrays();

#pragma omp parallel private(k,n,ro,rv)
   {
      k=omp_get_thread_num();

      n=nbh+k*nbt;
      ro=vo+Ns*n;
      rv=v+Ns*n;

      for (n=0;n<(Ns*nbt);n++)
      {
         ro[0].re=-rv[0].re;
         ro[0].im=-rv[0].im;

         rv+=1;
         ro+=1;
      }
   }

   cpv_int_bnd(1,vo);

   Aw=Awop();

#pragma omp parallel private(k,n,ifc,inn,rw,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      inn=inne+n;
      rw=w+Ns*n;
      A=Aw.Ahop+8*n;

      for (n=0;n<nbt;n++)
      {
         for (ifc=0;ifc<8;ifc++)
         {
            cmat_vec_assign(Ns,*A,vo+inn[0][ifc],rw);
            A+=1;
         }

         inn+=1;
         rw+=Ns;
      }
   }
}


void Awhat(complex *v,complex *w)
{
   int k,l,n,ifc,(*inn)[8];
   complex *ro,*rv,*rw,**A;
   Aw_t Aw;

   if (Ns==0)
      alloc_arrays();

   cpv_int_bnd(0,v);

   Aw=Awophat();

#pragma omp parallel private(k,n,ifc,inn,ro,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      inn=inno+n;
      ro=vo+Ns*(nbh+n);
      A=Aw.Ahop+8*(nbh+n);

      for (n=0;n<nbt;n++)
      {
         cmat_vec(Ns,*A,v+inn[0][0],ro);
         A+=1;

         for (ifc=1;ifc<8;ifc++)
         {
            cmat_vec_assign(Ns,*A,v+inn[0][ifc],ro);
            A+=1;
         }

         inn+=1;
         ro+=Ns;
      }
   }

   cpv_int_bnd(1,vo);

#pragma omp parallel private(k,l,n,ifc,inn,rv,rw,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      inn=inne+n;
      rv=v+Ns*n;
      rw=w+Ns*n;
      A=Aw.Ahop+8*n;

      for (n=0;n<nbt;n++)
      {
         cmat_vec(Ns,*A,vo+inn[0][0],rw);
         A+=1;

         for (ifc=1;ifc<8;ifc++)
         {
            cmat_vec_assign(Ns,*A,vo+inn[0][ifc],rw);
            A+=1;
         }

         inn+=1;

         for (l=0;l<Ns;l++)
         {
            rw[0].re=rv[0].re-rw[0].re;
            rw[0].im=rv[0].im-rw[0].im;

            rv+=1;
            rw+=1;
         }
      }
   }
}
