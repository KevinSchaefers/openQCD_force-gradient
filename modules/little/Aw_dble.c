
/*******************************************************************************
*
* File Aw_dble.c
*
* Copyright (C) 2007, 2011, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Application of the double-precision little Wilson-Dirac operator Aw.
*
*   void Aw_dble(complex_dble *vd,complex_dble *wd)
*     Applies the little Dirac operator to the field vd and assigns the
*     result to the field wd.
*
*   void Aweeinv_dble(complex_dble *vd,complex_dble *wd)
*     Applies the inverse of the even-even part of the little Dirac operator
*     to the field vd and assigns the result to the field wd on the even
*     blocks. On the odd blocks, wd is unchanged.
*
*   void Awooinv_dble(complex_dble *vd,complex_dble *wd)
*     Applies the inverse of the odd-odd part of the little Dirac operator
*     to the field vd and assigns the result to the field wd on the odd
*     blocks. On the even blocks, wd is unchanged.
*
*   void Awoe_dble(complex_dble *vd,complex_dble *wd)
*     Applies the odd-even part of the little Dirac operator to the field vd
*     and assigns the result to the field wd on the odd blocks. On the even
*     blocks, wd is unchanged.
*
*   void Aweo_dble(complex_dble *vd,complex_dble *wd)
*     Applies the even-odd part of the little Dirac operator to the field vd
*     and *subtracts* the result from the field wd on the even blocks. On the
*     odd blocks, wd is unchanged.
*
*   void Awhat_dble(complex_dble *vd,complex_dble *wd)
*     Applies the even-odd preconditioned little Dirac operator to the field
*     vd and assigns the result to the field wd on the even blocks. On the
*     odd blocks, wd is unchanged.
*
* The little Dirac operator and the associated data structures are described
* in the file README.Aw.
*
* The programs Aw_dble(), Awoe_dble() and Aweo_dble() take it for granted
* that the little Dirac operator is up-to-date, while the other programs,
* Aweeinv_dble(), Awooinv_dble() and Awhat_dble(), assume the even-odd
* preconditioned operator to be up-to-date (see Aw_ops.c).
*
* The input fields are assumed to be of the same size (or larger) than the
* fields made available through the workspace facility (see utils/wspace.c)
* Input and output fields must be different.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define AW_DBLE_C

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
static complex_dble *vdo;


static void alloc_arrays(void)
{
   int nb,nbb,n,ifc,(*inn)[8];
   dfl_parms_t dfl;
   dfl_grid_t *grd;

   dfl=dfl_parms();
   Ns=dfl.Ns;

   error_root(Ns==0,1,"alloc_arrays [Aw_dble.c]",
              "Deflation parameters are not set");

   set_dfl_geometry();
   grd=dfl_geometry();

   nb=(*grd).nb;
   nbb=(*grd).nbb;
   inn=(*grd).inn;
   nbh=nb/2;
   nbt=nbh/NTHREAD;

   inne=malloc(nb*sizeof(*inne));
   vdo=amalloc(Ns*(nbh+nbb/2)*sizeof(*vdo),5);
   error((inne==NULL)||(vdo==NULL),1,"alloc_arrays [Aw_dble.c]",
         "Unable to allocate auxiliary arrays");
   inno=inne+nbh;
   vdo-=Ns*nbh;

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


void Aw_dble(complex_dble *vd,complex_dble *wd)
{
   int k,n,ifc,(*inn)[8];
   complex_dble *rv,*rw,**A;
   Aw_dble_t Aw;

   if (Ns==0)
      alloc_arrays();

   Aw=Awop_dble();

#pragma omp parallel private(k,n,rv,rw,A)
   {
      k=omp_get_thread_num();

      n=2*k*nbt;
      rv=vd+Ns*n;
      rw=wd+Ns*n;
      A=Aw.Ablk+n;

      for (n=0;n<(2*nbt);n++)
      {
         cmat_vec_dble(Ns,*A,rv,rw);

         A+=1;
         rv+=Ns;
         rw+=Ns;
      }
   }

   cpvd_int_bnd(1,vd);

#pragma omp parallel private(k,n,ifc,inn,rw,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      inn=inne+n;
      rw=wd+Ns*n;
      A=Aw.Ahop+8*n;

      for (n=0;n<nbt;n++)
      {
         for (ifc=0;ifc<8;ifc++)
         {
            cmat_vec_assign_dble(Ns,*A,vd+inn[0][ifc],rw);
            A+=1;
         }

         inn+=1;
         rw+=Ns;
      }
   }

   cpvd_int_bnd(0,vd);

#pragma omp parallel private(k,n,ifc,inn,rw,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      inn=inno+n;
      rw=wd+Ns*(n+nbh);
      A=Aw.Ahop+8*(n+nbh);

      for (n=0;n<nbt;n++)
      {
         for (ifc=0;ifc<8;ifc++)
         {
            cmat_vec_assign_dble(Ns,*A,vd+inn[0][ifc],rw);
            A+=1;
         }

         inn+=1;
         rw+=Ns;
      }
   }
}


void Aweeinv_dble(complex_dble *vd,complex_dble *wd)
{
   int k,n;
   complex_dble *rv,*rw,**A;
   Aw_dble_t Aw;

   if (Ns==0)
      alloc_arrays();

   Aw=Awophat_dble();

#pragma omp parallel private(k,n,rv,rw,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      rv=vd+Ns*n;
      rw=wd+Ns*n;
      A=Aw.Ablk+n;

      for (n=0;n<nbt;n++)
      {
         cmat_vec_dble(Ns,*A,rv,rw);

         A+=1;
         rv+=Ns;
         rw+=Ns;
      }
   }
}


void Awooinv_dble(complex_dble *vd,complex_dble *wd)
{
   int k,n;
   complex_dble *rv,*rw,**A;
   Aw_dble_t Aw;

   if (Ns==0)
      alloc_arrays();

   Aw=Awophat_dble();

#pragma omp parallel private(k,n,rv,rw,A)
   {
      k=omp_get_thread_num();

      n=nbh+k*nbt;
      A=Aw.Ablk+n;
      rv=vd+Ns*n;
      rw=wd+Ns*n;

      for (n=0;n<nbt;n++)
      {
         cmat_vec_dble(Ns,*A,rv,rw);

         A+=1;
         rv+=Ns;
         rw+=Ns;
      }
   }
}


void Awoe_dble(complex_dble *vd,complex_dble *wd)
{
   int k,n,ifc,(*inn)[8];
   complex_dble *rw,**A;
   Aw_dble_t Aw;

   if (Ns==0)
      alloc_arrays();

   cpvd_int_bnd(0,vd);

   Aw=Awop_dble();

#pragma omp parallel private(k,n,ifc,inn,rw,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      inn=inno+n;
      rw=wd+Ns*(nbh+n);
      A=Aw.Ahop+8*(nbh+n);

      for (n=0;n<nbt;n++)
      {
         cmat_vec_dble(Ns,*A,vd+inn[0][0],rw);
         A+=1;

         for (ifc=1;ifc<8;ifc++)
         {
            cmat_vec_assign_dble(Ns,*A,vd+inn[0][ifc],rw);
            A+=1;
         }

         inn+=1;
         rw+=Ns;
      }
   }
}


void Aweo_dble(complex_dble *vd,complex_dble *wd)
{
   int k,n,ifc,(*inn)[8];
   complex_dble *ro,*rv,*rw,**A;
   Aw_dble_t Aw;

   if (Ns==0)
      alloc_arrays();

#pragma omp parallel private(k,n,ro,rv)
   {
      k=omp_get_thread_num();

      n=nbh+k*nbt;
      ro=vdo+Ns*n;
      rv=vd+Ns*n;

      for (n=0;n<(Ns*nbt);n++)
      {
         ro[0].re=-rv[0].re;
         ro[0].im=-rv[0].im;

         rv+=1;
         ro+=1;
      }
   }

   cpvd_int_bnd(1,vdo);

   Aw=Awop_dble();

#pragma omp parallel private(k,n,ifc,inn,rw,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      inn=inne+n;
      rw=wd+Ns*n;
      A=Aw.Ahop+8*n;

      for (n=0;n<nbt;n++)
      {
         for (ifc=0;ifc<8;ifc++)
         {
            cmat_vec_assign_dble(Ns,*A,vdo+inn[0][ifc],rw);
            A+=1;
         }

         inn+=1;
         rw+=Ns;
      }
   }
}


void Awhat_dble(complex_dble *vd,complex_dble *wd)
{
   int k,l,n,ifc,(*inn)[8];
   complex_dble *ro,*rv,*rw,**A;
   Aw_dble_t Aw;

   if (Ns==0)
      alloc_arrays();

   cpvd_int_bnd(0,vd);

   Aw=Awophat_dble();

#pragma omp parallel private(k,n,ifc,inn,ro,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      inn=inno+n;
      ro=vdo+Ns*(nbh+n);
      A=Aw.Ahop+8*(nbh+n);

      for (n=0;n<nbt;n++)
      {
         cmat_vec_dble(Ns,*A,vd+inn[0][0],ro);
         A+=1;

         for (ifc=1;ifc<8;ifc++)
         {
            cmat_vec_assign_dble(Ns,*A,vd+inn[0][ifc],ro);
            A+=1;
         }

         inn+=1;
         ro+=Ns;
      }
   }

   cpvd_int_bnd(1,vdo);

#pragma omp parallel private(k,l,n,ifc,inn,rv,rw,A)
   {
      k=omp_get_thread_num();

      n=k*nbt;
      inn=inne+n;
      rv=vd+Ns*n;
      rw=wd+Ns*n;
      A=Aw.Ahop+8*n;

      for (n=0;n<nbt;n++)
      {
         cmat_vec_dble(Ns,*A,vdo+inn[0][0],rw);
         A+=1;

         for (ifc=1;ifc<8;ifc++)
         {
            cmat_vec_assign_dble(Ns,*A,vdo+inn[0][ifc],rw);
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
