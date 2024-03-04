
/*******************************************************************************
*
* File uidx.c
*
* Copyright (C) 2010-2013, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Labeling of the link variables on the faces of the local lattice.
*
*   void set_uidx(void)
*     Allocates and initializes the index arrays labeling the link variables
*     at the faces of the local lattice.
*
*   uidx_t *uidx(void)
*     Returns an array idx[4] of uidx_t structures containing the offsets
*     of the link variables at the faces of the local lattice.
*
*   void plaq_uidx(int n,int ix,int *ip)
*     Calculates the offsets ip[4] of the links in the (mu,nu)-plaquette at
*     the point on the local lattice with label ix. The indices (mu,nu) are
*     determined by the parameter n=0,..,5.
*
* For a description of the layout of the gauge field array and the contents of
* the index arrays returned by uidx() see main/README.global and README.uidx.
*
* There are six planes
*
*  (mu,nu)={(0,1),(0,2),(0,3),(2,3),(3,1),(1,2)}
*
* labeled by an integer n running from 0 to 5 and the links in the
* (mu,nu)-plaquette at the point x are ordered such that
*
*   ip[0] -> U(x,mu)
*   ip[1] -> U(x+mu,nu)
*   ip[2] -> U(x,nu)
*   ip[3] -> U(x+nu,mu)
*
* In the program plaq_uidx() it is taken for granted that 0<=ix<VOLUME.
*
* If SF or open-SF boundary conditions are chosen, the offsets ip[0],..,ip[3]
* returned by plaq_uidx() at global time NPROC0*L0-1 take into account the
* fact that the constant boundary values of the gauge field are stored at the
* end of the field array. In all cases, the correct field variables are thus
* found at the calculated offsets.
*
* The program set_uidx() is assumed to be called by the OpenMP master thread
* on all MPI processes simultaneously. All other programs are thread-safe and
* can be locally called, but assume that the index arrays have been set up by
* set_uidx().
*
*******************************************************************************/

#define UIDX_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "global.h"

#define N0 (NPROC0*L0)

static const int plns[6][2]={{0,1},{0,2},{0,3},{2,3},{3,1},{1,2}};
static int bc,nfc[4],ofs[4],snu[4],init=0;
static uidx_t idx[4]={{0,0,NULL,NULL}};


static void alloc_idx(void)
{
   int mu,nu0,nuk;
   int *iu0,*iuk;

   bc=bc_type();
   nfc[0]=FACE0/2;
   nfc[1]=FACE1/2;
   nfc[2]=FACE2/2;
   nfc[3]=FACE3/2;

   ofs[0]=VOLUME+(FACE0/2);
   ofs[1]=ofs[0]+(FACE0/2)+(FACE1/2);
   ofs[2]=ofs[1]+(FACE1/2)+(FACE2/2);
   ofs[3]=ofs[2]+(FACE2/2)+(FACE3/2);

   snu[0]=0;
   snu[1]=snu[0]+(FACE0/2);
   snu[2]=snu[1]+(FACE1/2);
   snu[3]=snu[2]+(FACE2/2);

   iu0=malloc(7*(BNDRY/4)*sizeof(*iu0));
   error(iu0==NULL,1,"alloc_idx [uidx.c]",
         "Unable to allocate index array");
   iuk=iu0+(BNDRY/4);

   for (mu=0;mu<4;mu++)
   {
      nu0=nfc[mu];
      nuk=6*nfc[mu];

      idx[mu].nu0=nu0;
      idx[mu].nuk=nuk;

      if (nu0>0)
      {
         idx[mu].iu0=iu0;
         idx[mu].iuk=iuk;
         iu0+=nu0;
         iuk+=nuk;
      }
      else
      {
         idx[mu].iu0=NULL;
         idx[mu].iuk=NULL;
      }
   }
}


static int offset(int ix,int mu)
{
   int iy,ib;

   if (ix<(VOLUME/2))
   {
      iy=iup[ix][mu];

      if (iy<VOLUME)
         return 8*(iy-(VOLUME/2))+2*mu+1;
      else
      {
         ib=iy-ofs[mu]-(BNDRY/2);

         return 4*VOLUME+snu[mu]+ib;
      }
   }
   else
      return 8*(ix-(VOLUME/2))+2*mu;
}


static void set_idx(void)
{
   int n,m,mu;
   int k,l,ib,ib0,ib1,iy,iz,nu;
   int nu0,*iu0,*iuk;

   alloc_idx();

   for (mu=0;mu<4;mu++)
   {
      nu0=idx[mu].nu0;
      iu0=idx[mu].iu0;
      iuk=idx[mu].iuk;

      if (nu0)
      {
         n=nu0/NTHREAD;
         m=nu0-n*NTHREAD;

#pragma omp parallel private(k,l,ib,ib0,ib1,iy,iz,nu)
         {
            k=omp_get_thread_num();
            ib0=k*n;
            ib1=ib0+n;
            if (k==(NTHREAD-1))
               ib1+=m;

            for (ib=ib0;ib<ib1;ib++)
            {
               iy=ib+ofs[mu]+(BNDRY/2);
               iz=map[iy-VOLUME];
               iu0[ib]=8*(iz-(VOLUME/2))+2*mu+1;
            }

            for (ib=ib0;ib<ib1;ib++)
            {
               iy=ib+ofs[mu];
               iz=map[iy-VOLUME];

               for (l=0;l<3;l++)
               {
                  nu=l+(l>=mu);
                  iuk[3*ib+l]=offset(iz,nu);
               }
            }

            for (ib=ib0;ib<ib1;ib++)
            {
               iy=ib+ofs[mu]+(BNDRY/2);
               iz=map[iy-VOLUME];

               for (l=0;l<3;l++)
               {
                  nu=l+(l>=mu);
                  iuk[3*(ib+nu0)+l]=offset(iz,nu);
               }
            }
         }
      }
   }
}


void set_uidx(void)
{
   if (init==0)
   {
      error(ipt==NULL,1,"set_uidx [uidx.c]",
            "Geometry arrays are not set");
      if (BNDRY)
         set_idx();
      else
         bc=bc_type();

      init=1;
   }
}


uidx_t *uidx(void)
{
   return idx;
}


void plaq_uidx(int n,int ix,int *ip)
{
   int mu,nu;
   int iy,ic;

   mu=plns[n][0];
   nu=plns[n][1];

   ip[0]=offset(ix,mu);

   if ((mu==0)&&(global_time(ix)==(N0-1))&&((bc==1)||(bc==2)))
   {
      ip[1]=4*VOLUME+7*(BNDRY/4)+nu-1;
   }
   else
   {
      iy=iup[ix][mu];

      if (iy<VOLUME)
         ip[1]=offset(iy,nu);
      else
      {
         if (iy<(VOLUME+(BNDRY/2)))
            ic=iy-VOLUME-nfc[mu];
         else
            ic=iy-VOLUME-(BNDRY/2);

         ip[1]=4*VOLUME+(BNDRY/4)+3*ic+nu-(nu>mu);
      }
   }

   ip[2]=offset(ix,nu);
   iy=iup[ix][nu];

   if (iy<VOLUME)
      ip[3]=offset(iy,mu);
   else
   {
      if (iy<(VOLUME+(BNDRY/2)))
         ic=iy-VOLUME-nfc[nu];
      else
         ic=iy-VOLUME-(BNDRY/2);

      ip[3]=4*VOLUME+(BNDRY/4)+3*ic+mu-(mu>nu);
   }
}
