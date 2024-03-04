
/*******************************************************************************
*
* File geogen.c
*
* Copyright (C) 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic programs related to the global lattice geometry.
*
*   int ipr_global(int *n)
*     Returns the rank of the MPI process with Cartesian coordinates
*     n[0],..,n[3] in the process grid.
*
*   void set_cpr(void)
*     Computes the coordinates cpr[4] of the local MPI process and the
*     ranks npr[8] of the MPI processes that host the nearest-neighour
*     local lattices.
*
*   void set_sbofs(void)
*     Computes the offsets sbofs[16] and volumes sbvol[16] of the 16
*     blocks that divide the thread-local lattice.
*
*   void set_iupdn(void)
*     Assuming the array ipt[VOLUME] has been set up, the nearest-neighbour
*     index arrays iup[VOLUME][4] and idn[VOLUME][4] are allocated and their
*     elements are set to the indices of the neighbouring points if these
*     are in the local lattice. All other elements are set to VOLUME.
*
*   void set_map(void)
*     Assuming set_iupdn() has previously been called, the boundary index
*     array map[BNDRY] (map[1] if NPROC=1) is allocated and its elements
*     are set to the indices of the inner boundary points of the local
*     lattice. The program moreover completes the initialization of the
*     the arrays iup[VOLUME][4], idn[VOLUME][4] along the boundaries of
*     the local lattice.
*
* See main/README.global for a description of the lattice geometry and the
* meaning of the global arrays ipt[VOLUME], iup[VOLUME][4], idn[VOLUME][4],
* map[BNDRY], sbofs[16] and sbvol[16].
*
* Apart from ipr_global(), the programs are intended to be called only by
* the program geometry() [geometry.c], which sets up all global geometry
* index arrays.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define GEOGEN_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "utils.h"
#include "lattice.h"
#include "global.h"

#define NPROC_BLK (NPROC0_BLK*NPROC1_BLK*NPROC2_BLK*NPROC3_BLK)
#define NBLK0 (NPROC0/NPROC0_BLK)
#define NBLK1 (NPROC1/NPROC1_BLK)
#define NBLK2 (NPROC2/NPROC2_BLK)
#define NBLK3 (NPROC3/NPROC3_BLK)


int ipr_global(int *n)
{
   int ib,ip;
   int n0,n1,n2,n3;
   int nb0,nb1,nb2,nb3;
   int np0,np1,np2,np3;

   n0=safe_mod(n[0],NPROC0);
   n1=safe_mod(n[1],NPROC1);
   n2=safe_mod(n[2],NPROC2);
   n3=safe_mod(n[3],NPROC3);

   nb0=n0/NPROC0_BLK;
   nb1=n1/NPROC1_BLK;
   nb2=n2/NPROC2_BLK;
   nb3=n3/NPROC3_BLK;

   np0=n0%NPROC0_BLK;
   np1=n1%NPROC1_BLK;
   np2=n2%NPROC2_BLK;
   np3=n3%NPROC3_BLK;

   ib=nb0;
   ib=ib*NBLK1+nb1;
   ib=ib*NBLK2+nb2;
   ib=ib*NBLK3+nb3;

   ip=np0;
   ip=ip*NPROC1_BLK+np1;
   ip=ip*NPROC2_BLK+np2;
   ip=ip*NPROC3_BLK+np3;

   return ip+ib*NPROC_BLK;
}


static void set_npr(void)
{
   int mu,n[4];

   for (mu=0;mu<4;mu++)
      n[mu]=cpr[mu];

   for (mu=0;mu<4;mu++)
   {
      n[mu]-=1;
      npr[2*mu]=ipr_global(n);
      n[mu]+=2;
      npr[2*mu+1]=ipr_global(n);
      n[mu]-=1;
   }
}


void set_cpr(void)
{
   int ib,ip;
   int np,nr;

   MPI_Comm_size(MPI_COMM_WORLD,&np);
   MPI_Comm_rank(MPI_COMM_WORLD,&nr);

   error(np!=NPROC,1,"set_cpr [geogen.c]",
         "Actual number of processes does not match NPROC");

   ib=nr/NPROC_BLK;
   ip=nr%NPROC_BLK;

   cpr[3]=(ib%NBLK3)*NPROC3_BLK+(ip%NPROC3_BLK);
   ib/=NBLK3;
   ip/=NPROC3_BLK;

   cpr[2]=(ib%NBLK2)*NPROC2_BLK+(ip%NPROC2_BLK);
   ib/=NBLK2;
   ip/=NPROC2_BLK;

   cpr[1]=(ib%NBLK1)*NPROC1_BLK+(ip%NPROC1_BLK);
   ib/=NBLK1;
   ip/=NPROC1_BLK;

   cpr[0]=ib*NPROC0_BLK+ip;

   set_npr();
}


void set_sbofs(void)
{
   int ib,ic,i0,i1,i2,i3;
   int bs[2][4];

   bs[0][0]=L0_TRD/2+((L0_TRD/2)%2);
   bs[1][0]=L0_TRD-bs[0][0];
   bs[0][1]=L1_TRD/2+((L1_TRD/2)%2);
   bs[1][1]=L1_TRD-bs[0][1];
   bs[0][2]=L2_TRD/2+((L2_TRD/2)%2);
   bs[1][2]=L2_TRD-bs[0][2];
   bs[0][3]=L3_TRD/2+((L3_TRD/2)%2);
   bs[1][3]=L3_TRD-bs[0][3];

   for (ib=0;ib<16;ib++)
   {
      ic=ib;
      i3=ic%2;
      ic/=2;
      i2=ic%2;
      ic/=2;
      i1=ic%2;
      ic/=2;
      i0=ic;

      sbvol[ib]=bs[i0][0]*bs[i1][1]*bs[i2][2]*bs[i3][3];
      sbofs[ib]=0;

      for (ic=0;ic<ib;ic++)
         sbofs[ib]+=sbvol[ic];
   }
}


static void alloc_iupdn(void)
{
   iup=malloc(VOLUME*sizeof(*iup));
   idn=malloc(VOLUME*sizeof(*idn));

   error((iup==NULL)||(idn==NULL),1,"alloc_iupdn [geogen.c]",
         "Unable to allocate index arrays");
}


static int index(int x0,int x1,int x2,int x3)
{
   int y0,y1,y2,y3;

   y0=safe_mod(x0,L0);
   y1=safe_mod(x1,L1);
   y2=safe_mod(x2,L2);
   y3=safe_mod(x3,L3);

   return ipt[y3+y2*L3+y1*L2*L3+y0*L1*L2*L3];
}


void set_iupdn(void)
{
   int k,x0,x1,x2,x3;
   int ix,iy,iz;

   if (iup==NULL)
   {
      error(ipt==NULL,1,"set_iupdn [geogen.c]",
            "Index array ipt[VOLUME] is not allocated");
      alloc_iupdn();

#pragma omp parallel private(k,x0,x1,x2,x3,ix,iy,iz)
      {
         k=omp_get_thread_num();

         for (iy=k*VOLUME_TRD;iy<(k+1)*VOLUME_TRD;iy++)
         {
            iz=iy;
            x3=iz%L3;
            iz/=L3;
            x2=iz%L2;
            iz/=L2;
            x1=iz%L1;
            iz/=L1;
            x0=iz;

            ix=index(x0,x1,x2,x3);

            iup[ix][0]=index(x0+1,x1,x2,x3);
            idn[ix][0]=index(x0-1,x1,x2,x3);

            iup[ix][1]=index(x0,x1+1,x2,x3);
            idn[ix][1]=index(x0,x1-1,x2,x3);

            iup[ix][2]=index(x0,x1,x2+1,x3);
            idn[ix][2]=index(x0,x1,x2-1,x3);

            iup[ix][3]=index(x0,x1,x2,x3+1);
            idn[ix][3]=index(x0,x1,x2,x3-1);

            if ((x0==(L0-1))&&(NPROC0>1))
               iup[ix][0]=VOLUME;
            if ((x0==0)&&(NPROC0>1))
               idn[ix][0]=VOLUME;

            if ((x1==(L1-1))&&(NPROC1>1))
               iup[ix][1]=VOLUME;
            if ((x1==0)&&(NPROC1>1))
               idn[ix][1]=VOLUME;

            if ((x2==(L2-1))&&(NPROC2>1))
               iup[ix][2]=VOLUME;
            if ((x2==0)&&(NPROC2>1))
               idn[ix][2]=VOLUME;

            if ((x3==(L3-1))&&(NPROC3>1))
               iup[ix][3]=VOLUME;
            if ((x3==0)&&(NPROC3>1))
               idn[ix][3]=VOLUME;
         }
      }
   }
}


static int (*alloc_map(void))[16]
{
   int (*nfc)[16];

   map=malloc((BNDRY+(NPROC%2))*sizeof(*map));
   if (NPROC>1)
      nfc=amalloc(2*NTHREAD*sizeof(*nfc),6);
   else
      nfc=NULL;

   error((map==NULL)||((NPROC>1)&&(nfc==NULL)),1,"alloc_map [geogen.c]",
         "Unable to allocate index arrays");

   return nfc;
}


void set_map(void)
{
   int k,l,j,ix,iw,iz,ofs;
   int bs[4],*nfc,*ifc;
   int (*nfc_all)[16],(*ifc_all)[16];

   if (map==NULL)
   {
      error(iup==NULL,1,"set_map [geogen.c]",
            "Index array iup[VOLUME][4] is not allocated");
      nfc_all=alloc_map();
      ifc_all=nfc_all+NTHREAD;

      if (NPROC==1)
      {
         map[0]=0;
         return;
      }

      bs[0]=L0;
      bs[1]=L1;
      bs[2]=L2;
      bs[3]=L3;

#pragma omp parallel private(k,l,j,ix,iw,iz,ofs,nfc,ifc)
      {
         k=omp_get_thread_num();
         nfc=nfc_all[k];
         ifc=ifc_all[k];

         for (l=0;l<16;l++)
            nfc[l]=0;

         for (ix=k*VOLUME_TRD;ix<(k+1)*VOLUME_TRD;ix++)
         {
            if (ix<(VOLUME/2))
               ofs=0;
            else
               ofs=8;

            for (l=0;l<4;l++)
            {
               if (idn[ix][l]==VOLUME)
                  nfc[2*l+ofs]+=1;
               if (iup[ix][l]==VOLUME)
                  nfc[2*l+1+ofs]+=1;
            }
         }

#pragma omp barrier

         ifc[0]=(BNDRY/2);
         ifc[1]=ifc[0]+(FACE0/2);
         ifc[2]=ifc[1]+(FACE0/2);
         ifc[3]=ifc[2]+(FACE1/2);
         ifc[4]=ifc[3]+(FACE1/2);
         ifc[5]=ifc[4]+(FACE2/2);
         ifc[6]=ifc[5]+(FACE2/2);
         ifc[7]=ifc[6]+(FACE3/2);

         ifc[ 8]=0;
         ifc[ 9]=ifc[ 8]+(FACE0/2);
         ifc[10]=ifc[ 9]+(FACE0/2);
         ifc[11]=ifc[10]+(FACE1/2);
         ifc[12]=ifc[11]+(FACE1/2);
         ifc[13]=ifc[12]+(FACE2/2);
         ifc[14]=ifc[13]+(FACE2/2);
         ifc[15]=ifc[14]+(FACE3/2);

         for (l=0;l<16;l++)
         {
            for (j=0;j<k;j++)
               ifc[l]+=nfc_all[j][l];
         }

         for (ix=k*VOLUME_TRD;ix<(k+1)*VOLUME_TRD;ix++)
         {
            if (ix<(VOLUME/2))
               ofs=0;
            else
               ofs=8;

            for (l=0;l<4;l++)
            {
               if (idn[ix][l]==VOLUME)
               {
                  iz=ifc[2*l+ofs];
                  ifc[2*l+ofs]+=1;

                  idn[ix][l]=VOLUME+iz;
                  iw=ix;

                  for (j=1;j<bs[l];j++)
                     iw=iup[iw][l];

                  map[iz]=iw;
               }

               if (iup[ix][l]==VOLUME)
               {
                  iz=ifc[2*l+1+ofs];
                  ifc[2*l+1+ofs]+=1;

                  iup[ix][l]=VOLUME+iz;
                  iw=ix;

                  for (j=1;j<bs[l];j++)
                     iw=idn[iw][l];

                  map[iz]=iw;
               }
            }
         }
      }

      afree(nfc_all);
   }
}
