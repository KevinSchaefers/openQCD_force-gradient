
/*******************************************************************************
*
* File sdcom.c
*
* Copyright (C) 2005, 2008, 2011, 2013, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Communication functions for double-precision spinor fields.
*
*   void cpsd_int_bnd(int is,spinor_dble *sd)
*     Copies the spinors sd at the even interior boundary points of the
*     local lattice to the corresponding points on the neighbouring MPI
*     processes. Only half of the spinor components are copied, namely
*     theta[ifc^(is&0x1)]*sd, where ifc labels the faces of the local
*     lattice on the sending process.
*
*   void cpsd_ext_bnd(int is,spinor_dble *sd)
*     Copies the spinors sd at the even exterior boundary points of the
*     local lattice to the neighbouring MPI processes and *adds* them to
*     the field on the matching points of the target lattices. Only half
*     of the spinor components are copied, assuming the spinors sd satisfy
*     sd=theta[ifc^(is&0x1)]*sd, where ifc labels the faces of the local
*     lattice on the sending process.
*
* The spinor fields passed to cpsd_int_bnd() and cpsd_ext_bnd() must have at
* least NSPIN elements. They are interpreted as quark fields on the local
* lattice as described in main/README.global and doc/dirac.pdf. The projector
* theta[ifc] is defined at the top of the module sflds/Pbnd.c.
*
* If open, SF or open-SF boundary conditions are chosen, the programs do
* not copy any spinors in the time direction across the boundaries of the
* global lattice. The spinors on the even points at the boundaries are instead
* set to zero as required by the boundary conditions (see doc/dirac.pdf). More
* precisely, cpsd_int_bnd() sets them to zero *before* copying any spinors in
* the space directions, while cpsd_ext_bnd() does the opposite.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define SDCOM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "sflds.h"
#include "global.h"

static int bc,npts,nmu[8],nfc[8],pofs[8],sflg[8],tags[8];
static int *pts,(*ofs_pts)[2],(*ofs_get)[8][2],(*ofs_put)[8][2]=NULL;
static weyl_dble *sbufs[2],*rbufs[2];
static const spinor_dble sd0={{{0.0}}};
static MPI_Request snd_req[8],rcv_req[8];


static void set_ofs(void)
{
   int ifc,k;
   int *a,*b;

   error(ipt==NULL,1,"set_ofs [sdcom.c]",
         "Geometry arrays are not set");

   bc=bc_type();
   pts=bnd_pts(&npts);

   nfc[0]=FACE0/2;
   nfc[1]=FACE0/2;
   nfc[2]=FACE1/2;
   nfc[3]=FACE1/2;
   nfc[4]=FACE2/2;
   nfc[5]=FACE2/2;
   nfc[6]=FACE3/2;
   nfc[7]=FACE3/2;

   for (ifc=0;ifc<8;ifc++)
   {
      nmu[ifc]=cpr[ifc/2]&0x1;

      if (ifc>0)
         pofs[ifc]=pofs[ifc-1]+nfc[ifc-1];
      else
         pofs[ifc]=0;

      sflg[ifc]=(nfc[ifc]>0)&&((ifc>1)||
                               ((ifc==0)&&(cpr[0]!=0))||
                               ((ifc==1)&&(cpr[0]!=(NPROC0-1)))||
                               (bc==3));
      tags[ifc]=mpi_permanent_tag();
   }

   a=malloc(2*NTHREAD*sizeof(*a));
   ofs_pts=malloc(NTHREAD*sizeof(*ofs_pts));
   ofs_get=malloc(2*NTHREAD*sizeof(*ofs_get));
   error((a==NULL)||(ofs_pts==NULL)||(ofs_get==NULL),1,
         "set_ofs [scom.c]","Unable to allocate auxiliary arrays");
   b=a+NTHREAD;
   ofs_put=ofs_get+NTHREAD;

   divide_range(npts/2,NTHREAD,a,b);

   for (k=0;k<NTHREAD;k++)
   {
      ofs_pts[k][0]=a[k];
      ofs_pts[k][1]=b[k]-a[k];
   }

   for (ifc=0;ifc<8;ifc++)
   {
      if (ifc==0)
         divide_range(nfc[ifc],NTHREAD,a,b);
      else
      {
         a[0]=0;
         b[0]=nfc[ifc];
         divide_range(nfc[ifc],NTHREAD-1,a+1,b+1);
      }

      for (k=0;k<NTHREAD;k++)
      {
         ofs_get[k][ifc][0]=a[k];
         ofs_get[k][ifc][1]=b[k]-a[k];
      }

      if (ifc==7)
         divide_range(nfc[ifc],NTHREAD,a,b);
      else
      {
         a[0]=0;
         b[0]=nfc[ifc];
         divide_range(nfc[ifc],NTHREAD-1,a+1,b+1);
      }

      for (k=0;k<NTHREAD;k++)
      {
         ofs_put[k][ifc][0]=a[k];
         ofs_put[k][ifc][1]=b[k]-a[k];
      }
   }

   free(a);
}


static void alloc_sbufs(void)
{
   int ifc,n;
   weyl_dble *wd;

   n=0;

   for (ifc=0;ifc<8;ifc++)
   {
      if (n<nfc[ifc])
         n=nfc[ifc];
   }

   wd=amalloc(4*n*sizeof(*wd),ALIGN);
   error(wd==NULL,1,"alloc_sbufs [sdcom.c]",
         "Unable to allocate communication buffers");
   sbufs[0]=wd;
   wd+=n;
   sbufs[1]=wd;
   wd+=n;
   rbufs[0]=wd;
   wd+=n;
   rbufs[1]=wd;
}


static void set_requests(void)
{
   int ifc,sfc;
   int saddr,raddr,tag;

   for (ifc=0;ifc<8;ifc++)
   {
      sfc=ifc^nmu[ifc];

      if (sflg[ifc])
      {
         tag=tags[sfc];
         saddr=npr[ifc];
         raddr=saddr;

         MPI_Send_init(sbufs[sfc&0x1],12*nfc[ifc],MPI_DOUBLE,saddr,
                       tag,MPI_COMM_WORLD,&snd_req[ifc]);
         MPI_Recv_init(rbufs[sfc&0x1],12*nfc[ifc],MPI_DOUBLE,raddr,
                       tag,MPI_COMM_WORLD,&rcv_req[ifc]);
      }
      else
      {
         snd_req[ifc]=MPI_REQUEST_NULL;
         rcv_req[ifc]=MPI_REQUEST_NULL;
      }
   }
}


static void set_bnd_sd2zero(int ofs,int vol,spinor_dble *sd)
{
   int *pt,*ptm;

   pt=pts+ofs;
   ptm=pt+vol;

   for (;pt<ptm;pt++)
      sd[*pt]=sd0;
}

#if (defined x64)
#include "sse2.h"

static void zip_weyl(int vol,spinor_dble *pk,weyl_dble *pl)
{
   weyl_dble *pm;

   pm=pl+vol;

   for (;pl<pm;pl++)
   {
      __asm__ __volatile__ ("movapd %0, %%xmm0 \n\t"
                            "movapd %1, %%xmm1 \n\t"
                            "movapd %2, %%xmm2 \n\t"
                            "movapd %3, %%xmm3 \n\t"
                            "movapd %4, %%xmm4 \n\t"
                            "movapd %5, %%xmm5"
                            :
                            :
                            "m" ((*pk).c1.c1),
                            "m" ((*pk).c1.c2),
                            "m" ((*pk).c1.c3),
                            "m" ((*pk).c2.c1),
                            "m" ((*pk).c2.c2),
                            "m" ((*pk).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      __asm__ __volatile__ ("movapd %%xmm0, %0 \n\t"
                            "movapd %%xmm1, %1 \n\t"
                            "movapd %%xmm2, %2 \n\t"
                            "movapd %%xmm3, %3 \n\t"
                            "movapd %%xmm4, %4 \n\t"
                            "movapd %%xmm5, %5"
                            :
                            "=m" ((*pl).c1.c1),
                            "=m" ((*pl).c1.c2),
                            "=m" ((*pl).c1.c3),
                            "=m" ((*pl).c2.c1),
                            "=m" ((*pl).c2.c2),
                            "=m" ((*pl).c2.c3));

      pk+=1;
   }
}


static void unzip_weyl(int vol,weyl_dble *pk,spinor_dble *pl)
{
   spinor_dble *pm;

   __asm__ __volatile__ ("xorpd %%xmm6, %%xmm6 \n\t"
                         "xorpd %%xmm7, %%xmm7"
                         :
                         :
                         :
                         "xmm6", "xmm7");

   pm=pl+vol;

   for (;pl<pm;pl++)
   {
      __asm__ __volatile__ ("movapd %0, %%xmm0 \n\t"
                            "movapd %1, %%xmm1 \n\t"
                            "movapd %2, %%xmm2 \n\t"
                            "movapd %3, %%xmm3 \n\t"
                            "movapd %4, %%xmm4 \n\t"
                            "movapd %5, %%xmm5"
                            :
                            :
                            "m" ((*pk).c1.c1),
                            "m" ((*pk).c1.c2),
                            "m" ((*pk).c1.c3),
                            "m" ((*pk).c2.c1),
                            "m" ((*pk).c2.c2),
                            "m" ((*pk).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      __asm__ __volatile__ ("addpd %%xmm0, %%xmm0 \n\t"
                            "addpd %%xmm1, %%xmm1 \n\t"
                            "addpd %%xmm2, %%xmm2 \n\t"
                            "addpd %%xmm3, %%xmm3 \n\t"
                            "addpd %%xmm4, %%xmm4 \n\t"
                            "addpd %%xmm5, %%xmm5"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      __asm__ __volatile__ ("movapd %%xmm0, %0 \n\t"
                            "movapd %%xmm1, %1 \n\t"
                            "movapd %%xmm2, %2 \n\t"
                            "movapd %%xmm3, %3 \n\t"
                            "movapd %%xmm4, %4 \n\t"
                            "movapd %%xmm5, %5"
                            :
                            "=m" ((*pl).c1.c1),
                            "=m" ((*pl).c1.c2),
                            "=m" ((*pl).c1.c3),
                            "=m" ((*pl).c2.c1),
                            "=m" ((*pl).c2.c2),
                            "=m" ((*pl).c2.c3));

      __asm__ __volatile__ ("movapd %%xmm6, %0 \n\t"
                            "movapd %%xmm7, %1 \n\t"
                            "movapd %%xmm6, %2 \n\t"
                            "movapd %%xmm7, %3 \n\t"
                            "movapd %%xmm6, %4 \n\t"
                            "movapd %%xmm7, %5"
                            :
                            "=m" ((*pl).c3.c1),
                            "=m" ((*pl).c3.c2),
                            "=m" ((*pl).c3.c3),
                            "=m" ((*pl).c4.c1),
                            "=m" ((*pl).c4.c2),
                            "=m" ((*pl).c4.c3));

      pk+=1;
   }
}

#else

static const weyl_dble w0={{{0.0}}};


static void zip_weyl(int vol,spinor_dble *pk,weyl_dble *pl)
{
   weyl_dble *pm;

   pm=pl+vol;

   for (;pl<pm;pl++)
   {
      (*pl).c1=(*pk).c1;
      (*pl).c2=(*pk).c2;

      pk+=1;
   }
}


static void unzip_weyl(int vol,weyl_dble *pk,spinor_dble *pl)
{
   weyl_dble *pm;

   pm=pk+vol;

   for (;pk<pm;pk++)
   {
      _vector_add((*pl).c1,(*pk).c1,(*pk).c1);
      _vector_add((*pl).c2,(*pk).c2,(*pk).c2);
      (*pl).c3=w0.c1;
      (*pl).c4=w0.c2;

      pl+=1;
   }
}

#endif

static void send_bufs(int ifc)
{
   int sfc;
   MPI_Status stat_snd,stat_rcv;

   sfc=(ifc^nmu[ifc]);

   if (sflg[sfc])
   {
      if (sfc&0x1)
      {
         MPI_Start(&snd_req[sfc]);
         MPI_Start(&rcv_req[sfc]);

         MPI_Wait(&snd_req[sfc],&stat_snd);
         MPI_Wait(&rcv_req[sfc],&stat_rcv);
      }
      else
      {
         MPI_Start(&rcv_req[sfc]);
         MPI_Start(&snd_req[sfc]);

         MPI_Wait(&rcv_req[sfc],&stat_rcv);
         MPI_Wait(&snd_req[sfc],&stat_snd);
      }
   }
}


void cpsd_int_bnd(int is,spinor_dble *sd)
{
   int k,ifc,sfc,ofs,vol;
   spinor_dble *sdb;

   if (NPROC>1)
   {
      if (ofs_put==NULL)
      {
         set_ofs();
         alloc_sbufs();
         set_requests();
      }

      is&=0x1;
      sdb=sd+VOLUME;

#pragma omp parallel private(k,ifc,sfc,ofs,vol)
      {
         k=omp_get_thread_num();

         ifc=0;
         sfc=ifc^nmu[0];

         if (sflg[sfc])
         {
            ofs=ofs_get[k][0][0];
            vol=ofs_get[k][0][1];
            assign_sd2wd[sfc^is](map+pofs[sfc^0x1]+ofs,vol,sd,
                                 sbufs[ifc&0x1]+ofs);
         }
         else
         {
            ofs=ofs_pts[k][0];
            vol=ofs_pts[k][1];
            set_bnd_sd2zero(ofs,vol,sd);
         }

         for (ifc=1;ifc<=8;ifc++)
         {
#pragma omp barrier
            if (k==0)
               send_bufs(ifc-1);

            if ((NTHREAD==1)||(k>0))
            {
               if (ifc<8)
               {
                  sfc=ifc^nmu[ifc];

                  if (sflg[sfc])
                  {
                     ofs=ofs_get[k][ifc][0];
                     vol=ofs_get[k][ifc][1];
                     assign_sd2wd[sfc^is](map+pofs[sfc^0x1]+ofs,vol,sd,
                                          sbufs[ifc&0x1]+ofs);
                  }
               }

               if (ifc>1)
               {
                  sfc=(ifc-2)^nmu[ifc-2];

                  if (sflg[sfc])
                  {
                     ofs=ofs_put[k][ifc-2][0];
                     vol=ofs_put[k][ifc-2][1];
                     unzip_weyl(vol,rbufs[ifc&0x1]+ofs,sdb+pofs[sfc]+ofs);
                  }

                  if ((ifc==2)&&(sflg[0]==1)&&(sflg[1]==0)&&(bc!=0))
                  {
                     ofs=ofs_get[k][1][0];
                     vol=ofs_get[k][1][1];
                     set_sd2zero(vol,0,sdb+pofs[1]+ofs);
                  }
               }
            }
         }
#pragma omp barrier
         sfc=7^nmu[7];

         if (sflg[sfc])
         {
            ofs=ofs_put[k][7][0];
            vol=ofs_put[k][7][1];
            unzip_weyl(vol,rbufs[1]+ofs,sdb+pofs[sfc]+ofs);
         }
      }
   }
   else
      bnd_sd2zero(EVEN_PTS,sd);
}


void cpsd_ext_bnd(int is,spinor_dble *sd)
{
   int k,ifc,sfc,ofs,vol;
   spinor_dble *sdb;

   if (NPROC>1)
   {
      if (ofs_put==NULL)
      {
         set_ofs();
         alloc_sbufs();
         set_requests();
      }

      is&=0x1;
      sdb=sd+VOLUME;

#pragma omp parallel private(k,ifc,sfc,ofs,vol)
      {
         k=omp_get_thread_num();

         ifc=7;
         sfc=ifc^nmu[7];

         if (sflg[sfc])
         {
            ofs=ofs_put[k][7][0];
            vol=ofs_put[k][7][1];
            zip_weyl(vol,sdb+pofs[sfc]+ofs,sbufs[ifc&0x1]+ofs);
         }

         for (ifc=6;ifc>=-1;ifc--)
         {
#pragma omp barrier
            if (k==0)
               send_bufs(ifc+1);

            if ((NTHREAD==1)||(k>0))
            {
               if (ifc>=0)
               {
                  sfc=ifc^nmu[ifc];

                  if (sflg[sfc])
                  {
                     ofs=ofs_put[k][ifc][0];
                     vol=ofs_put[k][ifc][1];
                     zip_weyl(vol,sdb+pofs[sfc]+ofs,sbufs[ifc&0x1]+ofs);
                  }
               }

               if (ifc<6)
               {
                  sfc=(ifc+2)^nmu[ifc+2];

                  if (sflg[sfc])
                  {
                     ofs=ofs_get[k][ifc+2][0];
                     vol=ofs_get[k][ifc+2][1];
                     add_assign_wd2sd[(sfc^0x1)^is](map+pofs[sfc^0x1]+ofs,vol,
                                                    rbufs[(ifc+2)&0x1]+ofs,sd);
                  }
               }
            }
         }
#pragma omp barrier
         sfc=0^nmu[0];

         if (sflg[sfc])
         {
            ofs=ofs_get[k][0][0];
            vol=ofs_get[k][0][1];
            add_assign_wd2sd[(sfc^0x1)^is](map+pofs[sfc^0x1]+ofs,vol,
                                           rbufs[0]+ofs,sd);
         }
         else
         {
            ofs=ofs_pts[k][0];
            vol=ofs_pts[k][1];
            set_bnd_sd2zero(ofs,vol,sd);
         }
      }
   }
   else
      bnd_sd2zero(EVEN_PTS,sd);
}
