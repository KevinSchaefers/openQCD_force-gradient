
/*******************************************************************************
*
* File bstap.c
*
* Copyright (C) 2012-2016, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Allocation and computation of the boundary staple field.
*
*   void set_bstap(void)
*     Computes the boundary staples and copies them to the neighbouring
*     MPI processes (see doc/gauge_actions.pdf). If needed the boundary
*     staple field is allocated in the static memory of this module.
*
*   su3_dble *bstap(void)
*     Returns the address of the boundary staple field. A NULL pointer
*     pointer is returned if the field is not allocated.
*
* The boundary staple field has size 3*BNDRY and is logically divided into
* face segments. For the face with index ifc, the associated segment is
* at offset sofs[ifc] from the base address, where
*
*   sofs[0]=0
*   sofs[1]=sofs[0]+3*FACE0
*   sofs[2]=sofs[1]+3*FACE0
*   sofs[3]=sofs[2]+3*FACE1
*   sofs[4]=sofs[3]+3*FACE1
*   sofs[5]=sofs[4]+3*FACE2
*   sofs[6]=sofs[5]+3*FACE2
*   sofs[7]=sofs[6]+3*FACE3
*
* The ordering of the staples along the faces coincides with the ordering
* of the lattice points at the boundary (see main/README.global and also
* lattice/README.uidx for some further details). With open or SF boundary
* conditions, the program set_bstap() sets the time-like boundary staples
* at the boundaries of the lattice to zero.
*
* The program set_bstap() is assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously. It is taken for granted that
* the lattice geometry index arrays have been set up.
*
* The program bstap() is thread-safe and can be locally called.
*
*******************************************************************************/

#define BSTAP_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "global.h"

static const int plns[6][2]={{0,1},{0,2},{0,3},{2,3},{3,1},{1,2}};
static int bc,np,nfc[8],nmu[8],sflg[8],tags[8];
static int pofs[8],sofs[8],(*ofs_stap)[8][2];
static su3_dble *sbufs[2],*hdb=NULL;
static MPI_Request snd_req[8],rcv_req[8];


static void set_ofs(void)
{
   int ifc,k;
   int *a,*b;

   bc=bc_type();
   np=(cpr[0]+cpr[1]+cpr[2]+cpr[3])&0x1;

   nfc[0]=FACE0/2;
   nfc[1]=FACE0/2;
   nfc[2]=FACE1/2;
   nfc[3]=FACE1/2;
   nfc[4]=FACE2/2;
   nfc[5]=FACE2/2;
   nfc[6]=FACE3/2;
   nfc[7]=FACE3/2;

   pofs[0]=0;
   pofs[1]=pofs[0]+(FACE0/2);
   pofs[2]=pofs[1]+(FACE0/2);
   pofs[3]=pofs[2]+(FACE1/2);
   pofs[4]=pofs[3]+(FACE1/2);
   pofs[5]=pofs[4]+(FACE2/2);
   pofs[6]=pofs[5]+(FACE2/2);
   pofs[7]=pofs[6]+(FACE3/2);

   sofs[0]=0;
   sofs[1]=sofs[0]+3*FACE0;
   sofs[2]=sofs[1]+3*FACE0;
   sofs[3]=sofs[2]+3*FACE1;
   sofs[4]=sofs[3]+3*FACE1;
   sofs[5]=sofs[4]+3*FACE2;
   sofs[6]=sofs[5]+3*FACE2;
   sofs[7]=sofs[6]+3*FACE3;

   for (ifc=0;ifc<8;ifc++)
   {
      nmu[ifc]=cpr[ifc/2]&0x1;
      sflg[ifc]=((ifc>1)||(bc==3)||
                 ((ifc==1)&&(cpr[0]>0))||
                 ((ifc==0)&&(cpr[0]<(NPROC0-1))));
      tags[ifc]=mpi_permanent_tag();
   }

   a=malloc(2*NTHREAD*sizeof(*a));
   ofs_stap=malloc(NTHREAD*sizeof(*ofs_stap));
   error((a==NULL)||(ofs_stap==NULL),1,"set_ofs [bstap.c]",
         "Unable to allocate auxiliary arrays");
   b=a+NTHREAD;

   for (ifc=0;ifc<8;ifc++)
   {
      if (ifc==0)
         divide_range(2*nfc[ifc],NTHREAD,a,b);
      else
      {
         a[0]=0;
         b[0]=2*nfc[ifc];
         divide_range(2*nfc[ifc],NTHREAD-1,a+1,b+1);
      }

      for (k=0;k<NTHREAD;k++)
      {
         ofs_stap[k][ifc][0]=a[k];
         ofs_stap[k][ifc][1]=b[k]-a[k];
      }
   }

   free(a);
}


static void alloc_hdb(void)
{
   int ifc,n;

   n=0;

   for (ifc=0;ifc<8;ifc+=2)
   {
      if (n<nfc[ifc])
         n=nfc[ifc];
   }

   hdb=amalloc((3*BNDRY+12*n)*sizeof(*hdb),ALIGN);
   error(hdb==NULL,1,"alloc_hdb [bstap.c]",
         "Unable to allocate the boundary staple field");

   sbufs[0]=hdb+3*BNDRY;
   sbufs[1]=sbufs[0]+6*n;
}


static void set_requests(void)
{
   int ifc,sfc;
   int nbf,saddr,raddr,tag;
   su3_dble *sbuf,*rbuf;

   for (ifc=0;ifc<8;ifc++)
   {
      sfc=ifc^nmu[ifc];
      nbf=108*nfc[sfc];

      if (nbf>0)
      {
         sbuf=sbufs[ifc&0x1];
         rbuf=hdb+sofs[sfc^0x1];

         saddr=npr[sfc^0x1];
         raddr=saddr;
         tag=tags[ifc];

         MPI_Send_init(sbuf,nbf,MPI_DOUBLE,saddr,
                       tag,MPI_COMM_WORLD,&snd_req[ifc]);
         MPI_Recv_init(rbuf,nbf,MPI_DOUBLE,raddr,
                       tag,MPI_COMM_WORLD,&rcv_req[ifc]);
      }
      else
      {
         snd_req[ifc]=MPI_REQUEST_NULL;
         rcv_req[ifc]=MPI_REQUEST_NULL;
      }
   }
}


static void get_ofs(int mu,int nu,int ix,int *ip)
{
   int n,is;

   for (n=0;n<6;n++)
   {
      if (((plns[n][0]==mu)&&(plns[n][1]==nu))||
          ((plns[n][0]==nu)&&(plns[n][1]==mu)))
      {
         plaq_uidx(n,ix,ip);

         if (mu==plns[n][0])
         {
            is=ip[0];
            ip[0]=ip[2];
            ip[2]=is;

            is=ip[1];
            ip[1]=ip[3];
            ip[3]=is;
         }

         return;
      }
   }
}


static void get_staples(int k,int ifc)
{
   int ofs,vol;
   int sfc,ib,ix,mu,nu,l,ip[4];
   su3_dble *udb,*sbuf;
   su3_dble wd ALIGNED16;

   sfc=ifc^nmu[ifc];
   ofs=ofs_stap[k][ifc][0];
   vol=ofs_stap[k][ifc][1];

   if (vol>0)
   {
      udb=udfld();
      sbuf=sbufs[ifc&0x1]+3*ofs;
      mu=sfc/2;

      for (ib=ofs;ib<(ofs+vol);ib++)
      {
         if (ib<(nfc[sfc]))
            ix=map[pofs[sfc]+ib];
         else
            ix=map[(BNDRY/2)+pofs[sfc]+ib-nfc[sfc]];

         for (l=0;l<3;l++)
         {
            nu=l+(l>=mu);
            get_ofs(mu,nu,ix,ip);

            if (sfc&0x1)
            {
               if ((mu>0)||(cpr[0]>0)||(bc==3))
               {
                  su3xsu3dag(udb+ip[3],udb+ip[1],&wd);
                  su3xsu3(udb+ip[2],&wd,sbuf);
               }
            }
            else
            {
               if ((mu>0)||(cpr[0]<(NPROC0-1))||(bc==3))
               {
                  su3xsu3(udb+ip[0],udb+ip[1],&wd);
                  su3dagxsu3(udb+ip[2],&wd,sbuf);
               }
            }

            sbuf+=1;
         }
      }
   }
}


static void send_staples(int ifc)
{
   int sfc;
   MPI_Status stat_snd,stat_rcv;

   sfc=ifc^nmu[ifc];

   if (nfc[ifc]>0)
   {
      if (sflg[sfc])
      {
         if (np==0)
         {
            MPI_Start(&snd_req[ifc]);
            MPI_Start(&rcv_req[ifc]);

            MPI_Wait(&snd_req[ifc],&stat_snd);
            MPI_Wait(&rcv_req[ifc],&stat_rcv);
         }
         else
         {
            MPI_Start(&rcv_req[ifc]);
            MPI_Start(&snd_req[ifc]);

            MPI_Wait(&rcv_req[ifc],&stat_rcv);
            MPI_Wait(&snd_req[ifc],&stat_snd);
         }
      }
      else
         cm3x3_zero(3*FACE0,hdb+sofs[sfc^0x1]);
   }
}


void set_bstap(void)
{
   int k,ifc;

   if (NPROC>1)
   {
      if (query_flags(UDBUF_UP2DATE)!=1)
         copy_bnd_ud();

      if (hdb==NULL)
      {
         set_ofs();
         alloc_hdb();
         set_requests();
      }

      if (NTHREAD==1)
      {
         for (ifc=0;ifc<8;ifc++)
         {
            get_staples(0,ifc);
            send_staples(ifc);
         }
      }
      else
      {
#pragma omp parallel private(k,ifc)
         {
            k=omp_get_thread_num();
            get_staples(k,0);

            for (ifc=1;ifc<8;ifc++)
            {
#pragma omp barrier
               if (k>0)
                  get_staples(k,ifc);
               else
                  send_staples(ifc-1);
            }
         }

         send_staples(7);
      }
   }

   set_flags(SET_BSTAP);
}


su3_dble *bstap(void)
{
   return hdb;
}
