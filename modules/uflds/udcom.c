
/*******************************************************************************
*
* File udcom.c
*
* Copyright (C) 2005, 2009-2013, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Communication of the double-precision link variables residing at the
* exterior boundaries of the local lattices.
*
*   void copy_bnd_ud(void)
*     Copies the double-precision link variables from the neighbouring MPI
*     processes to the exterior boundaries of the local lattice. The field
*     variables on the spatial links at time NPROC0*L0 are fetched only in
*     the case of periodic boundary conditions.
*
* After calling copy_bnd_ud(), the double-precision link variables at the
* +0,+1,+2,+3 faces have the correct values (see main/README.global and
* lattice/README.uidx). Whether they are up-to-date can always be checked
* by querying the flags data base (see flags/flags.c).
*
* The program copy_bnd_ud() is assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define UDCOM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "lattice.h"
#include "uflds.h"
#include "global.h"

static int bc,np,tags[4];
static int (*ofs_u0)[4][2],(*ofs_uk)[4][2];
static su3_dble *sbufs[2];
static uidx_t *idx=NULL;
static MPI_Request snd_u0_req[4],rcv_u0_req[4];
static MPI_Request snd_uk_req[4],rcv_uk_req[4];


static void set_ofs(void)
{
   int mu,k,*a,*b;

   bc=bc_type();
   np=(cpr[0]+cpr[1]+cpr[2]+cpr[3])&0x1;

   set_uidx();
   idx=uidx();

   for (mu=0;mu<4;mu++)
      tags[mu]=mpi_permanent_tag();

   a=malloc(2*NTHREAD*sizeof(*a));
   ofs_u0=malloc(2*NTHREAD*sizeof(*ofs_u0));
   error((a==NULL)||(ofs_u0==NULL),1,"set_ofs [udcom.c]",
         "Unable to allocate auxiliary arrays");
   b=a+NTHREAD;
   ofs_uk=ofs_u0+NTHREAD;

   for (mu=0;mu<4;mu++)
   {
      if (mu==0)
         divide_range(idx[mu].nu0,NTHREAD,a,b);
      else
      {
         a[0]=0;
         b[0]=idx[mu].nu0;
         divide_range(idx[mu].nu0,NTHREAD-1,a+1,b+1);
      }

      for (k=0;k<NTHREAD;k++)
      {
         ofs_u0[k][mu][0]=a[k];
         ofs_u0[k][mu][1]=b[k]-a[k];
      }

      if (mu==0)
         divide_range(idx[mu].nuk,NTHREAD,a,b);
      else
      {
         a[0]=0;
         b[0]=idx[mu].nuk;
         divide_range(idx[mu].nuk,NTHREAD-1,a+1,b+1);
      }

      for (k=0;k<NTHREAD;k++)
      {
         ofs_uk[k][mu][0]=a[k];
         ofs_uk[k][mu][1]=b[k]-a[k];
      }
   }

   free(a);
}


static void alloc_sbufs(void)
{
   int mu,nuk,n[2];

   n[0]=0;
   n[1]=0;

   for (mu=0;mu<4;mu++)
   {
      nuk=idx[mu].nuk;

      if (nuk>n[mu&0x1])
         n[mu&0x1]=nuk;
   }

   sbufs[0]=amalloc((n[0]+n[1])*sizeof(*sbufs[0]),ALIGN);
   error(sbufs[0]==NULL,1,"alloc_sbufs [udcom.c]",
         "Unable to allocate send buffers");
   sbufs[1]=sbufs[0]+n[0];
}


static void set_requests(void)
{
   int mu,nbf,saddr,raddr,tag;
   su3_dble *sbuf,*rbuf;

   rbuf=udfld()+4*VOLUME;

   for (mu=0;mu<4;mu++)
   {
      nbf=18*idx[mu].nu0;

      if (nbf)
      {
         sbuf=sbufs[mu&0x1];
         saddr=npr[2*mu];
         raddr=npr[2*mu+1];
         tag=tags[mu];

         MPI_Send_init(sbuf,nbf,MPI_DOUBLE,saddr,
                       tag,MPI_COMM_WORLD,&snd_u0_req[mu]);
         MPI_Recv_init(rbuf,nbf,MPI_DOUBLE,raddr,
                       tag,MPI_COMM_WORLD,&rcv_u0_req[mu]);
      }
      else
      {
         snd_u0_req[mu]=MPI_REQUEST_NULL;
         rcv_u0_req[mu]=MPI_REQUEST_NULL;
      }

      rbuf+=idx[mu].nu0;
   }

   for (mu=0;mu<4;mu++)
   {
      nbf=18*idx[mu].nuk;

      if (nbf)
      {
         sbuf=sbufs[mu&0x1];
         saddr=npr[2*mu];
         raddr=npr[2*mu+1];
         tag=tags[mu];

         MPI_Send_init(sbuf,nbf,MPI_DOUBLE,saddr,
                       tag,MPI_COMM_WORLD,&snd_uk_req[mu]);
         MPI_Recv_init(rbuf,nbf,MPI_DOUBLE,raddr,
                       tag,MPI_COMM_WORLD,&rcv_uk_req[mu]);
      }
      else
      {
         snd_uk_req[mu]=MPI_REQUEST_NULL;
         rcv_uk_req[mu]=MPI_REQUEST_NULL;
      }

      rbuf+=idx[mu].nuk;
   }
}


static void pack_ud0(int k,int mu)
{
   int ofs,vol;
   int *iu,*ium;
   su3_dble *ud,*udb;

   ofs=ofs_u0[k][mu][0];
   vol=ofs_u0[k][mu][1];

   if (vol>0)
   {
      udb=udfld();
      ud=sbufs[mu&0x1]+ofs;
      iu=idx[mu].iu0+ofs;
      ium=iu+vol;

      for (;iu<ium;iu++)
      {
         (*ud)=udb[*iu];
         ud+=1;
      }
   }
}


static void pack_udk(int k,int mu)
{
   int ofs,vol;
   int *iu,*ium;
   su3_dble *ud,*udb;

   ofs=ofs_uk[k][mu][0];
   vol=ofs_uk[k][mu][1];

   if ((vol>0)&&((mu>0)||(cpr[0]>0)||(bc==3)))
   {
      udb=udfld();
      ud=sbufs[mu&0x1]+ofs;
      iu=idx[mu].iuk+ofs;
      ium=iu+vol;

      for (;iu<ium;iu++)
      {
         (*ud)=udb[*iu];
         ud+=1;
      }
   }
}


static void send_ud0(int mu)
{
   MPI_Status stat_snd,stat_rcv;

   if (idx[mu].nu0>0)
   {
      if (np==0)
      {
         MPI_Start(&snd_u0_req[mu]);
         MPI_Start(&rcv_u0_req[mu]);

         MPI_Wait(&snd_u0_req[mu],&stat_snd);
         MPI_Wait(&rcv_u0_req[mu],&stat_rcv);
      }
      else
      {
         MPI_Start(&rcv_u0_req[mu]);
         MPI_Start(&snd_u0_req[mu]);

         MPI_Wait(&rcv_u0_req[mu],&stat_rcv);
         MPI_Wait(&snd_u0_req[mu],&stat_snd);
      }
   }
}


static void send_udk(int mu)
{
   int sflg[2];
   MPI_Status stat_snd,stat_rcv;

   sflg[0]=((mu>0)||(cpr[0]>0)||(bc==3));
   sflg[1]=((mu>0)||(cpr[0]<(NPROC0-1))||(bc==3));

   if (idx[mu].nuk>0)
   {
      if (np==0)
      {
         if (sflg[0])
            MPI_Start(&snd_uk_req[mu]);
         if (sflg[1])
            MPI_Start(&rcv_uk_req[mu]);

         if (sflg[0])
            MPI_Wait(&snd_uk_req[mu],&stat_snd);
         if (sflg[1])
            MPI_Wait(&rcv_uk_req[mu],&stat_rcv);
      }
      else
      {
         if (sflg[1])
            MPI_Start(&rcv_uk_req[mu]);
         if (sflg[0])
            MPI_Start(&snd_uk_req[mu]);

         if (sflg[1])
            MPI_Wait(&rcv_uk_req[mu],&stat_rcv);
         if (sflg[0])
            MPI_Wait(&snd_uk_req[mu],&stat_snd);
      }
   }
}


void copy_bnd_ud(void)
{
   int k,mu;

   if (NPROC>1)
   {
      if (idx==NULL)
      {
         set_ofs();
         alloc_sbufs();
         set_requests();
      }

      if (NTHREAD==1)
      {
         for (mu=0;mu<4;mu++)
         {
            pack_ud0(0,mu);
            send_ud0(mu);
         }

         for (mu=0;mu<4;mu++)
         {
            pack_udk(0,mu);
            send_udk(mu);
         }
      }
      else
      {
#pragma omp parallel private(k,mu)
         {
            k=omp_get_thread_num();

            pack_ud0(k,0);

            for (mu=1;mu<4;mu++)
            {
#pragma omp barrier
               if (k>0)
                  pack_ud0(k,mu);
               else
                  send_ud0(mu-1);
            }
#pragma omp barrier
            if (k==0)
               send_ud0(3);

#pragma omp barrier
            pack_udk(k,0);

            for (mu=1;mu<4;mu++)
            {
#pragma omp barrier
               if (k>0)
                  pack_udk(k,mu);
               else
                  send_udk(mu-1);
            }
         }

         send_udk(3);
      }
   }

   set_flags(COPIED_BND_UD);
}
