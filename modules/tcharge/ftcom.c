
/*******************************************************************************
*
* File ftcom.c
*
* Copyright (C) 2011, 2013, 2016, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Communication of the field tensor components residing at the boundaries
* of the local lattices.
*
*   void copy_bnd_ft(int n,u3_alg_dble *ft)
*     Fetches the boundary values the field ft from the neighbouring MPI
*     processes (see the notes). The boundary values at time NPROC0*L0
*     are fetched from the field at time 0 only in the case of periodic
*     boundary conditions.
*
*   void add_bnd_ft(int n,u3_alg_dble *ft)
*     Adds the boundary values of the field ft to the field on the
*     neighbouring MPI processes. The boundary values at time NPROC0*L0
*     are added to the field at time 0 only in the case of periodic
*     boundary conditions.
*
* Both communication programs assume that the field ft has the same size as
* the n-th component of the symmetric field tensor F_{mu nu}, where n=0,..,5
* labels the (mu,nu)-planes (0,1),(0,2),(0,3),(2,3),(3,1),(1,2). For further
* explanations, see the files lattice/README.ftidx and tcharge/ftensor.c.
*
* The programs in this module act globally and must be called simultaneously
* by the OpenMP master thread on all MPI processes.
*
*******************************************************************************/

#define FTCOM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "tcharge.h"
#include "global.h"

static const int plns[6][2]={{0,1},{0,2},{0,3},{2,3},{3,1},{1,2}};
static int tags[4],sflg[4],rflg[4],(*ofs_ft)[6][2][2];
static u3_alg_dble *ftbuf;
static ftidx_t *idx=NULL;


static void alloc_ftbuf(void)
{
   int n,dir,nft,nbf;
   int bc,mu;
   int k,*a,*b;

   set_ftidx();
   idx=ftidx();
   nbf=0;

   for (n=0;n<6;n++)
   {
      nft=idx[n].nft[0];
      if (nft>nbf)
         nbf=nft;

      nft=idx[n].nft[1];
      if (nft>nbf)
         nbf=nft;
   }

   ftbuf=amalloc(nbf*sizeof(*ftbuf),ALIGN);
   ofs_ft=malloc(NTHREAD*sizeof(*ofs_ft));
   a=malloc(2*NTHREAD*sizeof(*a));
   b=a+NTHREAD;
   error((ftbuf==NULL)||(ofs_ft==NULL)||(a==NULL),1,"alloc_ftbuf [ftcom.c]",
         "Unable to allocate communication buffers");

   bc=bc_type();

   for (mu=0;mu<4;mu++)
   {
      tags[mu]=mpi_permanent_tag();
      sflg[mu]=((mu>0)||(cpr[0]>0)||(bc==3));
      rflg[mu]=((mu>0)||(cpr[0]<(NPROC0-1))||(bc==3));
   }

   for (n=0;n<6;n++)
   {
      for (dir=0;dir<2;dir++)
      {
         divide_range(idx[n].nft[dir],NTHREAD,a,b);

         for (k=0;k<NTHREAD;k++)
         {
            ofs_ft[k][n][dir][0]=a[k];
            ofs_ft[k][n][dir][1]=b[k]-a[k];
         }
      }
   }

   free(a);
}


static void pack_buf(int n,int dir,u3_alg_dble *ft)
{
   int mu,k,ofs,vol;
   int *ift,*ifm;
   u3_alg_dble *fb;

   if (idx[n].nft[dir]>0)
   {
      mu=plns[n][dir];

      if (sflg[mu])
      {
#pragma omp parallel private(k,ofs,vol,ift,ifm,fb)
         {
            k=omp_get_thread_num();
            ofs=ofs_ft[k][n][dir][0];
            vol=ofs_ft[k][n][dir][1];

            ift=idx[n].ift[dir]+ofs;
            ifm=ift+vol;
            fb=ftbuf+ofs;

            for (;ift<ifm;ift++)
            {
               fb[0]=ft[*ift];
               fb+=1;
            }
         }
      }
   }
}


static void fwd_send(int n,int dir,u3_alg_dble *ft)
{
   int mu,nft,nbf;
   int tag,saddr,raddr,np;
   u3_alg_dble *sbuf,*rbuf;
   MPI_Status snd_stat,rcv_stat;
   MPI_Request snd_req,rcv_req;

   np=(cpr[0]+cpr[1]+cpr[2]+cpr[3])&0x1;
   nft=idx[n].nft[dir];

   if (nft>0)
   {
      mu=plns[n][dir];
      tag=tags[mu];
      saddr=npr[2*mu];
      raddr=npr[2*mu+1];
      sbuf=ftbuf;
      rbuf=ft+VOLUME;
      if (dir==1)
         rbuf+=idx[n].nft[0];
      nbf=9*nft;

      if (np==0)
      {
         if (sflg[mu])
            MPI_Isend(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD,&snd_req);
         if (rflg[mu])
            MPI_Irecv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&rcv_req);

         if (sflg[mu])
            MPI_Wait(&snd_req,&snd_stat);
         if (rflg[mu])
            MPI_Wait(&rcv_req,&rcv_stat);
      }
      else
      {
         if (rflg[mu])
            MPI_Irecv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&rcv_req);
         if (sflg[mu])
            MPI_Isend(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD,&snd_req);

         if (rflg[mu])
            MPI_Wait(&rcv_req,&rcv_stat);
         if (sflg[mu])
            MPI_Wait(&snd_req,&snd_stat);
      }
   }
}


void copy_bnd_ft(int n,u3_alg_dble *ft)
{
   if (NPROC>1)
   {
      if (idx==NULL)
         alloc_ftbuf();

      pack_buf(n,1,ft);
      fwd_send(n,1,ft);
      pack_buf(n,0,ft);
      fwd_send(n,0,ft);
   }
}


static void bck_send(int n,int dir,u3_alg_dble *ft)
{
   int mu,nft,nbf;
   int tag,saddr,raddr,np;
   u3_alg_dble *sbuf,*rbuf;
   MPI_Status snd_stat,rcv_stat;
   MPI_Request snd_req,rcv_req;

   np=(cpr[0]+cpr[1]+cpr[2]+cpr[3])&0x1;
   nft=idx[n].nft[dir];

   if (nft>0)
   {
      mu=plns[n][dir];
      tag=tags[mu];
      saddr=npr[2*mu+1];
      raddr=npr[2*mu];
      sbuf=ft+VOLUME;
      if (dir==1)
         sbuf+=idx[n].nft[0];
      rbuf=ftbuf;
      nbf=9*nft;

      if (np==0)
      {
         if (rflg[mu])
            MPI_Isend(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD,&snd_req);
         if (sflg[mu])
            MPI_Irecv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&rcv_req);

         if (rflg[mu])
            MPI_Wait(&snd_req,&snd_stat);
         if (sflg[mu])
            MPI_Wait(&rcv_req,&rcv_stat);
      }
      else
      {
         if (sflg[mu])
            MPI_Irecv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&rcv_req);
         if (rflg[mu])
            MPI_Isend(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD,&snd_req);

         if (sflg[mu])
            MPI_Wait(&rcv_req,&rcv_stat);
         if (rflg[mu])
            MPI_Wait(&snd_req,&snd_stat);
      }
   }
}


static void unpack_buf(int n,int dir,u3_alg_dble *ft)
{
   int mu,k,ofs,vol;
   int *ift,*ifm;
   u3_alg_dble *fb;

   if (idx[n].nft[dir]>0)
   {
      mu=plns[n][dir];

      if (sflg[mu])
      {
#pragma omp parallel private(k,ofs,vol,ift,ifm,fb)
         {
            k=omp_get_thread_num();
            ofs=ofs_ft[k][n][dir][0];
            vol=ofs_ft[k][n][dir][1];

            ift=idx[n].ift[dir]+ofs;
            ifm=ift+vol;
            fb=ftbuf+ofs;

            for (;ift<ifm;ift++)
            {
               _u3_alg_add_assign(ft[*ift],fb[0]);
               fb+=1;
            }
         }
      }
   }
}


void add_bnd_ft(int n,u3_alg_dble *ft)
{
   if (NPROC>1)
   {
      if (idx==NULL)
         alloc_ftbuf();

      bck_send(n,0,ft);
      unpack_buf(n,0,ft);
      bck_send(n,1,ft);
      unpack_buf(n,1,ft);
   }
}
