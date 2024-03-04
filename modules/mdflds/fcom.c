
/*******************************************************************************
*
* File fcom.c
*
* Copyright (C) 2010, 2011, 2013, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Communication of the force variables residing at the exterior boundaries
* of the local lattices.
*
*   void copy_bnd_frc(void)
*     Copies the force variables from the neighbouring MPI processes to
*     the exterior boundaries of the local lattice. The field variables
*     on the spatial links at time NPROC0*L0 are fetched only in the case
*     of periodic boundary conditions.
*
*   void add_bnd_frc(void)
*     Adds the force variables on the exterior boundaries of the local
*     lattice to the field variables on the neighbouring MPI processes.
*     The field variables on the spatial links at time NPROC0*L0 are
*     added only in the case of periodic boundary conditions.
*
* The force field is the one returned by mdflds(). Its elements are ordered
* in the same way as those of the global gauge fields (see main/README.global
* and lattice/README.uidx).
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define FCOM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "mdflds.h"
#include "global.h"

static int bc,np;
static int tags[4],sflg[4],rflg[4],(*ofs_f0)[4][2],(*ofs_fk)[4][2];
static su3_alg_dble *sbuf,*rbuf_f0[4],*rbuf_fk[4],*frcfld;
static uidx_t *idx=NULL;


static void set_ofs(void)
{
   int mu,k,*a,*b;

   error(ipt==NULL,1,"set_ofs [fcom.c]",
         "The geometry arrays are not set");

   bc=bc_type();
   np=(cpr[0]+cpr[1]+cpr[2]+cpr[3])&0x1;

   for (mu=0;mu<4;mu++)
   {
      tags[mu]=mpi_permanent_tag();
      sflg[mu]=((mu>0)||(cpr[0]>0)||(bc==3));
      rflg[mu]=((mu>0)||(cpr[0]<(NPROC0-1))||(bc==3));
   }

   a=malloc(2*NTHREAD*sizeof(*a));
   ofs_f0=malloc(2*NTHREAD*sizeof(*ofs_f0));
   error((a==NULL)||(ofs_f0==NULL),1,"set_ofs [fcom.c]",
         "Unable to allocate auxiliary arrays");
   b=a+NTHREAD;
   ofs_fk=ofs_f0+NTHREAD;

   set_uidx();
   idx=uidx();

   for (mu=0;mu<4;mu++)
   {
      divide_range(idx[mu].nu0,NTHREAD,a,b);

      for (k=0;k<NTHREAD;k++)
      {
         ofs_f0[k][mu][0]=a[k];
         ofs_f0[k][mu][1]=b[k]-a[k];
      }

      divide_range(idx[mu].nuk,NTHREAD,a,b);

      for (k=0;k<NTHREAD;k++)
      {
         ofs_fk[k][mu][0]=a[k];
         ofs_fk[k][mu][1]=b[k]-a[k];
      }
   }

   free(a);
}


static void alloc_sbuf(void)
{
   int mu,nbf;
   mdflds_t *mdfs;

   set_ofs();
   nbf=0;

   for (mu=0;mu<4;mu++)
   {
      if (nbf<idx[mu].nuk)
         nbf=idx[mu].nuk;
   }

   sbuf=amalloc(nbf*sizeof(*sbuf),ALIGN);
   error(sbuf==NULL,1,"alloc_sbufs [fcom.c]",
         "Unable to allocate communication buffers");

   mdfs=mdflds();
   frcfld=(*mdfs).frc;
   rbuf_f0[0]=frcfld+4*VOLUME;

   for (mu=1;mu<4;mu++)
      rbuf_f0[mu]=rbuf_f0[mu-1]+idx[mu-1].nu0;

   rbuf_fk[0]=rbuf_f0[3]+idx[3].nu0;

   for (mu=1;mu<4;mu++)
      rbuf_fk[mu]=rbuf_fk[mu-1]+idx[mu-1].nuk;
}


static void pack_f0(int mu)
{
   int k,ofs,vol;
   int *iu,*ium;
   su3_alg_dble *fb;

   if (idx[mu].nu0>0)
   {
#pragma omp parallel private(k,ofs,vol,iu,ium,fb)
      {
         k=omp_get_thread_num();

         ofs=ofs_f0[k][mu][0];
         vol=ofs_f0[k][mu][1];

         fb=sbuf+ofs;
         iu=idx[mu].iu0+ofs;
         ium=iu+vol;

         for (;iu<ium;iu++)
         {
            fb[0]=frcfld[*iu];
            fb+=1;
         }
      }
   }
}


static void pack_fk(int mu)
{
   int k,ofs,vol;
   int *iu,*ium;
   su3_alg_dble *fb;

   if ((idx[mu].nuk>0)&&(sflg[mu]))
   {
#pragma omp parallel private(k,ofs,vol,iu,ium,fb)
      {
         k=omp_get_thread_num();

         ofs=ofs_fk[k][mu][0];
         vol=ofs_fk[k][mu][1];

         fb=sbuf+ofs;
         iu=idx[mu].iuk+ofs;
         ium=iu+vol;

         for (;iu<ium;iu++)
         {
            fb[0]=frcfld[*iu];
            fb+=1;
         }
      }
   }
}


static void fwd_send_f0(int mu)
{
   int nbf,tag,saddr,raddr;
   su3_alg_dble *rbuf;
   MPI_Status snd_stat,rcv_stat;
   MPI_Request snd_req,rcv_req;

   if (idx[mu].nu0>0)
   {
      nbf=8*idx[mu].nu0;
      rbuf=rbuf_f0[mu];
      saddr=npr[2*mu];
      raddr=npr[2*mu+1];
      tag=tags[mu];

      if (np==0)
      {
         MPI_Isend(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD,&snd_req);
         MPI_Irecv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&rcv_req);

         MPI_Wait(&snd_req,&snd_stat);
         MPI_Wait(&rcv_req,&rcv_stat);
      }
      else
      {
         MPI_Irecv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&rcv_req);
         MPI_Isend(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD,&snd_req);

         MPI_Wait(&rcv_req,&rcv_stat);
         MPI_Wait(&snd_req,&snd_stat);
      }
   }
}


static void fwd_send_fk(int mu)
{
   int nbf,tag,saddr,raddr;
   su3_alg_dble *rbuf;
   MPI_Status snd_stat,rcv_stat;
   MPI_Request snd_req,rcv_req;

   if ((idx[mu].nuk>0)&&((sflg[mu])||(rflg[mu])))
   {
      nbf=8*idx[mu].nuk;
      rbuf=rbuf_fk[mu];
      saddr=npr[2*mu];
      raddr=npr[2*mu+1];
      tag=tags[mu];

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


void copy_bnd_frc(void)
{
   int mu;

   if (NPROC>1)
   {
      if (idx==NULL)
         alloc_sbuf();

      for (mu=0;mu<4;mu++)
      {
         pack_f0(mu);
         fwd_send_f0(mu);
      }

      for (mu=0;mu<4;mu++)
      {
         pack_fk(mu);
         fwd_send_fk(mu);
      }
   }
}


static void bck_send_f0(int mu)
{
   int nbf,tag,saddr,raddr;
   su3_alg_dble *rbuf;
   MPI_Status snd_stat,rcv_stat;
   MPI_Request snd_req,rcv_req;

   if (idx[mu].nu0>0)
   {
      nbf=8*idx[mu].nu0;
      rbuf=rbuf_f0[mu];
      saddr=npr[2*mu+1];
      raddr=npr[2*mu];
      tag=tags[mu];

      if (np==0)
      {
         MPI_Isend(rbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD,&snd_req);
         MPI_Irecv(sbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&rcv_req);

         MPI_Wait(&snd_req,&snd_stat);
         MPI_Wait(&rcv_req,&rcv_stat);
      }
      else
      {
         MPI_Irecv(sbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&rcv_req);
         MPI_Isend(rbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD,&snd_req);

         MPI_Wait(&rcv_req,&rcv_stat);
         MPI_Wait(&snd_req,&snd_stat);
      }
   }
}


static void bck_send_fk(int mu)
{
   int nbf,tag,saddr,raddr;
   su3_alg_dble *rbuf;
   MPI_Status snd_stat,rcv_stat;
   MPI_Request snd_req,rcv_req;

   if ((idx[mu].nuk>0)&&((sflg[mu])||(rflg[mu])))
   {
      nbf=8*idx[mu].nuk;
      rbuf=rbuf_fk[mu];
      saddr=npr[2*mu+1];
      raddr=npr[2*mu];
      tag=tags[mu];

      if (np==0)
      {
         if (rflg[mu])
            MPI_Isend(rbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD,&snd_req);
         if (sflg[mu])
            MPI_Irecv(sbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&rcv_req);

         if (rflg[mu])
            MPI_Wait(&snd_req,&snd_stat);
         if (sflg[mu])
            MPI_Wait(&rcv_req,&rcv_stat);
      }
      else
      {
         if (sflg[mu])
            MPI_Irecv(sbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&rcv_req);
         if (rflg[mu])
            MPI_Isend(rbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD,&snd_req);

         if (sflg[mu])
            MPI_Wait(&rcv_req,&rcv_stat);
         if (rflg[mu])
            MPI_Wait(&snd_req,&snd_stat);
      }
   }
}


static void add_f0(int mu)
{
   int k,ofs,vol;
   int *iu,*ium;
   su3_alg_dble *fb,*frc;

   if (idx[mu].nu0>0)
   {
#pragma omp parallel private(k,ofs,vol,iu,ium,fb,frc)
      {
         k=omp_get_thread_num();

         ofs=ofs_f0[k][mu][0];
         vol=ofs_f0[k][mu][1];

         fb=sbuf+ofs;
         iu=idx[mu].iu0+ofs;
         ium=iu+vol;

         for (;iu<ium;iu++)
         {
            frc=frcfld+iu[0];

            (*frc).c1+=(*fb).c1;
            (*frc).c2+=(*fb).c2;
            (*frc).c3+=(*fb).c3;
            (*frc).c4+=(*fb).c4;
            (*frc).c5+=(*fb).c5;
            (*frc).c6+=(*fb).c6;
            (*frc).c7+=(*fb).c7;
            (*frc).c8+=(*fb).c8;

            fb+=1;
         }
      }
   }
}


static void add_fk(int mu)
{
   int k,ofs,vol;
   int *iu,*ium;
   su3_alg_dble *fb,*frc;

   if ((idx[mu].nuk>0)&&(sflg[mu]))
   {
#pragma omp parallel private(k,ofs,vol,iu,ium,fb,frc)
      {
         k=omp_get_thread_num();

         ofs=ofs_fk[k][mu][0];
         vol=ofs_fk[k][mu][1];

         fb=sbuf+ofs;
         iu=idx[mu].iuk+ofs;
         ium=iu+vol;

         for (;iu<ium;iu++)
         {
            frc=frcfld+iu[0];

            (*frc).c1+=(*fb).c1;
            (*frc).c2+=(*fb).c2;
            (*frc).c3+=(*fb).c3;
            (*frc).c4+=(*fb).c4;
            (*frc).c5+=(*fb).c5;
            (*frc).c6+=(*fb).c6;
            (*frc).c7+=(*fb).c7;
            (*frc).c8+=(*fb).c8;

            fb+=1;
         }
      }
   }
}


void add_bnd_frc(void)
{
   int mu;

   if (NPROC>1)
   {
      if (idx==NULL)
         alloc_sbuf();

      for (mu=0;mu<4;mu++)
      {
         bck_send_fk(mu);
         add_fk(mu);
      }

      for (mu=0;mu<4;mu++)
      {
         bck_send_f0(mu);
         add_f0(mu);
      }
   }
}
