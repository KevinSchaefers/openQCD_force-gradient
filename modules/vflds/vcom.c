
/*******************************************************************************
*
* File vcom.c
*
* Copyright (C) 2007, 2011, 2013, 2018, 2021  Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Communication function for the global double-precision vector fields.
*
*   void cpv_int_bnd(int ieo,complex *v)
*     Copies the components of the field v on the interior boundary of
*     the local block lattice to the field components at the exterior
*     boundaries of the block lattices on the neighbouring MPI processes.
*     Depending on whether ieo=0 or ieo=1, the components associated with
*     the even or odd blocks are copied.
*
* The fields passed to cpv_int_bnd() are interpreted as elements of the
* deflation subspace spanned by the Ns local modes in the DFL_BLOCKS block
* grid. They must have at least Ns*(nb+nbb/2) elements, where nb and nbb
* are the numbers of blocks in the DFL_BLOCKS block grid and its exterior
* boundary.
*
* After calling cpv_int_bnd(), the boundary values of the field on the
* face with index ifc are stored at offset Ns*(nb+obbe[ifc]) if ieo=0 and
* at offset Ns*(nb+obbo[ifc]-nbb/2) if ieo=1 (see dfl/dfl_geometry.c for
* further explanations).
*
* In the case of boundary conditions of type 0,1 and 2, the program does not
* copy any components of the fields across the boundaries of the lattice at
* global time 0 and NPROC0*L0-1 and instead sets the field at the exterior
* boundaries of the block lattice at these times to zero.
*
* The program cpv_int_bnd() is assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define VCOM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "dfl.h"
#include "vflds.h"
#include "global.h"

static int nmu[8],sflg[8],tags[8];
static int Ns,nbf[2][8],obb[2][8],*ipp=NULL;
static int (*ofs_int)[2][8][3],(*ofs_bnd)[2][2][2];
static complex *sbufs[2];


static void set_ofs(void)
{
   int bc,nb,nbb,ieo,ifc,k,*a,*b;
   dfl_parms_t dfl;
   dfl_grid_t *dgr;

   bc=bc_type();
   dfl=dfl_parms();
   Ns=dfl.Ns;

   error_root(Ns==0,1,"set_ofs [vcom.c]",
              "Deflation subspace parameters are not set");

   set_dfl_geometry();
   dgr=dfl_geometry();
   nb=(*dgr).nb;
   nbb=(*dgr).nbb;
   ipp=(*dgr).ipp;

   a=malloc(2*NTHREAD*sizeof(*a));
   ofs_int=malloc(NTHREAD*sizeof(*ofs_int));
   ofs_bnd=malloc(NTHREAD*sizeof(*ofs_bnd));
   error((a==NULL)||(ofs_int==NULL)||(ofs_bnd==NULL),1,
         "alloc_ofs [vcom.c]","Unable to allocate auxiliary arrays");
   b=a+NTHREAD;

   for (ifc=0;ifc<8;ifc++)
   {
      nmu[ifc]=cpr[ifc/2]&0x1;

      sflg[ifc]=((ifc>1)||
                 ((ifc==0)&&(cpr[0]!=0))||
                 ((ifc==1)&&(cpr[0]!=(NPROC0-1)))||
                 (bc==3));
      sflg[ifc]&=(((*dgr).nbbe[ifc]>0)||((*dgr).nbbo[ifc]>0));

      tags[ifc]=mpi_permanent_tag();

      nbf[0][ifc]=Ns*(*dgr).nbbe[ifc];
      nbf[1][ifc]=Ns*(*dgr).nbbo[ifc];

      obb[0][ifc]=Ns*(nb+(*dgr).obbe[ifc]);
      obb[1][ifc]=Ns*(nb+(*dgr).obbo[ifc]-(nbb/2));

      a[0]=0;
      b[0]=(*dgr).nbbo[ifc];
      divide_range((*dgr).nbbo[ifc],NTHREAD-1,a+1,b+1);

      for (k=0;k<NTHREAD;k++)
      {
         ofs_int[k][0][ifc][0]=a[k]+(*dgr).obbo[ifc];
         ofs_int[k][0][ifc][1]=b[k]-a[k];
         ofs_int[k][0][ifc][2]=Ns*a[k];
      }

      a[0]=0;
      b[0]=(*dgr).nbbe[ifc];
      divide_range((*dgr).nbbe[ifc],NTHREAD-1,a+1,b+1);

      for (k=0;k<NTHREAD;k++)
      {
         ofs_int[k][1][ifc][0]=a[k]+(*dgr).obbe[ifc];
         ofs_int[k][1][ifc][1]=b[k]-a[k];
         ofs_int[k][1][ifc][2]=Ns*a[k];
      }
   }

   for (ieo=0;ieo<2;ieo++)
   {
      for (ifc=0;ifc<2;ifc++)
      {
         divide_range(nbf[ieo][ifc],NTHREAD,a,b);

         for (k=0;k<NTHREAD;k++)
         {
            ofs_bnd[k][ieo][ifc][0]=a[k]+obb[ieo][ifc];
            ofs_bnd[k][ieo][ifc][1]=b[k]-a[k];
         }
      }
   }
}


static void alloc_sbufs(void)
{
   int n,ifc;
   complex *w;

   set_ofs();
   n=0;

   for (ifc=0;ifc<8;ifc++)
   {
      if (n<nbf[0][ifc])
         n=nbf[0][ifc];
   }

   w=amalloc(2*n*sizeof(*w),4);
   error(w==NULL,1,"alloc_sbufs [vcom.c]",
         "Unable to allocate communication buffers");

   sbufs[0]=w;
   sbufs[1]=w+n;
}


static void get_vbnd(int ofs,int vol,complex *v,complex *sbuf)
{
   int *ip,*im;
   complex *vv,*vm;

   ip=ipp+ofs;
   im=ip+vol;

   for (;ip<im;ip++)
   {
      vv=v+Ns*ip[0];
      vm=vv+Ns;

      for (;vv<vm;vv+=2)
      {
         sbuf[0]=vv[0];
         sbuf[1]=vv[1];
         sbuf+=2;
      }
   }
}


static void send_buf(int ieo,int ifc,complex *v)
{
   int sfc,nsbf,nrbf,tag,saddr,raddr;
   complex *sbuf,*rbuf;
   MPI_Status snd_stat,rcv_stat;
   MPI_Request snd_req,rcv_req;

   sfc=ifc^nmu[ifc];

   saddr=npr[sfc];
   raddr=saddr;
   tag=tags[ifc];

   sbuf=sbufs[ifc&0x1];
   rbuf=v+obb[ieo][sfc];

   nsbf=2*nbf[ieo^0x1][sfc];
   nrbf=2*nbf[ieo][sfc];

   if (nmu[ifc])
   {
      if (nsbf)
         MPI_Isend(sbuf,nsbf,MPI_FLOAT,saddr,tag,MPI_COMM_WORLD,&snd_req);
      if (nrbf)
         MPI_Irecv(rbuf,nrbf,MPI_FLOAT,raddr,tag,MPI_COMM_WORLD,&rcv_req);

      if (nsbf)
         MPI_Wait(&snd_req,&snd_stat);
      if (nrbf)
         MPI_Wait(&rcv_req,&rcv_stat);
   }
   else
   {
      if (nrbf)
         MPI_Irecv(rbuf,nrbf,MPI_FLOAT,raddr,tag,MPI_COMM_WORLD,&rcv_req);
      if (nsbf)
         MPI_Isend(sbuf,nsbf,MPI_FLOAT,saddr,tag,MPI_COMM_WORLD,&snd_req);

      if (nrbf)
         MPI_Wait(&rcv_req,&rcv_stat);
      if (nsbf)
         MPI_Wait(&snd_req,&snd_stat);
   }
}


void cpv_int_bnd(int ieo,complex *v)
{
   int k,ifc,sfc;
   int ofs,vol,sof;

   if (NPROC>1)
   {
      if (ipp==NULL)
         alloc_sbufs();

#pragma omp parallel private(k,ifc,sfc,ofs,vol,sof)
      {
         k=omp_get_thread_num();

         for (ifc=0;ifc<8;ifc++)
         {
            sfc=ifc^nmu[ifc];

            if (sflg[sfc])
            {
               if ((NTHREAD==1)||(k>0))
               {
                  ofs=ofs_int[k][ieo][sfc][0];
                  vol=ofs_int[k][ieo][sfc][1];
                  sof=ofs_int[k][ieo][sfc][2];

                  get_vbnd(ofs,vol,v,sbufs[ifc&0x1]+sof);
               }

#pragma omp barrier
               if (k==0)
                  send_buf(ieo,ifc,v);
            }
            else if ((NPROC0>1)&&(sfc<2))
            {
               ofs=ofs_bnd[k][ieo][sfc][0];
               vol=ofs_bnd[k][ieo][sfc][1];

               set_v2zero(vol,0,v+ofs);
            }
         }
      }
   }
}
