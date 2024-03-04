
/*******************************************************************************
*
* File Aw_com1.c
*
* Copyright (C) 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Communication of the basis fields on the interior boundary of the local
* lattice.
*
*   void cpse_int_bnd(int ieo,int ifc,spinor *flds)
*     Visits all even or odd blocks in the DFL_BLOCKS block grid along the
*     face of the local lattice in direction ifc or ifc^0x1 and copies the
*     components of the deflation modes on the interior boundary of the
*     blocks to the flds array on the neighbouring MPI process. The even
*     (odd) blocks are chosen if cpr[ifc/2]+ieo is even (odd), while the
*     direction is ifc if cpr[ifc/2] is even and ifc^0x1 otherwise.
*
* The extracted fields are stored in the array flds one after another, first
* all fields from the first block, then those from the next block, and so on,
* where the blocks are ordered as described in dfl/dfl_geometry.c.
*
* See README.Aw and README.Aw_com for further details.
*
* The program cpse_int_bnd() is assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously with same values of ieo and ifc.
*
*******************************************************************************/

#define AW_COM1_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "block.h"
#include "dfl.h"
#include "little.h"
#include "global.h"

static int Ns,nmu[8],tags[8],nbf[2][8];
static int (*ofs_int)[2][8][3],*idx,*ipp;
static spinor *sbuf=NULL;


static void set_ofs(void)
{
   int nb,isw,volh;
   int ifc,k,*a,*b;
   block_t *b0;
   dfl_parms_t dfl;
   dfl_grid_t *grd;

   b0=blk_list(DFL_BLOCKS,&nb,&isw);
   error(nb==0,1,"set_ofs [Aw_com1.c]",
         "The DFL_BLOCKS block grid is not allocated");

   dfl=dfl_parms();
   grd=dfl_geometry();
   Ns=dfl.Ns;
   idx=(*grd).idx;
   ipp=(*grd).ipp;

   ofs_int=malloc(NTHREAD*sizeof(*ofs_int));
   a=malloc(2*NTHREAD*sizeof(*a));
   error((ofs_int==NULL)||(a==NULL),1,"set_ofs [Aw_com1.c]",
         "Unable to allocate auxiliary arrays");
   b=a+NTHREAD;

   for (ifc=0;ifc<8;ifc++)
   {
      nmu[ifc]=cpr[ifc/2]&0x1;
      volh=(*b0).bb[ifc].vol/2;

      nbf[0][ifc]=Ns*volh*(*grd).nbbo[ifc];
      nbf[1][ifc]=Ns*volh*(*grd).nbbe[ifc];

      tags[ifc^nmu[ifc]]=mpi_permanent_tag();

      divide_range((*grd).nbbo[ifc],NTHREAD,a,b);

      for (k=0;k<NTHREAD;k++)
      {
         ofs_int[k][0][ifc][0]=a[k]+(*grd).obbo[ifc];
         ofs_int[k][0][ifc][1]=b[k]-a[k];
         ofs_int[k][0][ifc][2]=a[k]*Ns*volh;
      }

      divide_range((*grd).nbbe[ifc],NTHREAD,a,b);

      for (k=0;k<NTHREAD;k++)
      {
         ofs_int[k][1][ifc][0]=a[k]+(*grd).obbe[ifc];
         ofs_int[k][1][ifc][1]=b[k]-a[k];
         ofs_int[k][1][ifc][2]=a[k]*Ns*volh;
      }
   }

   free(a);
}


static void alloc_sbuf(void)
{
   int ieo,ifc,n;

   set_ofs();
   n=0;

   for (ieo=0;ieo<2;ieo++)
   {
      for (ifc=0;ifc<8;ifc++)
      {
         if (n<nbf[ieo][ifc])
            n=nbf[ieo][ifc];
      }
   }

   sbuf=amalloc(n*sizeof(*sbuf),5);
   error(sbuf==NULL,1,"alloc_sbuf [Aw_com1.c]",
         "Unable to allocate communication buffer");
}


static void get_flds(int ieo,int ifc)
{
   int nb,isw,ns;
   int k,ofs,vol,ibb,ib;
   spinor *s;
   block_t *b0;

   b0=blk_list(DFL_BLOCKS,&nb,&isw);
   ns=Ns*((*b0).bb[ifc].vol/2);

#pragma omp parallel private(k,ofs,vol,ibb,ib,s)
   {
      k=omp_get_thread_num();

      ofs=ofs_int[k][ieo][ifc][0];
      vol=ofs_int[k][ieo][ifc][1];
      s=sbuf+ofs_int[k][ieo][ifc][2];

      for (ibb=ofs;ibb<(ofs+vol);ibb++)
      {
         ib=idx[ipp[ibb]];
         gather_se(Ns,ifc,b0+ib,s);
         s+=ns;
      }
   }
}


static void send_sbuf(int ieo,int ifc,spinor *flds)
{
   int saddr,raddr,tag,nsbf;
   MPI_Status snd_stat,rcv_stat;
   MPI_Request snd_req,rcv_req;

   saddr=npr[ifc];
   raddr=saddr;
   tag=tags[ifc];
   nsbf=24*nbf[ieo][ifc];

   if (nmu[ifc])
   {
      MPI_Isend(sbuf,nsbf,MPI_FLOAT,saddr,tag,MPI_COMM_WORLD,&snd_req);
      MPI_Irecv(flds,nsbf,MPI_FLOAT,raddr,tag,MPI_COMM_WORLD,&rcv_req);

      MPI_Wait(&snd_req,&snd_stat);
      MPI_Wait(&rcv_req,&rcv_stat);
   }
   else
   {
      MPI_Irecv(flds,nsbf,MPI_FLOAT,raddr,tag,MPI_COMM_WORLD,&rcv_req);
      MPI_Isend(sbuf,nsbf,MPI_FLOAT,saddr,tag,MPI_COMM_WORLD,&snd_req);

      MPI_Wait(&rcv_req,&rcv_stat);
      MPI_Wait(&snd_req,&snd_stat);
   }
}


void cpse_int_bnd(int ieo,int ifc,spinor *flds)
{
   if (NPROC>1)
   {
      if (sbuf==NULL)
         alloc_sbuf();

      ieo^=nmu[ifc];
      ifc^=nmu[ifc];

      if (nbf[ieo][ifc]>0)
      {
         get_flds(ieo,ifc);
         send_sbuf(ieo,ifc,flds);
      }
   }
}
