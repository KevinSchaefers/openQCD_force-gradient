
/*******************************************************************************
*
* File Aw_com2.c
*
* Copyright (C) 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Communication of the hopping terms on the exterior boundary of the local
* lattice.
*
*   void cpAhop_ext_bnd(int ieo,int ifc,complex_dble **Amats)
*     Copies the hopping terms Amats along the face of the local lattice in
*     direction ifc or ifc^0x1 to the neighbouring MPI process and adds them
*     to the little Dirac operator. The hopping terms are assumed to go from
*     even to odd (odd to even) blocks if cpr[ifc/2]+ieo is even (odd), while
*     the direction is ifc if cpr[ifc/2] is even and ifc^0x1 otherwise.
*
* The array elements Amats[n][k*Ns+l] are labeled by an index n that counts
* the even or odd blocks along the considered face in the order described
* in dfl/dfl_geometry.c. The matrices are assumed to be stored contiguously
* with address *Amats.
*
* See README.Aw and README.Aw_com for further details.
*
* The program cpAhop_ext_bnd() is assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously with same values of ieo and ifc.
*
*******************************************************************************/

#define AW_COM2_C

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
static int (*ofs_int)[2][8][3],*ipp;
static complex_dble *rbuf=NULL;


static void set_ofs(void)
{
   int nb,isw;
   int ifc,k,*a,*b;
   dfl_parms_t dfl;
   dfl_grid_t *grd;

   (void)(blk_list(DFL_BLOCKS,&nb,&isw));
   error(nb==0,1,"set_ofs [Aw_com2.c]",
         "The DFL_BLOCKS block grid is not allocated");

   dfl=dfl_parms();
   grd=dfl_geometry();
   Ns=dfl.Ns;
   ipp=(*grd).ipp;

   ofs_int=malloc(NTHREAD*sizeof(*ofs_int));
   a=malloc(2*NTHREAD*sizeof(*a));
   error((ofs_int==NULL)||(a==NULL),1,"set_ofs [Aw_com2.c]",
         "Unable to allocate auxiliary arrays");
   b=a+NTHREAD;

   for (ifc=0;ifc<8;ifc++)
   {
      nmu[ifc]=cpr[ifc/2]&0x1;

      nbf[0][ifc]=Ns*Ns*(*grd).nbbo[ifc];
      nbf[1][ifc]=Ns*Ns*(*grd).nbbe[ifc];

      tags[ifc^nmu[ifc]]=mpi_permanent_tag();

      divide_range((*grd).nbbo[ifc],NTHREAD,a,b);

      for (k=0;k<NTHREAD;k++)
      {
         ofs_int[k][0][ifc][0]=a[k]+(*grd).obbo[ifc];
         ofs_int[k][0][ifc][1]=b[k]-a[k];
         ofs_int[k][0][ifc][2]=a[k]*Ns*Ns;
      }

      divide_range((*grd).nbbe[ifc],NTHREAD,a,b);

      for (k=0;k<NTHREAD;k++)
      {
         ofs_int[k][1][ifc][0]=a[k]+(*grd).obbe[ifc];
         ofs_int[k][1][ifc][1]=b[k]-a[k];
         ofs_int[k][1][ifc][2]=a[k]*Ns*Ns;
      }
   }

   free(a);
}


static void alloc_rbuf(void)
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

   rbuf=amalloc(n*sizeof(*rbuf),5);
   error(rbuf==NULL,1,"alloc_rbuf [Aw_com2.c]",
         "Unable to allocate communication buffer");
}


static void add_rbuf(int ieo,int ifc)
{
   int k,l,ofs,vol,ibb,n,nmat;
   complex_dble *A,*B,**Ahop;
   Aw_dble_t Aw;

   Aw=Awop_dble();
   Ahop=Aw.Ahop;
   nmat=Ns*Ns;

#pragma omp parallel private(k,l,ofs,vol,ibb,n,A,B)
   {
      k=omp_get_thread_num();

      ofs=ofs_int[k][ieo][ifc][0];
      vol=ofs_int[k][ieo][ifc][1];
      A=rbuf+ofs_int[k][ieo][ifc][2];

      for (ibb=ofs;ibb<(ofs+vol);ibb++)
      {
         n=ipp[ibb];
         B=Ahop[8*n+ifc];

         for (l=0;l<nmat;l++)
         {
            B[0].re+=A[0].re;
            B[0].im+=A[0].im;

            A+=1;
            B+=1;
         }
      }
   }
}


static void send_Amats(int ieo,int ifc,complex_dble *Amats)
{
   int saddr,raddr,tag,nsbf;
   MPI_Status snd_stat,rcv_stat;
   MPI_Request snd_req,rcv_req;

   saddr=npr[ifc];
   raddr=saddr;
   tag=tags[ifc];
   nsbf=2*nbf[ieo][ifc];

   if (nmu[ifc])
   {
      MPI_Isend(Amats,nsbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD,&snd_req);
      MPI_Irecv(rbuf,nsbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&rcv_req);

      MPI_Wait(&snd_req,&snd_stat);
      MPI_Wait(&rcv_req,&rcv_stat);
   }
   else
   {
      MPI_Irecv(rbuf,nsbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&rcv_req);
      MPI_Isend(Amats,nsbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD,&snd_req);

      MPI_Wait(&rcv_req,&rcv_stat);
      MPI_Wait(&snd_req,&snd_stat);
   }
}


void cpAhop_ext_bnd(int ieo,int ifc,complex_dble **Amats)
{
   if (NPROC>1)
   {
      if (rbuf==NULL)
         alloc_rbuf();

      ieo^=nmu[ifc];
      ifc^=nmu[ifc];

      if (nbf[ieo][ifc]>0)
      {
         send_Amats(ieo,ifc,*Amats);
         add_rbuf(ieo,ifc);
      }
   }
}
