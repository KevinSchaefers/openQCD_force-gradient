
/*******************************************************************************
*
* File gtrans.c
*
* Copyright (C) 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Gauge transformations.
*
*   su3_dble *gtrans(void)
*     Returns the address of the gobal double-precision gauge-transformation
*     field. If it is not already allocated, the field is allocated and set
*     to unity.
*
*   void random_gtrans(void)
*     Sets the global double-precision gauge-transformation field to random
*     values in SU(3). On the inactive links, the field is set to unity.
*
*   void apply_gtrans2ud(void)
*     Applies the current double-precision gauge transformation to the
*     global double-precision gauge field.
*
* Gauge transformations are internally represented by a field gd[NSPIN] of
* SU(3) matrices. The BNDRY/2 elements at the end of the array are used to
* store copies of the field at the even exterior boundary points of the
* local lattice.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define GTRANS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "random.h"
#include "uflds.h"
#include "mdflds.h"
#include "sflds.h"
#include "devfcts.h"
#include "global.h"

#define N0 (NPROC0*L0)

static int tags[8],nfc[8],ofs_bnd[8],(*ofs_buf)[8][2]=NULL;
static su3_dble *gd=NULL,*gdbuf;


static void alloc_gtrans(void)
{
   gd=amalloc((NSPIN+(BNDRY/2))*sizeof(*gd),5);
   error(gd==NULL,1,"alloc_gtrans [gtrans.c]",
         "Unable to allocate gauge-transformation field");

   if (BNDRY)
      gdbuf=gd+NSPIN;
   else
      gdbuf=NULL;

   set_ud2unity(VOLUME_TRD,2,gd);
}


su3_dble *gtrans(void)
{
   if (gd==NULL)
      alloc_gtrans();

   return gd;
}


static void set_ofs(void)
{
   int ifc,k,*a,*b;

   nfc[0]=FACE0/2;
   nfc[1]=FACE0/2;
   nfc[2]=FACE1/2;
   nfc[3]=FACE1/2;
   nfc[4]=FACE2/2;
   nfc[5]=FACE2/2;
   nfc[6]=FACE3/2;
   nfc[7]=FACE3/2;

   ofs_bnd[0]=0;
   ofs_bnd[1]=ofs_bnd[0]+nfc[0];
   ofs_bnd[2]=ofs_bnd[1]+nfc[1];
   ofs_bnd[3]=ofs_bnd[2]+nfc[2];
   ofs_bnd[4]=ofs_bnd[3]+nfc[3];
   ofs_bnd[5]=ofs_bnd[4]+nfc[4];
   ofs_bnd[6]=ofs_bnd[5]+nfc[5];
   ofs_bnd[7]=ofs_bnd[6]+nfc[6];

   ofs_buf=malloc(NTHREAD*sizeof(*ofs_buf));
   a=malloc(2*NTHREAD*sizeof(*a));
   b=a+NTHREAD;
   error((ofs_buf==NULL)||(a==NULL),1,"set_ofs [gtrans.c]",
         "Unable to allocate auxiliary arrays");

   for (ifc=0;ifc<8;ifc++)
   {
      tags[ifc]=mpi_permanent_tag();
      divide_range(nfc[ifc],NTHREAD,a,b);

      for (k=0;k<NTHREAD;k++)
      {
         ofs_buf[k][ifc][0]=a[k]+ofs_bnd[ifc];
         ofs_buf[k][ifc][1]=b[k]-a[k];
      }
   }

   free(a);
}


static void pack_gdbuf(void)
{
   int k,ofs,vol,ifc,ib;

#pragma omp parallel private(k,ofs,vol,ifc,ib)
   {
      k=omp_get_thread_num();

      for (ifc=0;ifc<8;ifc++)
      {
         ofs=ofs_buf[k][ifc][0];
         vol=ofs_buf[k][ifc][1];

         for (ib=ofs;ib<(ofs+vol);ib++)
            gdbuf[ib]=gd[map[ib]];
      }
   }
}


static void send_gdbuf(void)
{
   int ifc,np,saddr,raddr;
   int nbf,tag;
   su3_dble *sbuf,*rbuf;
   MPI_Status stat;

   np=cpr[0]+cpr[1]+cpr[2]+cpr[3];

   for (ifc=0;ifc<8;ifc++)
   {
      nbf=18*nfc[ifc];

      if (nbf>0)
      {
         tag=tags[ifc];
         saddr=npr[ifc^0x1];
         raddr=npr[ifc];
         sbuf=gdbuf+ofs_bnd[ifc];
         rbuf=gd+VOLUME+ofs_bnd[ifc];

         if (np&0x1)
         {
            MPI_Send(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
            MPI_Recv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&stat);
         }
         else
         {
            MPI_Recv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&stat);
            MPI_Send(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
         }
      }
   }
}


void random_gtrans(void)
{
   int bc;
   int k,ofs,vol,ix,t;
   su3_dble *g;

   if (gd==NULL)
      alloc_gtrans();

   bc=bc_type();

#pragma omp parallel private(k,ofs,vol,ix,t,g)
   {
      k=omp_get_thread_num();
      vol=VOLUME_TRD;
      ofs=k*vol;
      g=gd+ofs;

      for (ix=ofs;ix<(ofs+vol);ix++)
      {
         t=global_time(ix);

         if ((t>0)||(bc!=1))
            random_su3_dble(g);
         else
            cm3x3_unity(1,g);

         g+=1;
      }
   }

   if (BNDRY>0)
   {
      if (ofs_buf==NULL)
         set_ofs();

      pack_gdbuf();
      send_gdbuf();
   }
}


static void apply_gtrans2ud_part(int ofs,int vol)
{
   int bc,ix,iy,t,ifc;
   su3_dble wd ALIGNED16;
   su3_dble *ud;

   bc=bc_type();
   ud=udfld()+8*(ofs-(VOLUME/2));

   for (ix=ofs;ix<(ofs+vol);ix++)
   {
      t=global_time(ix);

      if (t==0)
      {
         iy=iup[ix][0];
         su3xsu3dag(ud,gd+iy,&wd);
         su3xsu3(gd+ix,&wd,ud);
         ud+=1;

         if (bc==3)
         {
            iy=idn[ix][0];
            su3xsu3dag(ud,gd+ix,&wd);
            su3xsu3(gd+iy,&wd,ud);
         }
         else if (bc!=0)
         {
            iy=idn[ix][0];
            su3xsu3(gd+iy,ud,&wd);
            (*ud)=wd;
         }

         ud+=1;

         for (ifc=2;ifc<8;ifc++)
         {
            if (bc!=1)
            {
               if (ifc&0x1)
               {
                  iy=idn[ix][ifc/2];
                  su3xsu3dag(ud,gd+ix,&wd);
                  su3xsu3(gd+iy,&wd,ud);
               }
               else
               {
                  iy=iup[ix][ifc/2];
                  su3xsu3dag(ud,gd+iy,&wd);
                  su3xsu3(gd+ix,&wd,ud);
               }
            }

            ud+=1;
         }
      }
      else if (t==(N0-1))
      {
         if (bc==3)
         {
            iy=iup[ix][0];
            su3xsu3dag(ud,gd+iy,&wd);
            su3xsu3(gd+ix,&wd,ud);
         }
         else if (bc!=0)
         {
            su3xsu3(gd+ix,ud,&wd);
            (*ud)=wd;
         }

         ud+=1;

         for (ifc=1;ifc<8;ifc++)
         {
            if (ifc&0x1)
            {
               iy=idn[ix][ifc/2];
               su3xsu3dag(ud,gd+ix,&wd);
               su3xsu3(gd+iy,&wd,ud);
            }
            else
            {
               iy=iup[ix][ifc/2];
               su3xsu3dag(ud,gd+iy,&wd);
               su3xsu3(gd+ix,&wd,ud);
            }

            ud+=1;
         }
      }
      else
      {
         for (ifc=0;ifc<8;ifc++)
         {
            if (ifc&0x1)
            {
               iy=idn[ix][ifc/2];
               su3xsu3dag(ud,gd+ix,&wd);
               su3xsu3(gd+iy,&wd,ud);
            }
            else
            {
               iy=iup[ix][ifc/2];
               su3xsu3dag(ud,gd+iy,&wd);
               su3xsu3(gd+ix,&wd,ud);
            }

            ud+=1;
         }
      }
   }
}


void apply_gtrans2ud(void)
{
   int k,ofs,vol;

   (void)(gtrans());
   (void)(udfld());

#pragma omp parallel private(k,ofs,vol)
   {
      k=omp_get_thread_num();
      vol=VOLUME_TRD/2;
      ofs=(VOLUME/2)+k*vol;
      apply_gtrans2ud_part(ofs,vol);
   }

   set_flags(UPDATED_UD);
}
