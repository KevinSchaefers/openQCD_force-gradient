
/*******************************************************************************
*
* File rotud.c
*
* Copyright (C) 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Rotation of the global gauge field.
*
*   int check_active(su3_alg_dble *fld)
*     Returns 0 if the link field fld vanishes on the inactive links
*     and is non-zero on the active ones. Otherwise 1 is returned.
*
*   void rot_ud(double eps)
*     Multiplies the global double-precision link variables U(x,mu) on
*     the active links (x,mu) by exp(eps*mom(x,mu)), where mom is the
*     global momentum field.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define ROTUD_C

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
#include "mdflds.h"
#include "devfcts.h"
#include "global.h"

#define N0 (NPROC0*L0)


static int is_zero(su3_alg_dble *X)
{
   int ie;

   ie=1;
   ie&=((*X).c1==0.0);
   ie&=((*X).c2==0.0);
   ie&=((*X).c3==0.0);
   ie&=((*X).c4==0.0);
   ie&=((*X).c5==0.0);
   ie&=((*X).c6==0.0);
   ie&=((*X).c7==0.0);
   ie&=((*X).c8==0.0);

   return ie;
}


int check_active(su3_alg_dble *fld)
{
   int bc,ie;
   int k,ofs,vol,ix,t,ifc;
   su3_alg_dble *fd;

   bc=bc_type();
   ie=0;

#pragma omp parallel private(k,ofs,vol,ix,t,ifc,fd) reduction(| : ie)
   {
      k=omp_get_thread_num();

      vol=(VOLUME_TRD/2);
      ofs=(VOLUME/2)+k*vol;
      fd=fld+8*k*vol;

      for (ix=ofs;ix<(ofs+vol);ix++)
      {
         t=global_time(ix);

         if ((t==0)&&(bc==0))
         {
            ie|=is_zero(fd);
            fd+=1;

            ie|=(is_zero(fd)^0x1);
            fd+=1;

            for (ifc=2;ifc<8;ifc++)
            {
               ie|=is_zero(fd);
               fd+=1;
            }
         }
         else if ((t==0)&&(bc==1))
         {
            ie|=is_zero(fd);
            fd+=1;

            ie|=is_zero(fd);
            fd+=1;

            for (ifc=2;ifc<8;ifc++)
            {
               ie|=(is_zero(fd)^0x1);
               fd+=1;
            }
         }
         else if ((t==(N0-1))&&(bc==0))
         {
            ie|=(is_zero(fd)^0x1);
            fd+=1;

            for (ifc=1;ifc<8;ifc++)
            {
               ie|=is_zero(fd);
               fd+=1;
            }
         }
         else
         {
            for (ifc=0;ifc<8;ifc++)
            {
               ie|=is_zero(fd);
               fd+=1;
            }
         }
      }
   }

   if (NPROC>1)
   {
      k=ie;
      MPI_Allreduce(&k,&ie,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
   }

   return ie;
}


void rot_ud(double eps)
{
   int bc;
   int k,ofs,vol,ix,t,ifc;
   su3_dble *ub,*ud;
   su3_alg_dble *mb,*mom;
   mdflds_t *mdfs;

   bc=bc_type();
   mdfs=mdflds();
   ub=udfld();
   mb=(*mdfs).mom;
   chexp_init();

#pragma omp parallel private(k,ofs,vol,ix,t,ifc,ud,mom)
   {
      k=omp_get_thread_num();

      vol=(VOLUME_TRD/2);
      ofs=(VOLUME/2)+k*vol;
      ud=ub+8*k*vol;
      mom=mb+8*k*vol;

      for (ix=ofs;ix<(ofs+vol);ix++)
      {
         t=global_time(ix);

         if (t==0)
         {
            expXsu3(eps,mom,ud);
            mom+=1;
            ud+=1;

            if (bc!=0)
               expXsu3(eps,mom,ud);
            mom+=1;
            ud+=1;

            for (ifc=2;ifc<8;ifc++)
            {
               if (bc!=1)
                  expXsu3(eps,mom,ud);
               mom+=1;
               ud+=1;
            }
         }
         else if (t==(N0-1))
         {
            if (bc!=0)
               expXsu3(eps,mom,ud);
            mom+=1;
            ud+=1;

            for (ifc=1;ifc<8;ifc++)
            {
               expXsu3(eps,mom,ud);
               mom+=1;
               ud+=1;
            }
         }
         else
         {
            for (ifc=0;ifc<8;ifc++)
            {
               expXsu3(eps,mom,ud);
               mom+=1;
               ud+=1;
            }
         }
      }
   }

   set_flags(UPDATED_UD);
}
