
/*******************************************************************************
*
* File dutils.c
*
* Copyright (C) 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Utility programs for the check programs in the devel directories.
*
*   void random_shift(int *svec)
*     Generates a non-zero random translation vector svec[0],..,svec[3]
*     with elements svec[mu] of magnitude less than or equal to half the
*     lattice size in direction mu. The computed vector is the same on
*     all MPI processes and the component svec[0] is set to zero if open,
*     SF and openSF boundary conditions are chosen.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define DUTILS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "random.h"
#include "devfcts.h"
#include "global.h"


void random_shift(int *svec)
{
   int my_rank,bc,mu,bs[4];
   double r[4];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      (void)(rlxd_size());

      bc=bc_type();
      bs[0]=NPROC0*L0;
      bs[1]=NPROC1*L1;
      bs[2]=NPROC2*L2;
      bs[3]=NPROC3*L3;

      for (mu=0;mu<4;mu++)
         svec[mu]=0;

      while ((svec[0]==0)&&(svec[1]==0)&&(svec[2]==0)&&(svec[3]==0))
      {
         ranlxd(r,4);

         for (mu=0;mu<4;mu++)
         {
            svec[mu]=(int)((double)(bs[mu])*r[mu]);
            if (svec[mu]>(bs[mu]/2))
               svec[mu]-=bs[mu];
         }

         if (bc!=3)
            svec[0]=0;
      }
   }

   MPI_Bcast(svec,4,MPI_INT,0,MPI_COMM_WORLD);
}
