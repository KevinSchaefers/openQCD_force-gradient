
/*******************************************************************************
*
* File map_sw2blk.c
*
* Copyright (C) 2005, 2011, 2016, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Copying of the SW fields to the blocks in a block grid.
*
*   int assign_swd2swbgr(blk_grid_t grid,ptset_t set)
*     Assigns the global double-precision SW field to the corresponding
*     single-precision fields in the specified grid. On the given point
*     set, the copied Pauli matrices are inverted before assignment and
*     the program returns a non-zero value if some workspace allocations
*     or matrix inversions failed.
*
*   void assign_swd2swdblk(blk_grid_t grid,int n)
*     Assigns the global double-precision SW field to the corresponding
*     double-precision field on the n'th block of the specified grid.
*
* The possible point sets are defined in utils.h. Independently of the
* specified set, the source field is left unchanged, i.e. the inversions
* are performed "on the fly" (no inversions are performed if set=NO_PTS).
* Pauli matrix inversions are considered unsafe if the Frobenius condition
* number of the matrix exceeds 100 (see pauli_dble.c).
*
* The program assign_swd2swbgr() is assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
* The program assign_swd2swdblk() is thread-safe, but takes it for granted
* that the global double-precision SW field and the block grid "grid" are
* allocated and that the argument "n" is not out of range.
*
*******************************************************************************/

#define MAP_SW2BLK_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "sw_term.h"
#include "block.h"
#include "global.h"


static int cp_swd2sw(block_t *b,ptset_t set,pauli_wsp_t *pwsp)
{
   int *imb,ifail;
   pauli *pb,*pm;
   pauli_dble *swd,*p;
   pauli_dble ms[2] ALIGNED16;

   swd=swdfld();
   pb=(*b).sw;
   pm=pb+(*b).vol;
   imb=(*b).imb;
   ifail=0;

   for (;pb<pm;pb+=2)
   {
      p=swd+2*(*imb);
      imb+=1;

      if ((set==ALL_PTS)||(set==EVEN_PTS))
      {
         ifail|=inv_pauli_dble(0.0,p,pwsp,ms);
         ifail|=inv_pauli_dble(0.0,p+1,pwsp,ms+1);
         assign_pauli(2,ms,pb);
      }
      else
         assign_pauli(2,p,pb);
   }

   pm+=(*b).vol;

   for (;pb<pm;pb+=2)
   {
      p=swd+2*(*imb);
      imb+=1;

      if ((set==ALL_PTS)||(set==ODD_PTS))
      {
         ifail|=inv_pauli_dble(0.0,p,pwsp,ms);
         ifail|=inv_pauli_dble(0.0,p+1,pwsp,ms+1);
         assign_pauli(2,ms,pb);
      }
      else
         assign_pauli(2,p,pb);
   }

   return ifail;
}


int assign_swd2swbgr(blk_grid_t grid,ptset_t set)
{
   int iprms[2],ie,io;
   int nb,isw,ifail,n,k;
   block_t *b,*bk,*bm;
   pauli_wsp_t *pwsp;

   if (NPROC>1)
   {
      iprms[0]=(int)(grid);
      iprms[1]=(int)(set);

      MPI_Bcast(iprms,2,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=(int)(grid))||(iprms[1]!=(int)(set)),1,
            "assign_swd2swbgr [map_sw2blk.c]","Parameters are not global");
   }

   b=blk_list(grid,&nb,&isw);

   if (nb==0)
   {
      error_root(1,1,"assign_swd2swbgr [map_sw2blk.c]",
                 "Block grid is not allocated");
      return 0;
   }

   if (((*b).sw==NULL)||((*b).shf&0x4))
   {
      error_root(1,1,"assign_swd2swbgr [map_sw2blk.c]",
                 "SW field on the grid is shared or not allocated");
      return 0;
   }

   ie=query_flags(SWD_E_INVERTED);
   io=query_flags(SWD_O_INVERTED);

   error_root(((ie)&&((set==ALL_PTS)||(set==EVEN_PTS)))||
              ((io)&&((set==ALL_PTS)||(set==ODD_PTS))),1,
              "assign_swd2swbgr [map_sw2blk.c]",
              "Attempt to invert the SW field a second time");

   ifail=0;

#pragma omp parallel private(k,bk,bm,pwsp) reduction(| : ifail)
   {
      k=omp_get_thread_num();
      bk=b+k;
      bm=b+nb;

      if (set!=NO_PTS)
      {
         pwsp=alloc_pauli_wsp();
         ifail|=(pwsp==NULL);
      }
      else
         pwsp=NULL;

      if (ifail==0)
      {
         for (;bk<bm;bk+=NTHREAD)
            ifail|=cp_swd2sw(bk,set,pwsp);
      }

      if (pwsp!=NULL)
         free_pauli_wsp(pwsp);
   }

   set_grid_flags(grid,ASSIGNED_SWD2SWBGR);

   if ((set==ALL_PTS)||(set==EVEN_PTS))
      set_grid_flags(grid,INVERTED_SW_E);
   if ((set==ALL_PTS)||(set==ODD_PTS))
      set_grid_flags(grid,INVERTED_SW_O);

   if (set!=NO_PTS)
   {
      n=ifail;
      MPI_Reduce(&n,&ifail,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&ifail,1,MPI_INT,0,MPI_COMM_WORLD);
   }

   return ifail;
}


void assign_swd2swdblk(blk_grid_t grid,int n)
{
   int nb,isw,*imb;
   pauli_dble *swd,*swb,*swm;
   block_t *b;

   b=blk_list(grid,&nb,&isw)+n;

   swd=swdfld();
   swb=(*b).swd;
   swm=swb+2*(*b).vol;
   imb=(*b).imb;

   for (;swb<swm;swb+=2)
   {
      swb[0]=swd[2*imb[0]];
      swb[1]=swd[2*imb[0]+1];
      imb+=1;
   }
}
