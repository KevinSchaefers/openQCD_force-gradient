
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2007-2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the programs in the module dfl_subspace.c.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "random.h"
#include "utils.h"
#include "flags.h"
#include "lattice.h"
#include "block.h"
#include "linalg.h"
#include "sflds.h"
#include "vflds.h"
#include "dfl.h"
#include "global.h"


static double check_basis(int Ns)
{
   int nb,isw,i,j;
   double dev,dmx;
   complex_dble z;
   complex_qflt zq;
   block_t *b,*bm;

   b=blk_list(DFL_BLOCKS,&nb,&isw);
   bm=b+nb;

   dmx=0.0;

   for (;b<bm;b++)
   {
      for (i=1;i<=Ns;i++)
      {
         assign_s2sd((*b).vol,0,(*b).s[i],(*b).sd[0]);
         dev=0.0;

         for (j=1;j<=Ns;j++)
         {
            assign_s2sd((*b).vol,0,(*b).s[j],(*b).sd[1]);
            zq=spinor_prod_dble((*b).vol,0,(*b).sd[0],(*b).sd[1]);
            z.re=zq.re.q[0];
            z.im=zq.im.q[0];

            if (i!=j)
               dev+=z.re*z.re+z.im*z.im;
            else
               dev+=(z.re-1.0)*(z.re-1.0);
         }

         if (dev>dmx)
            dmx=dev;
      }
   }

   if (NPROC>1)
   {
      dev=dmx;
      MPI_Reduce(&dev,&dmx,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmx,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }

   return sqrt(dmx);
}


int main(int argc,char *argv[])
{
   int my_rank,bc,i,j;
   int bs[4],Ns,nv;
   double phi[2],phi_prime[2],theta[3];
   double sm,dev,dmx[5];
   complex w,**vm;
   complex_dble z,**wvd;
   qflt rqsm;
   spinor **ws;
   FILE *fin=NULL,*flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);
      fin=freopen("check2.in","r",stdin);

      printf("\n");
      printf("Check of the programs in the module dfl_subspace.c\n");
      printf("--------------------------------------------------\n\n");

      print_lattice_sizes();

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      read_line("Ns","%d",&Ns);
      fclose(fin);

      printf("bs = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
      printf("Ns = %d\n\n",Ns);

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check2.c]",
                    "Syntax: check2 [-bc <type>]");
   }

   check_machine();
   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   phi[0]=0.0;
   phi[1]=0.0;
   phi_prime[0]=0.0;
   phi_prime[1]=0.0;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;
   set_bc_parms(bc,1.0,1.0,1.0,1.0,phi,phi_prime,theta);
   print_bc_parms(0);

   start_ranlux(0,123456);
   geometry();
   set_dfl_parms(bs,Ns);

   alloc_ws(2*Ns);
   alloc_wvd(2);

   ws=reserve_ws(2*Ns);
   vm=vflds()+Ns;
   wvd=reserve_wvd(2);
   nv=Ns*VOLUME/(bs[0]*bs[1]*bs[2]*bs[3]);

   for (i=0;i<Ns;i++)
   {
      random_s(VOLUME_TRD,2,ws[i],1.0f);
      bnd_s2zero(ALL_PTS,ws[i]);
   }

   dfl_subspace(ws);
   dmx[0]=check_basis(Ns);
   dmx[1]=0.0;
   dmx[2]=0.0;
   dmx[3]=0.0;
   dmx[4]=0.0;

   for (i=0;i<Ns;i++)
   {
      assign_v2vd(nv,0,vm[i],wvd[0]);
      dfl_vd2s(wvd[0],ws[Ns]);
      mulr_spinor_add(VOLUME_TRD,2,ws[Ns],ws[i],-1.0f);
      dev=(double)(norm_square(VOLUME_TRD,3,ws[Ns])/
                   norm_square(VOLUME_TRD,3,ws[i]));
      dev=sqrt(dev);

      if (dev>dmx[1])
         dmx[1]=dev;
   }

   for (i=0;i<10;i++)
   {
      random_vd(nv,0,wvd[0],1.0);
      dfl_vd2s(wvd[0],ws[Ns]);
      dfl_s2vd(ws[Ns],wvd[1]);
      z.re=-1.0;
      z.im=0.0;
      mulc_vadd_dble(nv,0,wvd[0],wvd[1],z);
      rqsm=vnorm_square_dble(nv,1,wvd[0]);
      dev=rqsm.q[0];
      rqsm=vnorm_square_dble(nv,1,wvd[1]);
      dev/=rqsm.q[0];
      dev=sqrt(dev);

      if (dev>dmx[2])
         dmx[2]=dev;
   }

   for (i=0;i<Ns;i++)
      random_s(VOLUME_TRD,2,ws[i+Ns],1.0f);

   dfl_restore_modes(ws+Ns);

   for (i=0;i<Ns;i++)
   {
      mulr_spinor_add(VOLUME_TRD,2,ws[i+Ns],ws[i],-1.0f);
      dev=(double)(norm_square(VOLUME_TRD,3,ws[i+Ns])/
                   norm_square(VOLUME_TRD,3,ws[i]));
      dev=sqrt(dev);

      if (dev>dmx[3])
         dmx[3]=dev;
   }

   dfl_renormalize_modes(ws);

   for (i=0;i<Ns;i++)
   {
      sm=0.0;

      for (j=0;j<Ns;j++)
      {
         if (j!=i)
         {
            w=spinor_prod(VOLUME_TRD,3,ws[i],ws[j]);
            sm+=(double)(w.re*w.re+w.im*w.im);
         }
      }

      dev=(double)(norm_square(VOLUME_TRD,3,ws[i]));
      sm=sqrt(sm/dev);

      if (sm>dmx[4])
         dmx[4]=sm;

      dev=fabs(1.0-dev);

      if (dev>dmx[4])
         dmx[4]=dev;
   }

   if (my_rank==0)
   {
      printf("Orthonormality of the basis vectors: %.1e\n",dmx[0]);
      printf("Check of the global vector modes:    %.1e\n",dmx[1]);
      printf("Check of dfl_s2vd() and dfl_vd2s():  %.1e\n",dmx[2]);
      printf("Check of dfl_restore_modes():        %.1e\n",dmx[3]);
      printf("Check of dfl_renormalize_modes():    %.1e\n\n",dmx[4]);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
