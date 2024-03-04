
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2005, 2011, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the programs in the module vinit.c.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "linalg.h"
#include "vflds.h"
#include "global.h"

#define NFLDS 5

static float sig[NFLDS];
static double sigd[NFLDS];


int main(int argc,char *argv[])
{
   int my_rank,ie,k,ix;
   int bs[4],Ns,nb,nv,nv_trd;
   double var,var_all,d,dmax;
   qflt rqsm;
   complex z;
   complex_dble zd;
   complex **wv;
   complex_dble **wvd;
   FILE *fin=NULL,*flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);
      fin=freopen("check1.in","r",stdin);

      printf("\n");
      printf("Check of the programs in the module vinit\n");
      printf("-----------------------------------------\n\n");

      print_lattice_sizes();

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      read_line("Ns","%d",&Ns);
      fclose(fin);

      printf("bs = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
      printf("Ns = %d\n\n",Ns);
      fflush(flog);
   }

   check_machine();
   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);

   start_ranlux(0,12345);
   geometry();
   set_dfl_parms(bs,Ns);

   alloc_wv(2*NFLDS);
   alloc_wvd(2*NFLDS);
   wv=reserve_wv(2*NFLDS);
   wvd=reserve_wvd(2*NFLDS);

   nb=VOLUME/(bs[0]*bs[1]*bs[2]*bs[3]);
   nv=Ns*nb;
   nv_trd=nv/NTHREAD;
   z.im=0.0f;
   zd.im=0.0;
   ie=0;

   if (my_rank==0)
   {
      printf("Choose random single-precision fields\n");
      ranlxs(sig,NFLDS);
   }

   MPI_Bcast(sig,NFLDS,MPI_FLOAT,0,MPI_COMM_WORLD);

   for (k=0;k<NFLDS;k++)
   {
      random_v(nv_trd,2,wv[k],sig[k]);
      var=0.0;

      for (ix=0;ix<nv;ix++)
         var+=(double)((wv[k][ix].re*wv[k][ix].re+
                        wv[k][ix].im*wv[k][ix].im));

      MPI_Reduce(&var,&var_all,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

      if (my_rank==0)
      {
         var_all/=((double)(NPROC)*(double)(nv));
         printf("<s[%d]^2> = %.4e (sigma^2 = %.4e)\n",
                k,var_all,sig[k]*sig[k]);
      }
   }

   if (my_rank==0)
   {
      printf("\n");
      printf("Choose random double-precision fields\n");
      ranlxd(sigd,NFLDS);
   }

   MPI_Bcast(sigd,NFLDS,MPI_DOUBLE,0,MPI_COMM_WORLD);

   for (k=0;k<NFLDS;k++)
   {
      random_vd(nv_trd,2,wvd[k],sigd[k]);
      var=0.0;

      for (ix=0;ix<nv;ix++)
         var+=(wvd[k][ix].re*wvd[k][ix].re+wvd[k][ix].im*wvd[k][ix].im);

      MPI_Reduce(&var,&var_all,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

      if (my_rank==0)
      {
         var_all/=((double)(NPROC)*(double)(nv));
         printf("<sd[%d]^2> = %.4e (sigma^2 = %.4e)\n",
                k,var_all,sigd[k]*sigd[k]);
      }
   }

   for (k=0;k<NFLDS;k++)
   {
      random_v(nv_trd,2,wv[k],1.0f);
      random_vd(nv_trd,2,wvd[k],1.0);
      assign_v2v(nv_trd,2,wv[k],wv[k+NFLDS]);
      assign_vd2vd(nv_trd,2,wvd[k],wvd[k+NFLDS]);

      z.re=-1.0f;
      zd.re=-1.0;
      mulc_vadd(nv_trd,2,wv[k],wv[k+NFLDS],z);
      mulc_vadd_dble(nv_trd,2,wvd[k],wvd[k+NFLDS],zd);

      for (ix=0;ix<nv;ix++)
      {
         if ((wv[k][ix].re!=0.0f)||(wv[k][ix].im!=0.0f))
            ie=1;
         if ((wvd[k][ix].re!=0.0)||(wvd[k][ix].im!=0.0))
            ie=2;
      }
   }

   error(ie==1,1,"main [check2.c]","assign_v2v() is incorrect");
   error(ie==2,1,"main [check2.c]","assign_vd2vd() is incorrect");

   for (k=0;k<NFLDS;k++)
   {
      random_v(nv_trd,2,wv[k],1.0f);
      assign_v2vd(nv_trd,2,wv[k],wvd[k]);
      assign_vd2v(nv_trd,2,wvd[k],wv[k+NFLDS]);

      z.re=-1.0f;
      mulc_vadd(nv_trd,2,wv[k],wv[k+NFLDS],z);

      for (ix=0;ix<nv;ix++)
      {
         if ((wv[k][ix].re!=0.0f)||(wv[k][ix].im!=0.0f))
            ie=1;
      }
   }

   error(ie==1,1,"main [check2.c]",
         "assign_v2vd() or assign_vd2v() is incorrect");

   dmax=0.0;

   for (k=1;k<NFLDS;k++)
   {
      random_vd(nv_trd,2,wvd[0],1.0);
      random_vd(nv_trd,2,wvd[k],1.0);
      assign_vd2vd(nv_trd,2,wvd[k],wvd[k+NFLDS]);

      add_vd2vd(nv_trd,2,wvd[0],wvd[k]);
      zd.re=1.0;
      mulc_vadd_dble(nv_trd,2,wvd[0],wvd[k+NFLDS],zd);
      zd.re=-1.0;
      mulc_vadd_dble(nv_trd,2,wvd[0],wvd[k],zd);

      rqsm=vnorm_square_dble(nv_trd,3,wvd[0]);
      d=rqsm.q[0];
      rqsm=vnorm_square_dble(nv_trd,3,wvd[k]);
      d/=rqsm.q[0];
      if (d>dmax)
         dmax=d;
   }

   if (my_rank==0)
   {
      printf("\n");
      printf("add_vd2vd():   %.1e (should be 0.0)\n",
             sqrt(dmax));
   }

   dmax=0.0;

   for (k=1;k<NFLDS;k++)
   {
      random_vd(nv_trd,2,wvd[0],1.0);
      random_vd(nv_trd,2,wvd[k],1.0);
      assign_vd2vd(nv_trd,2,wvd[k],wvd[k+NFLDS]);

      diff_vd2vd(nv_trd,2,wvd[0],wvd[k]);
      zd.re=-1.0;
      mulc_vadd_dble(nv_trd,2,wvd[0],wvd[k+NFLDS],zd);
      mulc_vadd_dble(nv_trd,2,wvd[0],wvd[k],zd);

      rqsm=vnorm_square_dble(nv_trd,3,wvd[0]);
      d=rqsm.q[0];
      rqsm=vnorm_square_dble(nv_trd,3,wvd[k]);
      d/=rqsm.q[0];
      if (d>dmax)
         dmax=d;
   }

   if (my_rank==0)
      printf("diff_vd2vd():  %.1e (should be 0.0)\n",
             sqrt(dmax));

   dmax=0.0;

   for (k=1;k<NFLDS;k++)
   {
      random_v(nv_trd,2,wv[0],1.0f);
      random_v(nv_trd,2,wv[k],1.0f);
      assign_v2v(nv_trd,2,wv[k],wv[k+NFLDS]);

      diff_v2v(nv_trd,2,wv[0],wv[k]);
      z.re=-1.0f;
      mulc_vadd(nv_trd,2,wv[0],wv[k+NFLDS],z);
      mulc_vadd(nv_trd,2,wv[0],wv[k],z);

      d=vnorm_square(nv_trd,3,wv[0])/vnorm_square(nv_trd,3,wv[k]);
      if (d>dmax)
         dmax=d;
   }

   if (my_rank==0)
      printf("diff_v2v():    %.1e (should be 0.0)\n\n",
             sqrt(dmax));

   dmax=0.0;

   for (k=0;k<NFLDS;k++)
   {
      random_v(nv_trd,2,wv[k],1.0f);
      random_vd(nv_trd,2,wvd[k],1.0);
      assign_vd2vd(nv_trd,2,wvd[k],wvd[k+NFLDS]);

      add_v2vd(nv_trd,2,wv[k],wvd[k]);
      rqsm=vnorm_square_dble(nv_trd,3,wvd[k]);
      d=rqsm.q[0];
      zd.re=-1.0;
      mulc_vadd_dble(nv_trd,2,wvd[k],wvd[k+NFLDS],zd);
      assign_v2vd(nv_trd,2,wv[k],wvd[k+NFLDS]);
      mulc_vadd_dble(nv_trd,2,wvd[k],wvd[k+NFLDS],zd);

      rqsm=vnorm_square_dble(nv_trd,3,wvd[k]);
      d=rqsm.q[0]/d;
      if (d>dmax)
         dmax=d;
   }

   if (my_rank==0)
      printf("add_v2vd():  %.1e (should be less than 1.0e-7 or so)\n",
             sqrt(dmax));

   dmax=0.0;

   for (k=0;k<NFLDS;k++)
   {
      random_vd(nv_trd,2,wvd[k],1.0);
      random_vd(nv_trd,2,wvd[k+NFLDS],1.0);

      diff_vd2v(nv_trd,2,wvd[k],wvd[k+NFLDS],wv[k]);
      zd.re=-1.0;
      mulc_vadd_dble(nv_trd,2,wvd[k],wvd[k+NFLDS],zd);
      rqsm=vnorm_square_dble(nv_trd,3,wvd[k]);
      d=rqsm.q[0];
      assign_v2vd(nv_trd,2,wv[k],wvd[k+NFLDS]);
      mulc_vadd_dble(nv_trd,2,wvd[k],wvd[k+NFLDS],zd);

      rqsm=vnorm_square_dble(nv_trd,3,wvd[k]);
      d=rqsm.q[0]/d;
      if (d>dmax)
         dmax=d;
   }

   if (my_rank==0)
   {
      printf("diff_vd2v(): %.1e (should be less than 1.0e-7 or so)\n\n",
             sqrt(dmax));
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
