
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2007-2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the program set_ltl_modes().
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "sflds.h"
#include "vflds.h"
#include "linalg.h"
#include "dirac.h"
#include "dfl.h"
#include "little.h"
#include "global.h"


static void random_basis(int Ns)
{
   int i;
   spinor **ws;

   ws=reserve_ws(Ns);

   for (i=0;i<Ns;i++)
   {
      random_s(VOLUME_TRD,2,ws[i],1.0f);
      bnd_s2zero(ALL_PTS,ws[i]);
   }

   dfl_subspace(ws);
   release_ws();
}


static void dfl_v2s(complex *v,spinor *s)
{
   int Ns,nb,nbh,isw,vol;
   int k,n,m,i;
   complex *w;
   block_t *b;
   spinor **sb;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   Ns=dfl.Ns;
   b=blk_list(DFL_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;

#pragma omp parallel private(k,n,m,i,w,sb)
   {
      k=omp_get_thread_num();

      for (n=k;n<nb;n+=NTHREAD)
      {
         if (n<nbh)
            m=n+isw*nbh;
         else
            m=n-isw*nbh;

         sb=b[m].s;
         set_s2zero(vol,0,sb[0]);
         w=v+Ns*n;

         for (i=1;i<=Ns;i++)
         {
            mulc_spinor_add(vol,0,sb[0],sb[i],*w);
            w+=1;
         }

         assign_sblk2s(DFL_BLOCKS,m,ALL_PTS,0,s);
      }
   }
}


static double check_vd(int Ns,int nvh)
{
   int i,j;
   double d,dev;
   complex_dble **vd,z;
   complex_qflt cqsm;

   vd=vdflds();
   dev=0.0;

   for (i=0;i<Ns;i++)
   {
      for (j=0;j<=i;j++)
      {
         cqsm=vprod_dble(nvh,1,vd[i],vd[j]);
         z.re=cqsm.re.q[0];
         z.im=cqsm.im.q[0];

         if (i==j)
            z.re-=1.0;

         d=z.re*z.re+z.im*z.im;

         if (d>dev)
            dev=d;
      }
   }

   return sqrt(dev);
}


static double check_Awvd(int Ns,int nvh)
{
   int i;
   double d,dev;
   qflt rqsm;
   complex_dble **vd,**wvd,z;

   vd=vdflds();
   wvd=reserve_wvd(2);

   dev=0.0;
   z.re=-1.0;
   z.im=0.0;

   for (i=0;i<Ns;i++)
   {
      assign_vd2vd(nvh,0,vd[i],wvd[0]);
      Awhat_dble(wvd[0],wvd[1]);
      mulc_vadd_dble(nvh,0,wvd[1],vd[i]+nvh,z);
      rqsm=vnorm_square_dble(nvh,1,wvd[1]);
      d=rqsm.q[0];
      rqsm=vnorm_square_dble(nvh,1,vd[i]+nvh);
      d/=rqsm.q[0];

      if (d>dev)
         dev=d;
   }

   release_wvd();

   return sqrt(dev);
}


static double check_ltl_matrix(int Ns,int nvh)
{
   int i,j,ie;
   double dev;
   complex_dble **vd,*amat,*bmat,*cmat,z;
   complex_qflt cqsm;

   vd=vdflds();
   amat=ltl_matrix();
   bmat=amalloc(2*Ns*Ns*sizeof(*amat),5);
   error(bmat==NULL,1,"check_ltl_matrix [check3.c]",
         "Unable to allocate auxiliary arrays");
   cmat=bmat+Ns*Ns;

   for (i=0;i<Ns;i++)
   {
      for (j=0;j<Ns;j++)
      {
         cqsm=vprod_dble(nvh,1,vd[i],vd[j]+nvh);
         bmat[i*Ns+j].re=cqsm.re.q[0];
         bmat[i*Ns+j].im=cqsm.im.q[0];
      }
   }

   cmat_mul_dble(Ns,amat,bmat,cmat);
   dev=0.0;

   for (i=0;i<Ns;i++)
   {
      for (j=0;j<Ns;j++)
      {
         z.re=cmat[i*Ns+j].re;
         z.im=cmat[i*Ns+j].im;

         if (i==j)
            z.re-=1.0;

         dev+=(z.re*z.re+z.im*z.im);
      }
   }

   assign_vd2vd(Ns*Ns,0,amat,bmat);
   MPI_Bcast((double*)(amat),2*Ns*Ns,MPI_DOUBLE,0,MPI_COMM_WORLD);

   ie=0;

   for (i=0;i<(Ns*Ns);i++)
      if ((amat[i].re!=bmat[i].re)||(amat[i].im!=bmat[i].im))
         ie=1;

   error(ie!=0,1,"check_ltl_matrix [check3.c]",
         "Little matrix is not globally the same");

   return sqrt(dev)/(double)(Ns);
}


static double check_dfl_dble(int Ns,int nvh)
{
   int i,j;
   double d,dev;
   qflt r;
   complex_dble z,*eta,*chi;
   complex_dble **vd,**wvd;
   complex_qflt s;

   vd=vdflds();
   wvd=reserve_wvd(4);
   eta=wvd[2];
   chi=wvd[3];
   dev=0.0;

   for (i=0;i<8;i++)
   {
      random_vd(2*nvh,0,wvd[0],1.0);
      random_vd(2*nvh,0,wvd[1],1.0);
      assign_vd2vd(2*nvh,0,wvd[0],eta);
      assign_vd2vd(2*nvh,0,wvd[1],chi);

      dfl_LRvd(eta,chi);

      z.re=-1.0;
      z.im=0.0;
      mulc_vadd_dble(nvh,0,wvd[0]+nvh,eta+nvh,z);
      mulc_vadd_dble(nvh,0,wvd[1]+nvh,chi+nvh,z);

      r=vnorm_square_dble(nvh,0,wvd[0]+nvh);
      d=r.q[0];
      r=vnorm_square_dble(nvh,0,wvd[1]+nvh);
      d+=r.q[0];

      error(d!=0.0,1,"check_dfl [check3.c]",
            "dfl_LRvd() changes the odd parts of the fields");

      for (j=0;j<Ns;j++)
      {
         s=vprod_dble(nvh,1,vd[j],eta);
         z.re=s.re.q[0];
         z.im=s.im.q[0];
         d+=(z.re*z.re+z.im*z.im);
      }

      r=vnorm_square_dble(nvh,1,wvd[0]);
      d/=r.q[0];

      if (d>dev)
         dev=d;

      Awhat_dble(chi,wvd[1]);

      z.re=1.0;
      z.im=0.0;
      mulc_vadd_dble(nvh,0,eta,wvd[1],z);

      z.re=-1.0;
      z.im=0.0;
      mulc_vadd_dble(nvh,0,eta,wvd[0],z);

      r=vnorm_square_dble(nvh,1,eta);
      d=r.q[0];
      r=vnorm_square_dble(nvh,1,wvd[0]);
      d/=r.q[0];

      if (d>dev)
         dev=d;

      random_vd(2*nvh,0,wvd[0],1.0);
      random_vd(2*nvh,0,wvd[1],1.0);
      assign_vd2vd(2*nvh,0,wvd[0],eta);
      assign_vd2vd(2*nvh,0,wvd[1],chi);

      dfl_LRvd(eta,chi);
      dfl_Lvd(wvd[0]);
      mulc_vadd_dble(2*nvh,0,eta,wvd[0],z);

      r=vnorm_square_dble(2*nvh,1,eta);
      d=r.q[0];
      r=vnorm_square_dble(nvh,1,wvd[0]);
      d/=r.q[0];

      if (d>dev)
         dev=d;

      random_vd(2*nvh,0,wvd[0],1.0);
      random_vd(2*nvh,0,wvd[1],1.0);
      assign_vd2vd(2*nvh,0,wvd[0],eta);
      assign_vd2vd(2*nvh,0,wvd[1],chi);

      dfl_RLvd(eta,chi);
      mulc_vadd_dble(2*nvh,0,wvd[0],eta,z);
      assign_vd2vd(2*nvh,0,wvd[0],eta);

      mulc_vadd_dble(nvh,0,chi+nvh,wvd[1]+nvh,z);

      dfl_LRvd(wvd[1],wvd[0]);
      set_vd2zero(nvh,0,wvd[0]+nvh);
      set_vd2zero(nvh,0,wvd[1]+nvh);

      mulc_vadd_dble(nvh,0,eta,wvd[0],z);
      r=vnorm_square_dble(2*nvh,1,eta);
      d=r.q[0];
      r=vnorm_square_dble(nvh,1,wvd[0]);
      d/=r.q[0];

      if (d>dev)
         dev=d;

      mulc_vadd_dble(nvh,0,chi,wvd[1],z);
      r=vnorm_square_dble(2*nvh,1,chi);
      d=r.q[0];
      r=vnorm_square_dble(nvh,1,wvd[1]);
      d/=r.q[0];

      if (d>dev)
         dev=d;
   }

   release_wvd();

   return sqrt(dev);
}


static double check_vflds(int Ns,int nvh)
{
   int i;
   double d,dev;
   qflt rqsm;
   complex **v;
   complex_dble **vd,**wvd,z;

   z.re=-1.0;
   z.im=0.0;
   dev=0.0;

   v=vflds();
   vd=vdflds();
   wvd=reserve_wvd(1);

   for (i=0;i<Ns;i++)
   {
      assign_v2vd(nvh,0,v[i],wvd[0]);
      mulc_vadd_dble(nvh,0,wvd[0],vd[i],z);
      rqsm=vnorm_square_dble(nvh,1,wvd[0]);
      d=rqsm.q[0];
      rqsm=vnorm_square_dble(nvh,1,vd[i]);
      d/=rqsm.q[0];
      if (d>dev)
         dev=d;
   }

   for (i=0;i<Ns;i++)
   {
      assign_v2vd(nvh,0,v[i]+nvh,wvd[0]);
      mulc_vadd_dble(nvh,0,wvd[0],vd[i]+nvh,z);
      rqsm=vnorm_square_dble(nvh,1,wvd[0]);
      d=rqsm.q[0];
      rqsm=vnorm_square_dble(nvh,1,vd[i]+nvh);
      d/=rqsm.q[0];
      if (d>dev)
         dev=d;
   }

   release_wvd();

   return sqrt(dev);
}



static double check_dfl(int Ns,int nvh)
{
   int i;
   double d,dev;
   qflt r;
   complex **wv;
   complex_dble z,**wvd;

   wv=reserve_wv(1);
   wvd=reserve_wvd(2);
   z.re=-1.0;
   z.im=0.0;
   dev=0.0;

   for (i=0;i<8;i++)
   {
      random_v(2*nvh,0,wv[0],1.0f);
      assign_v2vd(2*nvh,0,wv[0],wvd[0]);
      dfl_Lv(wv[0]);
      dfl_Lvd(wvd[0]);
      assign_v2vd(2*nvh,0,wv[0],wvd[1]);

      mulc_vadd_dble(2*nvh,0,wvd[0],wvd[1],z);

      r=vnorm_square_dble(2*nvh,0,wvd[0]);
      d=r.q[0];
      r=vnorm_square_dble(2*nvh,0,wvd[1]);
      d/=r.q[0];

      if (d>dev)
         dev=d;
   }

   release_wvd();
   release_wv();

   return sqrt(dev);
}


static double check_mds(int Ns,int nvh)
{
   int nv,k,l;
   double d,dev;
   qflt rqsm;
   complex **vs;
   complex_dble **vd,**wvd;
   spinor **mds,**ws;

   nv=2*nvh;
   mds=reserve_ws(Ns);
   ws=reserve_ws(1);
   vs=vflds();
   dev=0.0;

   for (k=0;k<Ns;k++)
   {
      dfl_v2s(vs[Ns+k],ws[0]);
      mulr_spinor_add(VOLUME_TRD,2,ws[0],mds[k],-1.0);

      d=(double)(norm_square(VOLUME,1,ws[0])/vnorm_square(nv,1,vs[Ns+k]));
      if (d>dev)
         dev=d;
   }

   release_ws();
   release_ws();

   vd=vdflds();
   wvd=reserve_wvd(1);

   for (k=0;k<Ns;k++)
   {
      assign_v2vd(nvh,0,vs[Ns+k],wvd[0]);

      for (l=0;l<Ns;l++)
         vproject_dble(nvh,1,wvd[0],vd[l]);

      rqsm=vnorm_square_dble(nvh,1,wvd[0]);
      d=rqsm.q[0];
      d/=(double)(vnorm_square(nvh,1,vs[Ns+k]));
      if (d>dev)
         dev=d;
   }

   release_wvd();

   return sqrt(dev);
}


int main(int argc,char *argv[])
{
   int my_rank,bc,is,ifail;
   int bs[4],Ns,nb,nvh;
   double phi[2],phi_prime[2],theta[3];
   double mu,dev;
   FILE *fin=NULL,*flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      fin=freopen("check1.in","r",stdin);

      printf("\n");
      printf("Check of the program set_ltl_modes()\n");
      printf("------------------------------------\n\n");

      print_lattice_sizes();

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      read_line("Ns","%d",&Ns);
      fclose(fin);

      printf("bs = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
      printf("Ns = %d\n\n",Ns);

      bc=find_opt(argc,argv,"-bc");
      is=find_opt(argc,argv,"-sw");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check3.c]",
                    "Syntax: check3 [-bc <type>] [-sw <type>]");

      if (is!=0)
         error_root(sscanf(argv[is+1],"%d",&is)!=1,1,"main [check3.c]",
                    "Syntax: check3 [-bc <type>] [-sw <type>]");
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&is,1,MPI_INT,0,MPI_COMM_WORLD);

   check_machine();
   set_lat_parms(5.5,1.0,0,NULL,is,1.978);
   print_lat_parms(0x2);

   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.38;
   theta[1]=-1.25;
   theta[2]=0.54;
   set_bc_parms(bc,1.0,1.0,0.9012,1.2034,phi,phi_prime,theta);
   print_bc_parms(0x2);

   set_sw_parms(0.125);
   set_dfl_parms(bs,Ns);
   mu=0.0376;

   start_ranlux(0,123456);
   geometry();

   alloc_ws(Ns+1);
   alloc_wv(4);
   alloc_wvd(4);

   nb=VOLUME/(bs[0]*bs[1]*bs[2]*bs[3]);
   nvh=Ns*(nb/2);

   random_ud();
   set_ud_phase();
   random_basis(Ns);
   ifail=set_Awhat(mu);
   error_root(ifail!=0,1,"main [check3.c]",
              "Computation of the little Dirac operator failed");

   if (my_rank==0)
      printf("Maximal relative deviations found:\n\n");

   dev=check_vd(Ns,nvh);

   if (my_rank==0)
      printf("Orthonormality of vdflds: %.2e\n",dev);

   dev=check_Awvd(Ns,nvh);

   if (my_rank==0)
      printf("Awhat*vdflds:             %.2e\n",dev);

   dev=check_ltl_matrix(Ns,nvh);

   if (my_rank==0)
      printf("Little-little matrix:     %.2e\n",dev);

   dev=check_dfl_dble(Ns,nvh);

   if (my_rank==0)
      printf("Deflation projection:     %.2e\n\n",dev);

   dev=check_vflds(Ns,nvh);

   if (my_rank==0)
      printf("Single-precision fields:  %.2e\n",dev);

   dev=check_dfl(Ns,nvh);

   if (my_rank==0)
      printf("Deflation projection:     %.2e\n",dev);

   dev=check_mds(Ns,nvh);

   if (my_rank==0)
   {
      printf("Global deflation modes:   %.2e\n\n",dev);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
