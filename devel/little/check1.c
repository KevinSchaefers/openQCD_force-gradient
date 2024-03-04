
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2007-2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Direct checks of Aw_dble() and Aw().
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
#include "sw_term.h"
#include "dirac.h"
#include "dfl.h"
#include "little.h"
#include "global.h"

static Aw_dble_t Awds={0,0,NULL,NULL};


static void save_Awd(void)
{
   int Ns,nb,n;
   complex_dble *w,**ww;
   Aw_dble_t Awd;

   Awd=Awop_dble();
   Ns=Awd.Ns;
   nb=Awd.nb;

   if (Awds.Ns==0)
   {
      ww=malloc(9*nb*sizeof(*ww));
      w=amalloc(9*nb*Ns*Ns*sizeof(*w),6);

      error((ww==NULL)||(w==NULL),1,"save_Awd [check1.c]",
            "Unable to allocate matrix arrays");

      for (n=0;n<(9*nb);n++)
      {
         ww[n]=w;
         w+=Ns*Ns;
      }

      Awds.Ablk=ww;
      Awds.Ahop=ww+nb;
   }

   assign_vd2vd(9*nb*Ns*Ns,0,Awd.Ablk[0],Awds.Ablk[0]);
}


static void check_Awd(double *dev)
{
   int Ns,nb,n,ifc,l;
   int (*inn)[8];
   double d,dmy[3];
   complex_dble z;
   Aw_dble_t Awd;
   dfl_grid_t *grd;

   grd=dfl_geometry();
   inn=(*grd).inn;

   Awd=Awop_dble();
   Ns=Awd.Ns;
   nb=Awd.nb;

   dev[0]=0.0;
   dev[1]=0.0;
   dev[2]=0.0;

   for (n=0;n<nb;n++)
   {
      for (l=0;l<(Ns*Ns);l++)
      {
         z.re=Awd.Ablk[n][l].re-Awds.Ablk[n][l].re;
         z.im=Awd.Ablk[n][l].im-Awds.Ablk[n][l].im;

         d=z.re*z.re+z.im*z.im;
         if (d>dev[0])
            dev[0]=d;
      }
   }

   for (n=0;n<nb;n++)
   {
      for (ifc=0;ifc<8;ifc++)
      {
         for (l=0;l<(Ns*Ns);l++)
         {
            z.re=Awd.Ahop[8*n+ifc][l].re-Awds.Ahop[8*n+ifc][l].re;
            z.im=Awd.Ahop[8*n+ifc][l].im-Awds.Ahop[8*n+ifc][l].im;

            d=z.re*z.re+z.im*z.im;

            if (inn[n][ifc]<nb)
            {
               if (d>dev[1])
                  dev[1]=d;
            }
            else
            {
               if (d>dev[2])
                  dev[2]=d;
            }
         }
      }
   }

   dev[0]=sqrt(dev[0]);
   dev[1]=sqrt(dev[1]);
   dev[2]=sqrt(dev[2]);

   if (NPROC>1)
   {
      MPI_Reduce(dev,dmy,3,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(dmy,3,MPI_DOUBLE,0,MPI_COMM_WORLD);

      dev[0]=dmy[0];
      dev[1]=dmy[1];
      dev[2]=dmy[2];
   }
}


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


static void dfl_sd2vd(spinor_dble *sd,complex_dble *vd)
{
   int Ns,nb,nbh,isw,vol;
   int k,n,m,i;
   complex_dble *wd;
   complex_qflt cqsm;
   block_t *b;
   spinor **sb;
   spinor_dble **sdb;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   Ns=dfl.Ns;
   b=blk_list(DFL_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;

#pragma omp parallel private(k,n,m,i,cqsm,wd,sb,sdb)
   {
      k=omp_get_thread_num();

      for (n=k;n<nb;n+=NTHREAD)
      {
         if (n<nbh)
            m=n+isw*nbh;
         else
            m=n-isw*nbh;

         assign_sd2sdblk(DFL_BLOCKS,m,ALL_PTS,sd,0);
         sb=b[m].s;
         sdb=b[m].sd;
         wd=vd+Ns*n;

         for (i=1;i<=Ns;i++)
         {
            assign_s2sd(vol,0,sb[i],sdb[1]);
            cqsm=spinor_prod_dble(vol,0,sdb[1],sdb[0]);
            (*wd).re=cqsm.re.q[0];
            (*wd).im=cqsm.im.q[0];
            wd+=1;
         }
      }
   }
}


static void dfl_vd2sd(complex_dble *vd,spinor_dble *sd)
{
   int Ns,nb,nbh,isw,vol;
   int k,n,m,i;
   complex_dble *wd;
   block_t *b;
   spinor **sb;
   spinor_dble **sdb;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   Ns=dfl.Ns;
   b=blk_list(DFL_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;

#pragma omp parallel private(k,n,m,i,wd,sb,sdb)
   {
      k=omp_get_thread_num();

      for (n=k;n<nb;n+=NTHREAD)
      {
         if (n<nbh)
            m=n+isw*nbh;
         else
            m=n-isw*nbh;

         sb=b[m].s;
         sdb=b[m].sd;
         set_sd2zero(vol,0,sdb[0]);
         wd=vd+Ns*n;

         for (i=1;i<=Ns;i++)
         {
            assign_s2sd(vol,0,sb[i],sdb[1]);
            mulc_spinor_add_dble(vol,0,sdb[0],sdb[1],*wd);
            wd+=1;
         }

         assign_sdblk2sd(DFL_BLOCKS,m,ALL_PTS,0,sd);
      }
   }
}


static void dfl_s2v(spinor *s,complex *v)
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

         assign_s2sblk(DFL_BLOCKS,m,ALL_PTS,s,0);
         sb=b[m].s;
         w=v+Ns*n;

         for (i=1;i<=Ns;i++)
         {
            (*w)=spinor_prod(vol,0,sb[i],sb[0]);
            w+=1;
         }
      }
   }
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


int main(int argc,char *argv[])
{
   int my_rank,bc,is;
   int bs[4],Ns,nb,nv;
   int ieo,im0,imu;
   double phi[2],phi_prime[2],theta[3];
   double m0[2],mu[2],dev[3];
   qflt rqsm;
   complex **wv,z;
   complex_dble **wvd,zd;
   spinor **ws;
   spinor_dble **wsd;
   FILE *fin=NULL,*flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);
      fin=freopen("check1.in","r",stdin);

      printf("\n");
      printf("Direct checks of Aw_dble() and Aw()\n");
      printf("-----------------------------------\n\n");

      print_lattice_sizes();

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      read_line("Ns","%d",&Ns);
      fclose(fin);

      printf("bs = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
      printf("Ns = %d\n\n",Ns);

      bc=find_opt(argc,argv,"-bc");
      is=find_opt(argc,argv,"-sw");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check1.c]",
                    "Syntax: check1 [-bc <type>] [-sw <type>]");

      if (is!=0)
         error_root(sscanf(argv[is+1],"%d",&is)!=1,1,"main [check1.c]",
                    "Syntax: check1 [-bc <type>] [-sw <type>]");
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

   set_dfl_parms(bs,Ns);
   start_ranlux(0,123456);
   geometry();

   alloc_ws(Ns+2);
   alloc_wsd(2);
   alloc_wv(3);
   alloc_wvd(2);

   ws=reserve_ws(2);
   wsd=reserve_wsd(2);
   wv=reserve_wv(3);
   wvd=reserve_wvd(2);
   nb=VOLUME/(bs[0]*bs[1]*bs[2]*bs[3]);
   nv=Ns*nb;

   m0[0]=-0.0123;
   m0[1]= 0.0257;

   mu[0]=0.0157;
   mu[1]=0.0239;

   random_ud();
   set_ud_phase();
   random_basis(Ns);

   set_sw_parms(m0[0]);
   set_Aw(mu[0]);
   save_Awd();
   set_flags(UPDATED_UD);
   set_Aw(mu[0]);
   check_Awd(dev);

   if (my_rank==0)
   {
      printf("Compute-erase-compute test: "
             "dev(Ablk)=%.1e, dev(Ahop)=%.1e, dev(Abnd)=%.1e\n\n",
             dev[0],dev[1],dev[2]);
   }

   for (ieo=0;ieo<2;ieo++)
   {
      set_tm_parms(ieo);

      for (im0=0;im0<2;im0++)
      {
         set_sw_parms(m0[im0]);

         for (imu=0;imu<2;imu++)
         {
            set_Aw(mu[imu]);

            random_v(nv,0,wv[0],1.0f);
            assign_v2vd(nv,0,wv[0],wvd[0]);
            Aw_dble(wvd[0],wvd[1]);
            Aw(wv[0],wv[1]);
            assign_v2vd(nv,0,wv[1],wvd[0]);

            zd.re=-1.0;
            zd.im=0.0;
            mulc_vadd_dble(nv,0,wvd[0],wvd[1],zd);
            rqsm=vnorm_square_dble(nv,1,wvd[0]);
            dev[0]=rqsm.q[0];
            rqsm=vnorm_square_dble(nv,1,wvd[1]);
            dev[0]/=rqsm.q[0];

            if (my_rank==0)
               printf("Relative dev[0]iations (ieo=%d,im0=%d,imu=%d): "
                      "Aw_dble vs Aw = %.1e",ieo,im0,imu,sqrt(dev[0]));

            random_vd(nv,0,wvd[0],1.0);
            Aw_dble(wvd[0],wvd[1]);

            sw_term(NO_PTS);
            dfl_vd2sd(wvd[0],wsd[0]);
            Dw_dble(mu[imu],wsd[0],wsd[1]);
            dfl_sd2vd(wsd[1],wvd[0]);
            zd.re=-1.0;
            zd.im=0.0;
            mulc_vadd_dble(nv,0,wvd[0],wvd[1],zd);
            rqsm=vnorm_square_dble(nv,1,wvd[0]);
            dev[0]=rqsm.q[0];
            rqsm=vnorm_square_dble(nv,1,wvd[1]);
            dev[0]/=rqsm.q[0];

            if (my_rank==0)
               printf(", Aw_dble vs Dw_dble = %.1e",sqrt(dev[0]));

            random_v(nv,0,wv[0],1.0f);
            Aw(wv[0],wv[1]);

            assign_ud2u();
            assign_swd2sw();
            dfl_v2s(wv[0],ws[0]);
            Dw((float)(mu[imu]),ws[0],ws[1]);
            dfl_s2v(ws[1],wv[2]);

            z.re=-1.0f;
            z.im=0.0f;
            mulc_vadd(nv,0,wv[2],wv[1],z);
            dev[0]=(double)(vnorm_square(nv,1,wv[2])/vnorm_square(nv,1,wv[1]));

            if (my_rank==0)
               printf(", Aw vs Dw = %.1e\n",sqrt(dev[0]));
         }
      }
   }

   if (my_rank==0)
   {
      printf("\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
