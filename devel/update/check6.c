
/*******************************************************************************
*
* File check6.c
*
* Copyright (C) 2012-2016, 2018, 2022 Stefan Schaefer, Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Comparison of rwtm*eo() with action4().
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "sflds.h"
#include "linalg.h"
#include "dirac.h"
#include "sap.h"
#include "dfl.h"
#include "forces.h"
#include "update.h"
#include "global.h"


static void read_run_parms(void)
{
   int isp;

   for (isp=0;isp<3;isp++)
      read_solver_parms(isp);

   read_sap_parms("SAP",0x1);
   read_dfl_parms("Deflation subspace");
   read_dfl_pro_parms("Deflation projection");
   read_dfl_gen_parms("Deflation subspace generation");
}


static qflt random_pf(void)
{
   qflt nrm;
   spinor_dble *phi,**wsd;
   mdflds_t *mdfs;

   wsd=reserve_wsd(1);
   phi=wsd[0];
   random_sd(VOLUME_TRD/2,2,phi,1.0);
   bnd_sd2zero(EVEN_PTS,phi);
   nrm=norm_square_dble(VOLUME_TRD/2,3,phi);

   mdfs=mdflds();
   assign_sd2sd(VOLUME_TRD/2,2,phi,(*mdfs).pf[0]);
   release_wsd();

   return nrm;
}


static void divide_pf(double mu,int isp,int *status)
{
   int ifail[2];
   spinor_dble *phi,**wsd;
   spinor_dble *chi,**rsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);

   wsd=reserve_wsd(1);
   phi=wsd[0];
   mdfs=mdflds();
   assign_sd2sd(VOLUME_TRD/2,2,(*mdfs).pf[0],phi);
   sp=solver_parms(isp);

   if (sp.solver==CGNE)
   {
      tmcgeo(sp.nmx,sp.istop,sp.res,mu,phi,phi,ifail,status);

      if (ifail[0]<0)
      {
         print_status("tmcgeo",ifail,status);
         error_root(1,1,"divide_pf [check6.c]","CGNE solver failed "
                    "(mu=%.2e, parameter set no %d)",mu,isp);
      }

      rsd=reserve_wsd(1);
      chi=rsd[0];
      assign_sd2sd(VOLUME_TRD/2,2,phi,chi);
      Dwhat_dble(-mu,chi,phi);
      mulg5_dble(VOLUME_TRD/2,2,phi);
      release_wsd();
   }
   else if (sp.solver==SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      mulg5_dble(VOLUME_TRD/2,2,phi);
      set_sd2zero(VOLUME_TRD/2,2,phi+VOLUME/2);
      sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu,phi,phi,ifail,status);

      if (ifail[0]<0)
      {
         print_status("sap_gcr",ifail,status);
         error_root(1,1,"divide_pf [check6.c]","SAP_GCR solver failed "
                    "(mu=%.2e, parameter set no %d)",mu,isp);
      }
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      mulg5_dble(VOLUME_TRD/2,2,phi);
      set_sd2zero(VOLUME_TRD/2,2,phi+VOLUME/2);
      dfl_sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu,phi,phi,ifail,status);

      if ((ifail[0]<-2)||(ifail[1]<0))
      {
         print_status("dfl_sap_gcr",ifail,status);
         error_root(1,1,"divide_pf [check6.c]","DFL_SAP_GCR solver failed "
                    "(mu=%.2e, parameter set no %d)",mu,isp);
      }
   }

   assign_sd2sd(VOLUME_TRD/2,2,phi,(*mdfs).pf[0]);
   release_wsd();
}


int main(int argc,char *argv[])
{
   int my_rank,bc,irw,isp,ifail[1],*status;
   int n,idmy,isap,idfl;
   double chi[2],chi_prime[2],theta[3];
   double mu1,mu2,mu1sq,mu2sq;
   double da,ds,damx,dsmx,rdmy;
   qflt act0,act1,sqn0,sqn1;
   solver_parms_t sp;
   dfl_parms_t dfl;
   dfl_pro_parms_t dpr;
   FILE *flog=NULL,*fin=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check6.log","w",stdout);
      fin=freopen("check6.in","r",stdin);

      printf("\n");
      printf("Comparison of rwtm*eo() with action4()\n");
      printf("--------------------------------------\n\n");

      print_lattice_sizes();

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check6.c]",
                    "Syntax: check6 [-bc <type>]");
   }

   read_run_parms();

   if (my_rank==0)
      fclose(fin);

   check_machine();
   set_lat_parms(5.5,1.0,0,NULL,0,1.782);
   print_lat_parms(0x3);

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   chi[0]=0.123;
   chi[1]=-0.534;
   chi_prime[0]=0.912;
   chi_prime[1]=0.078;
   theta[0]=0.34;
   theta[1]=-1.25;
   theta[2]=0.58;
   set_bc_parms(bc,1.0,1.0,0.953,1.203,chi,chi_prime,theta);
   print_bc_parms(0x2);

   idmy=0;
   rdmy=0.0;
   set_action_parms(0,ACF_TM1_EO,0,0,NULL,&idmy,&idmy);
   set_hmc_parms(1,&idmy,1,1,&rdmy,1,1.0);

   print_solver_parms(&isap,&idfl);
   print_sap_parms(0x0);
   print_dfl_parms(0x0);

   start_ranlux(0,1245);
   geometry();

   dfl=dfl_parms();
   dpr=dfl_pro_parms();

   n=5;
   if (n<(dfl.Ns+2))
      n=dfl.Ns+2;
   sp=solver_parms(1);
   if (n<(2*sp.nkv+1))
      n=2*sp.nkv+1;
   sp=solver_parms(2);
   if (n<(2*sp.nkv+2))
      n=2*sp.nkv+2;
   alloc_ws(n);
   alloc_wsd(5);
   alloc_wv(2*dpr.nmx_gcr+3);
   alloc_wvd(2*dpr.nkv+4);
   status=alloc_std_status();

   damx=0.0;
   dsmx=0.0;

   for (irw=1;irw<5;irw++)
   {
      for (isp=0;isp<3;isp++)
      {
         if (isp==0)
         {
            set_sw_parms(1.0877);
            if (irw<3)
               mu1=1.0;
            else
               mu1=0.0;
            mu2=1.23;
         }
         else if (isp==1)
         {
            set_sw_parms(0.0877);
            if (irw<3)
               mu1=0.1;
            else
               mu1=0.0;
            mu2=0.123;
         }
         else
         {
            set_sw_parms(-0.0123);
            if (irw<3)
               mu1=0.01;
            else
               mu1=0.0;
            mu2=0.0123;
         }

         mu1sq=mu1*mu1;
         mu2sq=mu2*mu2;

         random_ud();
         set_ud_phase();

         if (isp==2)
         {
            dfl_modes(ifail,status);

            if (ifail[0]<0)
            {
               print_status("dfl_modes",ifail,status);
               error_root(1,1,"main [check6.c]","dfl_modes() failed");
            }
         }

         start_ranlux(0,8910+isp);
         sqn0=random_pf();

         if ((irw&0x1)==1)
         {
            if (my_rank==0)
            {
               printf("Solver number %d, mu1 = %.2e\n",isp,mu1);
               printf("action4(): ");
            }

            act0=action4(mu1,0,0,isp,0,1,status);

            if (isp==0)
               print_std_status("tmcgeo",NULL,status);
            else if (isp==1)
               print_std_status("sap_gcr",NULL,status);
            else
               print_std_status("dfl_sap_gcr2",NULL,status);

            scl_qflt(mu2sq-mu1sq,act0.q);
         }
         else
         {
            if (my_rank==0)
            {
               printf("Solver number %d, mu1 = %.2e, mu2 = %.2e\n",isp,mu1,mu2);
               printf("action4(): ");
            }

            divide_pf(mu1,isp,status);
            act0=action4(mu1,0,0,isp,0,1,status);

            if (isp==0)
               print_std_status("tmcgeo",NULL,status);
            else if (isp==1)
               print_std_status("sap_gcr",NULL,status);
            else
               print_std_status("dfl_sap_gcr2",NULL,status);

            if (my_rank==0)
               printf("action4(): ");

            act1=action4(sqrt(2.0)*mu2,0,0,isp,0,1,status);

            if (isp==0)
               print_std_status("tmcgeo",NULL,status);
            else if (isp==1)
               print_std_status("sap_gcr",NULL,status);
            else
               print_std_status("dfl_sap_gcr2",NULL,status);

            scl_qflt(mu1sq*(mu2sq-mu1sq),act0.q);
            scl_qflt(2.0*mu2sq*mu2sq,act1.q);
            add_qflt(act1.q,act0.q,act0.q);
            scl_qflt((mu2sq-mu1sq)/(2.0*mu2sq-mu1sq),act0.q);
         }

         start_ranlux(0,8910+isp);

         if ((irw&0x1)==1)
            act1=rwtm1eo(mu1,mu2,isp,&sqn1,status);
         else
            act1=rwtm2eo(mu1,mu2,isp,&sqn1,status);

         act1.q[0]=-act1.q[0];
         act1.q[1]=-act1.q[1];
         sqn1.q[0]=-sqn1.q[0];
         sqn1.q[1]=-sqn1.q[1];
         add_qflt(act0.q,act1.q,act1.q);
         add_qflt(sqn0.q,sqn1.q,sqn1.q);

         da=fabs(act1.q[0]/act0.q[0]);
         ds=fabs(sqn1.q[0]/sqn0.q[0]);

         if (da>damx)
            damx=da;
         if (ds>dsmx)
            dsmx=ds;

         if (my_rank==0)
         {
            if ((irw&0x1)==1)
            {
               printf("rwtm1eo(): ");

               if (isp==0)
                  print_std_status("tmcgeo",NULL,status);
               else if (isp==1)
                  print_std_status("sap_gcr",NULL,status);
               else
                  print_std_status("dfl_sap_gcr2",NULL,status);
            }
            else
            {
               printf("rwtm2eo(): ");

               if (isp==0)
                  print_std_status("tmcgeo","tmcg",status);
               else if (isp==1)
                  print_std_status("sap_gcr","sap_gcr",status);
               else
                  print_std_status("dfl_sap_gcr2","dfl_sap_gcr2",status);
            }

            printf("|1-act1/act0| = %.1e, |1-sqn1/sqn0| = %.1e\n\n",da,ds);
         }
      }
   }

   if (my_rank==0)
   {
      printf("max|1-act1/act0| = %.1e, max|1-sqn1/sqn0| = %.1e\n\n",damx,dsmx);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
