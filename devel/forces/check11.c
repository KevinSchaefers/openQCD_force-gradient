
/*******************************************************************************
*
* File check11.c
*
* Copyright (C) 2012-2018, 2020, 2021 Martin Luescher, Stefan Schaefer
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of force5() and action5().
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
#include "dirac.h"
#include "mdflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dfl.h"
#include "forces.h"
#include "devfcts.h"
#include "global.h"


static void read_run_parms(int bc,int is)
{
   int isp,idmy;
   double rdmy,chi[2],chi_prime[2],theta[3];

   set_lat_parms(5.5,1.0,0,NULL,is,1.782);

   chi[0]=0.123;
   chi[1]=-0.534;
   chi_prime[0]=0.912;
   chi_prime[1]=0.078;
   theta[0]=0.38;
   theta[1]=-1.25;
   theta[2]=0.54;
   set_bc_parms(bc,1.0,1.0,0.953,1.203,chi,chi_prime,theta);

   idmy=0;
   rdmy=0.0;
   set_action_parms(0,ACF_TM1,0,0,NULL,&idmy,&idmy);
   set_hmc_parms(1,&idmy,1,1,&rdmy,1,1.0);

   read_sap_parms("SAP",0x1);
   read_dfl_parms("Deflation subspace");
   read_dfl_gen_parms("Deflation subspace generation");
   read_dfl_pro_parms("Deflation projection");

   for (isp=0;isp<3;isp++)
      read_solver_parms(isp);
}


static void alloc_wspace(void)
{
   int isp,ns,nsd,nv,nvd;
   solver_parms_t sp;
   dfl_parms_t dfl;
   dfl_pro_parms_t dpr;

   ns=0;
   nsd=0;
   nv=0;
   nvd=0;

   for (isp=0;isp<3;isp++)
   {
      sp=solver_parms(isp);

      if (sp.solver==CGNE)
      {
         if (ns<5)
            ns=5;
         if (nsd<3)
            nsd=3;
      }
      else if (sp.solver==SAP_GCR)
      {
         if (ns<2*sp.nkv)
            ns=2*sp.nkv;
         if (nsd<2)
            nsd=2;
      }
      else if (sp.solver==DFL_SAP_GCR)
      {
         if (ns<(2*sp.nkv+2))
            ns=2*sp.nkv+2;
         if (nsd<2)
            nsd=2;

         dfl=dfl_parms();
         dpr=dfl_pro_parms();

         if (ns<(dfl.Ns+2))
            ns=dfl.Ns+2;
         if (nv<(2*dpr.nmx_gcr+3))
            nv=2*dpr.nmx_gcr+3;
         if (nvd<(2*dpr.nkv+4))
            nvd=2*dpr.nkv+4;
      }
   }

   if (ns)
      alloc_ws(ns);
   alloc_wsd(nsd+2);
   if (nv)
      alloc_wv(nv);
   if (nvd)
      alloc_wvd(nvd);
}


static double check_rotpf(double mu0,double mu1,int ipf,int isp,int *status)
{
   double nrm,dev;
   spinor_dble *phi,*psi,*chi,**wsd;
   mdflds_t *mdfs;

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];

   save_ranlux();
   (void)(setpf5(mu0,mu1,ipf,isp,0,status));
   nrm=unorm_dble(VOLUME_TRD/2,3,phi);

   restore_ranlux();
   rotpf5(mu0,mu1,ipf,0,isp,0,1.234,-1.234,status);
   dev=unorm_dble(VOLUME_TRD/2,3,phi)/nrm;

   restore_ranlux();
   (void)(setpf5(mu0,mu1,ipf,isp,0,status));

   wsd=reserve_wsd(2);
   psi=wsd[0];
   chi=wsd[1];
   assign_sd2sd(VOLUME_TRD/2,2,phi,psi);
   sw_term(ODD_PTS);
   Dwhat_dble(mu1,psi,chi);
   mulg5_dble(VOLUME_TRD/2,2,chi);

   restore_ranlux();
   (void)(setpf4(mu0,ipf,0,0));
   mulr_spinor_add_dble(VOLUME_TRD/2,2,chi,phi,-1.0);
   dev+=unorm_dble(VOLUME_TRD/2,3,chi)/nrm;

   release_wsd();

   return dev;
}


static qflt dSdt(double mu0,double mu1,int ipf,int isp,int *status)
{
   mdflds_t *mdfs;

   mdfs=mdflds();
   set_frc2zero();
   force5(mu0,mu1,ipf,isp,0,1.2345,status);
   check_active((*mdfs).frc);

   return scalar_prod_alg(4*VOLUME_TRD,3,(*mdfs).mom,(*mdfs).frc);
}


int main(int argc,char *argv[])
{
   int my_rank,bc,is,ifail[1],*status;
   int isp,isap,idfl;
   double mu0,mu1,dev,eps,*qact[1];
   double dev_act[2],dev_frc,sig_loss,rdmy;
   qflt dsdt,act0,act1,act;
   FILE *flog=NULL,*fin=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check11.log","w",stdout);
      fin=freopen("check6.in","r",stdin);

      printf("\n");
      printf("Check of force5() and action5()\n");
      printf("-------------------------------\n\n");

      print_lattice_sizes();

      bc=find_opt(argc,argv,"-bc");
      is=find_opt(argc,argv,"-sw");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check11.c]",
                    "Syntax: check11 [-bc <type>] [-sw <type>]");

      if (is!=0)
         error_root(sscanf(argv[is+1],"%d",&is)!=1,1,"main [check11.c]",
                    "Syntax: check11 [-bc <type>] [-sw <type>]");
   }

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&is,1,MPI_INT,0,MPI_COMM_WORLD);

   check_machine();
   read_run_parms(bc,is);
   if (my_rank==0)
      fclose(fin);

   print_lat_parms(0x2);
   print_bc_parms(0x2);
   print_solver_parms(&isap,&idfl);
   print_sap_parms(0x1);
   print_dfl_parms(0x0);

   start_ranlux(0,1245);
   geometry();
   alloc_wspace();
   status=alloc_std_status();

   set_sw_parms(-0.0123);

   for (isp=0;isp<3;isp++)
   {
      if (isp==0)
      {
         mu0=1.0;
         mu1=1.5;
         eps=1.0e-4;
      }
      else if (isp==1)
      {
         mu0=0.1;
         mu1=0.25;
         eps=2.0e-4;
      }
      else
      {
         mu0=0.01;
         mu1=0.02;
         eps=3.0e-4;
      }

      random_ud();
      set_ud_phase();
      random_mom();

      if (isp==2)
      {
         dfl_modes(ifail,status);

         if (ifail[0]<0)
         {
            print_status("dfl_modes",ifail,status);
            error_root(1,1,"main [check11.c]","dfl_modes() failed");
         }
      }

      dev=check_rotpf(mu0,mu1,0,isp,status);
      act0=setpf5(mu0,mu1,0,isp,0,status);
      act1=action5(mu0,mu1,0,isp,0,0,status);
      act.q[0]=-act1.q[0];
      act.q[1]=-act1.q[1];
      add_qflt(act0.q,act.q,act.q);
      rdmy=fabs(act.q[0]);
      MPI_Reduce(&rdmy,dev_act,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(dev_act,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      qact[0]=act.q;
      global_qsum(1,qact,qact);
      dev_act[1]=act.q[0];

      dsdt=dSdt(mu0,mu1,0,isp,status);

      if (my_rank==0)
      {
         printf("Solver number %d\n",isp);

         if (isp==0)
            print_std_status("tmcg",NULL,status);
         else if (isp==1)
            print_std_status("sap_gcr","sap_gcr",status);
         else
            print_std_status("dfl_sap_gcr","dfl_sap_gcr",status);

         printf("Absolute action difference |setpf5-action5| = %.1e,",
                fabs(dev_act[1]));
         printf(" %.1e (local)\n",dev_act[0]);
         printf("Check of rotpf5 = %.1e\n",dev);
         fflush(flog);
      }

      rot_ud(eps);
      act0=action5(mu0,mu1,0,isp,0,1,status);
      scl_qflt(2.0/3.0,act0.q);
      rot_ud(-eps);

      rot_ud(-eps);
      act1=action5(mu0,mu1,0,isp,0,1,status);
      scl_qflt(-2.0/3.0,act1.q);
      rot_ud(eps);

      rot_ud(2.0*eps);
      act=action5(mu0,mu1,0,isp,0,1,status);
      scl_qflt(-1.0/12.0,act.q);
      add_qflt(act0.q,act.q,act0.q);
      rot_ud(-2.0*eps);

      rot_ud(-2.0*eps);
      act=action5(mu0,mu1,0,isp,0,1,status);
      scl_qflt(1.0/12.0,act.q);
      add_qflt(act1.q,act.q,act1.q);
      rot_ud(2.0*eps);

      add_qflt(act0.q,act1.q,act.q);
      sig_loss=-log10(fabs(act.q[0]/act0.q[0]));
      scl_qflt(-1.2345/eps,act.q);
      add_qflt(dsdt.q,act.q,act.q);
      dev_frc=fabs(act.q[0]/dsdt.q[0]);

      if (my_rank==0)
      {
         printf("Relative deviation of dS/dt = %.2e ",dev_frc);
         printf("[significance loss = %d digits]\n\n",(int)(sig_loss));
         fflush(flog);
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
