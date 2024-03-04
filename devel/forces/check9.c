
/*******************************************************************************
*
* File check9.c
*
* Copyright (C) 2012-2016, 2018, 2020, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of force3() and action3().
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
#include "ratfcts.h"
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

   read_rat_parms(0);
   read_sap_parms("SAP",0x1);
   read_dfl_parms("Deflation subspace");
   read_dfl_gen_parms("Deflation subspace generation");
   read_dfl_pro_parms("Deflation projection");

   for (isp=0;isp<3;isp++)
      read_solver_parms(isp);
}


static void alloc_wspace(void)
{
   int isp,ns,nsd,nv,nvd,np,n;
   solver_parms_t sp;
   rat_parms_t rp;
   dfl_parms_t dfl;
   dfl_pro_parms_t dpr;

   ns=0;
   nsd=0;
   nv=0;
   nvd=0;

   for (isp=0;isp<3;isp++)
   {
      sp=solver_parms(isp);

      if (sp.solver==MSCG)
      {
         rp=rat_parms(0);
         np=(rp.degree/3)+1;

         if (np==1)
            n=np+6;
         else
            n=2*np+4;

         if (nsd<n)
            nsd=n;
      }
      else if (sp.solver==SAP_GCR)
      {
         if (ns<2*sp.nkv)
            ns=2*sp.nkv;
         if (nsd<5)
            nsd=5;
      }
      else if (sp.solver==DFL_SAP_GCR)
      {
         if (ns<(2*sp.nkv+2))
            ns=2*sp.nkv+2;
         if (nsd<5)
            nsd=5;

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
   alloc_wsd(nsd+3);
   if (nv)
      alloc_wv(nv);
   if (nvd)
      alloc_wvd(nvd);
}


static double check_rotpf(int *irat,int ipf,int isp,int *status)
{
   int i,np;
   double nrm,dev,*mu,*nu;
   spinor_dble *phi,*psi,*chi,*eta,**wsd;
   ratfct_t rf;
   mdflds_t *mdfs;

   for (i=0;i<3;i++)
      status[i]=0;

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];

   save_ranlux();

   (void)(setpf3(irat,ipf,0,isp,0,status));
   nrm=unorm_dble(VOLUME_TRD/2,3,phi);

   restore_ranlux();
   rotpf3(irat,ipf,isp,1.234,-1.234,status);
   dev=unorm_dble(VOLUME_TRD/2,3,phi)/nrm;

   restore_ranlux();
   (void)(setpf3(irat,ipf,0,isp,0,status));

   rf=ratfct(irat);
   np=rf.np;
   mu=rf.mu;
   nu=rf.nu;

   wsd=reserve_wsd(3);
   psi=wsd[0];
   chi=wsd[1];
   eta=wsd[2];

   assign_sd2sd(VOLUME_TRD/2,2,phi,psi);
   sw_term(ODD_PTS);

   for (i=0;i<np;i++)
   {
      Dwhat_dble(nu[i],psi,chi);
      mulg5_dble(VOLUME_TRD/2,2,chi);
      assign_sd2sd(VOLUME_TRD/2,2,chi,psi);
   }

   nrm=unorm_dble(VOLUME_TRD/2,3,psi);

   restore_ranlux();
   (void)(setpf4(mu[0],ipf,0,0));
   assign_sd2sd(VOLUME_TRD/2,2,phi,eta);

   for (i=1;i<np;i++)
   {
      Dwhat_dble(mu[i],eta,chi);
      mulg5_dble(VOLUME_TRD/2,2,chi);
      assign_sd2sd(VOLUME_TRD/2,2,chi,eta);
   }

   mulr_spinor_add_dble(VOLUME_TRD/2,2,psi,eta,-1.0);
   dev+=unorm_dble(VOLUME_TRD/2,3,psi)/nrm;

   release_wsd();

   return dev;
}


static qflt dSdt(int *irat,int ipf,int isw,int isp,int *status)
{
   mdflds_t *mdfs;

   mdfs=mdflds();
   set_frc2zero();
   force3(irat,ipf,isw,isp,1.2345,status);
   check_active((*mdfs).frc);

   return scalar_prod_alg(4*VOLUME_TRD,3,(*mdfs).mom,(*mdfs).frc);
}


int main(int argc,char *argv[])
{
   int my_rank,bc,is,irat[3],ifail[1],*status;
   int isw,isp,isap,idfl;
   double dev,eps,*qact[1];
   double dev_act,dev_frc,sig_loss;
   qflt dsdt,act0,act1,act;
   rat_parms_t rp;
   FILE *flog=NULL,*fin=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check9.log","w",stdout);
      fin=freopen("check9.in","r",stdin);

      printf("\n");
      printf("Check of force3() and action3()\n");
      printf("-------------------------------\n\n");

      print_lattice_sizes();

      bc=find_opt(argc,argv,"-bc");
      is=find_opt(argc,argv,"-sw");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check9.c]",
                    "Syntax: check9 [-bc <type>] [-sw <type>]");

      if (is!=0)
         error_root(sscanf(argv[is+1],"%d",&is)!=1,1,"main [check9.c]",
                    "Syntax: check9 [-bc <type>] [-sw <type>]");
   }

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&is,1,MPI_INT,0,MPI_COMM_WORLD);

   check_machine();
   read_run_parms(bc,is);
   if (my_rank==0)
      fclose(fin);

   print_rat_parms();
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
   rp=rat_parms(0);
   irat[0]=0;

   for (isw=0;isw<(1+(is==0));isw++)
   {
      for (isp=0;isp<3;isp++)
      {
         if (isp==0)
         {
            irat[1]=0;
            irat[2]=rp.degree/3;
            eps=1.0e-4;
         }
         else if (isp==1)
         {
            irat[1]=rp.degree/3+1;
            irat[2]=(2*rp.degree)/3;
            eps=2.0e-4;
         }
         else
         {
            irat[1]=(2*rp.degree)/3+1;
            irat[2]=rp.degree-1;
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
               error_root(1,1,"main [check9.c]","dfl_modes() failed");
            }
         }

         if (isw==0)
            dev=check_rotpf(irat,0,isp,status);
         else
            dev=0.0;
         act0=setpf3(irat,0,isw,isp,0,status);
         act1=action3(irat,0,isw,isp,0,status);
         act.q[0]=-act1.q[0];
         act.q[1]=-act1.q[1];
         add_qflt(act0.q,act.q,act.q);
         qact[0]=act.q;
         global_qsum(1,qact,qact);
         dev_act=act.q[0];

         dsdt=dSdt(irat,0,isw,isp,status);

         if (my_rank==0)
         {
            printf("Solver number %d, poles %d,..,%d",
                   isp,irat[1],irat[2]);

            if (is==0)
            {
               if (isw)
                  printf(", det(D_oo) included\n");
               else
                  printf(", det(D_oo) omitted\n");
            }
            else
               printf("\n");

            if (isp==0)
               print_std_status("tmcgm",NULL,status);
            else if (isp==1)
               print_std_status("sap_gcr","sap_gcr",status);
            else
               print_std_status("dfl_sap_gcr","dfl_sap_gcr",status);

            printf("Absolute action difference |setpf3-action3| = %.1e\n",
                   fabs(dev_act));
            if (isw==0)
               printf("Check of rotpf3 = %.1e\n",dev);
            fflush(flog);
         }

         rot_ud(eps);
         act0=action3(irat,0,isw,isp,1,status);
         scl_qflt(2.0/3.0,act0.q);
         rot_ud(-eps);

         rot_ud(-eps);
         act1=action3(irat,0,isw,isp,1,status);
         scl_qflt(-2.0/3.0,act1.q);
         rot_ud(eps);

         rot_ud(2.0*eps);
         act=action3(irat,0,isw,isp,1,status);
         scl_qflt(-1.0/12.0,act.q);
         add_qflt(act0.q,act.q,act0.q);
         rot_ud(-2.0*eps);

         rot_ud(-2.0*eps);
         act=action3(irat,0,isw,isp,1,status);
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
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
