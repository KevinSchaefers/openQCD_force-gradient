
/*******************************************************************************
*
* File check12.c
*
* Copyright (C) 2018, 2020, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the numerical accuracy of the calculated actions and forces.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
#include "uflds.h"
#include "sflds.h"
#include "mdflds.h"
#include "linalg.h"
#include "dfl.h"
#include "sw_term.h"
#include "forces.h"
#include "update.h"
#include "global.h"

static struct
{
   array_t *act,*frc,*phi,*dphi;
   array_t *dact0,*dact1,*dfrc;
} arrays;

static int my_rank,first,last,step,nsolv;
static iodat_t iodat[1];
static char nbase[NAME_SIZE],cnfg_file[NAME_SIZE];
static smd_parms_t smd;
static FILE *flog=NULL,*fin=NULL;


static void read_run_parms(void)
{
   int isap,idfl;

   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   read_iodat("Configurations","i",iodat);

   if (my_rank==0)
   {
      read_line("first","%d",&first);
      read_line("last","%d",&last);
      read_line("step","%d",&step);

      error_root((first<1)||(last<first)||(step<1)||((last-first)%step!=0),1,
                 "read_run_parms [check5.c]","Improper configuration range");
   }

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   read_lat_parms("Lattice parameters",0x3);
   read_bc_parms("Boundary conditions",0x3);

   read_smd_parms("Actions",0x1);
   read_all_action_parms();
   read_all_force_parms();
   read_all_solver_parms(&isap,&idfl);

   if (isap)
      read_sap_parms("SAP",0x1);
   if (idfl)
   {
      read_dfl_parms("Deflation subspace");
      read_dfl_gen_parms("Deflation subspace generation");
      read_dfl_pro_parms("Deflation projection");
   }

   smd=smd_parms();
   set_mdint_parms(0,LPFR,1.0,1,smd.nact,smd.iact);
}


static void add_solvers(void)
{
   int i,j,k,nsp,nact,*iact;
   action_parms_t ap;
   force_parms_t fp;
   solver_parms_t sp;

   nact=smd.nact;
   iact=smd.iact;
   nsolv=0;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if ((ap.action==ACF_TM1)||
          (ap.action==ACF_TM1_EO)||
          (ap.action==ACF_TM1_EO_SDET)||
          (ap.action==ACF_TM2)||
          (ap.action==ACF_TM2_EO)||
          (ap.action==ACF_RAT)||
          (ap.action==ACF_RAT_SDET))
      {
         if ((ap.action==ACF_TM2)||(ap.action==ACF_TM2_EO)||
             (ap.action==ACF_RAT)||(ap.action==ACF_RAT_SDET))
            nsp=2;
         else
            nsp=1;

         for (j=0;j<nsp;j++)
         {
            k=ap.isp[j];

            if (k>nsolv)
               nsolv=k;
         }
      }

      fp=force_parms(iact[i]);

      if ((fp.force==FRF_TM1)||
          (fp.force==FRF_TM1_EO)||
          (fp.force==FRF_TM1_EO_SDET)||
          (fp.force==FRF_TM2)||
          (fp.force==FRF_TM2_EO)||
          (fp.force==FRF_RAT)||
          (fp.force==FRF_RAT_SDET))
      {
         j=fp.isp[0];

         if (j>nsolv)
            nsolv=j;
      }
   }

   nsolv+=1;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if ((ap.action==ACF_TM1)||
          (ap.action==ACF_TM1_EO)||
          (ap.action==ACF_TM1_EO_SDET)||
          (ap.action==ACF_TM2)||
          (ap.action==ACF_TM2_EO)||
          (ap.action==ACF_RAT)||
          (ap.action==ACF_RAT_SDET))
      {
         if ((ap.action==ACF_TM2)||(ap.action==ACF_TM2_EO)||
             (ap.action==ACF_RAT)||(ap.action==ACF_RAT_SDET))
            nsp=2;
         else
            nsp=1;

         for (j=0;j<nsp;j++)
         {
            k=ap.isp[j];
            sp=solver_parms(k+nsolv);

            if (sp.solver==SOLVERS)
            {
               sp=solver_parms(k);
               set_solver_parms(k+nsolv,sp.solver,sp.nkv,sp.isolv,
                                sp.nmr,sp.ncy,sp.nmx,sp.istop,0.05*sp.res);
            }
         }
      }

      fp=force_parms(iact[i]);

      if ((fp.force==FRF_TM1)||
          (fp.force==FRF_TM1_EO)||
          (fp.force==FRF_TM1_EO_SDET)||
          (fp.force==FRF_TM2)||
          (fp.force==FRF_TM2_EO)||
          (fp.force==FRF_RAT)||
          (fp.force==FRF_RAT_SDET))
      {
         j=fp.isp[0];
         sp=solver_parms(j+nsolv);

         if (sp.solver==SOLVERS)
         {
            sp=solver_parms(j);
            set_solver_parms(j+nsolv,sp.solver,sp.nkv,sp.isolv,
                             sp.nmr,sp.ncy,sp.nmx,sp.istop,0.05*sp.res);
         }
      }
   }
}


static void check_files(void)
{
   error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
         "check_files [check5.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);
}


static qflt set_pf(int iact,int irs,int *status,double *nrm)
{
   int k,ipf,vol;
   double *mu;
   qflt act;
   spinor_dble *pf,**wsd;
   mdflds_t *mdfs;
   action_parms_t ap;

   ap=action_parms(iact);
   set_sw_parms(sea_quark_mass(ap.im0));
   ipf=ap.ipf;

   mdfs=mdflds();
   pf=(*mdfs).pf[ipf];
   if ((*mdfs).eo[ipf])
      vol=VOLUME_TRD/2;
   else
      vol=VOLUME_TRD;

   mu=smd.mu;
   act.q[0]=0.0;
   act.q[1]=0.0;
   reset_std_status(status);
   wsd=reserve_wsd(1);

   if (irs==0)
      assign_sd2sd(vol,2,pf,wsd[0]);

   if (ap.action==ACF_TM1)
      act=setpf1(mu[ap.imu[0]],ipf,1);
   else if (ap.action==ACF_TM1_EO)
      act=setpf4(mu[ap.imu[0]],ipf,0,1);
   else if (ap.action==ACF_TM1_EO_SDET)
      act=setpf4(mu[ap.imu[0]],ipf,1,1);
   else
   {
      k=ap.isp[1]+irs*nsolv;

      if (ap.action==ACF_TM2)
         act=setpf2(mu[ap.imu[0]],mu[ap.imu[1]],ipf,k,1,status);
      else if (ap.action==ACF_TM2_EO)
         act=setpf5(mu[ap.imu[0]],mu[ap.imu[1]],ipf,k,1,status);
      else if (ap.action==ACF_RAT)
         act=setpf3(ap.irat,ipf,0,k,1,status);
      else if (ap.action==ACF_RAT_SDET)
         act=setpf3(ap.irat,ipf,1,k,1,status);
   }

   if (irs==0)
   {
      mulr_spinor_add_dble(vol,2,wsd[0],pf,-1.0);
      (*nrm)=unorm_dble(vol,3,wsd[0]);
   }
   else
      (*nrm)=unorm_dble(vol,3,pf);

   release_wsd();

   return act;
}


static qflt get_action(int iact,int irs,int *status)
{
   int k,ipf;
   double *mu;
   qflt act;
   action_parms_t ap;

   ap=action_parms(iact);
   set_sw_parms(sea_quark_mass(ap.im0));
   ipf=ap.ipf;
   k=ap.isp[0]+irs*nsolv;

   mu=smd.mu;
   act.q[0]=0.0;
   act.q[1]=0.0;

   if (ap.action==ACF_TM1)
      act=action1(mu[ap.imu[0]],ipf,k,0,1,status);
   else if (ap.action==ACF_TM1_EO)
      act=action4(mu[ap.imu[0]],ipf,0,k,0,1,status);
   else if (ap.action==ACF_TM1_EO_SDET)
      act=action4(mu[ap.imu[0]],ipf,1,k,0,1,status);
   else if (ap.action==ACF_TM2)
      act=action2(mu[ap.imu[0]],mu[ap.imu[1]],ipf,k,0,1,status);
   else if (ap.action==ACF_TM2_EO)
      act=action5(mu[ap.imu[0]],mu[ap.imu[1]],ipf,k,0,1,status);
   else if (ap.action==ACF_RAT)
      act=action3(ap.irat,ipf,0,k,1,status);
   else if (ap.action==ACF_RAT_SDET)
      act=action3(ap.irat,ipf,1,k,1,status);

   return act;
}


static void set_force(int ifr,int irs,int *status)
{
   int k,ipf;
   double *mu;
   force_parms_t fp;

   fp=force_parms(ifr);
   set_sw_parms(sea_quark_mass(fp.im0));
   ipf=fp.ipf;
   k=fp.isp[0]+irs*nsolv;

   mu=smd.mu;
   set_frc2zero();

   if (fp.force==FRF_TM1)
      force1(mu[fp.imu[0]],ipf,k,0,1.0,status);
   else if (fp.force==FRF_TM1_EO)
      force4(mu[fp.imu[0]],ipf,0,k,0,1.0,status);
   else if (fp.force==FRF_TM1_EO_SDET)
      force4(mu[fp.imu[0]],ipf,1,k,0,1.0,status);
   else if (fp.force==FRF_TM2)
      force2(mu[fp.imu[0]],mu[fp.imu[1]],ipf,k,0,1.0,status);
   else if (fp.force==FRF_TM2_EO)
      force5(mu[fp.imu[0]],mu[fp.imu[1]],ipf,k,0,1.0,status);
   else if (fp.force==FRF_RAT)
      force3(fp.irat,ipf,0,k,1.0,status);
   else if (fp.force==FRF_RAT_SDET)
      force3(fp.irat,ipf,1,k,1.0,status);
}


static void print_status_act(int iact,int *status,double wdt)
{
   action_parms_t ap;
   solver_parms_t sp;

   ap=action_parms(iact);
   sp=solver_parms(ap.isp[0]);

   if (sp.solver==CGNE)
      print_std_status("tmcg",NULL,status);
   else if (sp.solver==MSCG)
      print_std_status("tmcgm",NULL,status);
   else if (sp.solver==SAP_GCR)
      print_std_status("sap_gcr",NULL,status);
   else if (sp.solver==DFL_SAP_GCR)
      print_std_status("dfl_sap_gcr",NULL,status);

   printf("(time = %.2e sec)\n",wdt);
}


static void print_status_pf(int iact,int *status,double wdt)
{
   action_parms_t ap;
   solver_parms_t sp;

   ap=action_parms(iact);
   sp=solver_parms(ap.isp[1]);

   if (sp.solver==CGNE)
      print_std_status("tmcg",NULL,status);
   else if (sp.solver==MSCG)
      print_std_status("tmcgm",NULL,status);
   else if (sp.solver==SAP_GCR)
      print_std_status("sap_gcr",NULL,status);
   else if (sp.solver==DFL_SAP_GCR)
      print_std_status("dfl_sap_gcr",NULL,status);

   printf("(time = %.2e sec)\n",wdt);
}


static void print_status_frc(int ifr,int *status,double wdt)
{
   force_parms_t fp;
   solver_parms_t sp;

   fp=force_parms(ifr);
   sp=solver_parms(fp.isp[0]);

   if (sp.solver==CGNE)
      print_std_status("tmcg",NULL,status);
   else if (sp.solver==MSCG)
      print_std_status("tmcgm",NULL,status);
   else if (sp.solver==SAP_GCR)
      print_std_status("sap_gcr","sap_gcr",status);
   else if (sp.solver==DFL_SAP_GCR)
      print_std_status("dfl_sap_gcr","dfl_sap_gcr",status);

   printf("(time = %.2e sec)\n",wdt);
}


static void alloc_arrays(void)
{
   size_t n[3];

   n[0]=smd.nact+1;
   n[1]=3;
   arrays.act=alloc_array(2,n,sizeof(double),0);
   arrays.frc=alloc_array(2,n,sizeof(double),0);
   arrays.phi=alloc_array(2,n,sizeof(double),0);
   arrays.dphi=alloc_array(2,n,sizeof(double),0);
   arrays.dact0=alloc_array(2,n,sizeof(double),0);

   n[1]=2;
   n[2]=3;
   arrays.dact1=alloc_array(3,n,sizeof(double),0);
   arrays.dfrc=alloc_array(3,n,sizeof(double),0);
}


static void set_val(double v,double *a)
{
   a[0]=v;
   a[1]=v;
   a[2]=v;
}


static void add_val(double v,double *a)
{
   if (v<a[0])
      a[0]=v;
   if (v>a[1])
      a[1]=v;
   a[2]+=v;
}


int main(int argc,char *argv[])
{
   int icnfg,ncnfg,ifail,*status;
   int isap,idfl;
   int n,i,k,nact,*iact;
   int nwud,nwfd,nws,nwv,nwvd;
   double nrm0,nrm1,dev,atot,datot,wt1,wt2;
   double **act,**frc,**phi,**dphi,**dact0;
   double ***dact1,***dfrc;
   qflt qact0,qact1,qact2;
   su3_alg_dble **wfd;
   mdflds_t *mdfs;
   dfl_parms_t dfl;
   action_parms_t ap;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check12.log","w",stdout);
      fin=freopen("check12.in","r",stdin);

      printf("\n");
      printf("Numerical precision of the calculated actions and forces\n");
      printf("--------------------------------------------------------\n\n");

      print_lattice_sizes();
   }

   read_run_parms();
   add_solvers();

   if (my_rank==0)
      fclose(fin);

   if (my_rank==0)
   {
      print_lat_parms(0x2);
      print_bc_parms(0x3);

      printf("Actions =");
      for (i=0;i<smd.nact;i++)
         printf(" %d",smd.iact[i]);
      printf("\n");
      printf("npf = %d\n",smd.npf);

      if (smd.nmu>0)
      {
         printf("mu =");

         for (i=0;i<smd.nmu;i++)
         {
            n=fdigits(smd.mu[i]);
            printf(" %.*f",IMAX(n,1),smd.mu[i]);
         }
         printf("\n");
      }

      printf("\n");
      print_action_parms();
      print_rat_parms();
      print_mdint_parms();
      print_force_parms(0x0);
      print_solver_parms(&isap,&idfl);

      if (isap)
         print_sap_parms(0x0);

      if (idfl)
         print_dfl_parms(0x0);

      print_iodat("i",iodat);

      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   check_machine();
   start_ranlux(0,1234);
   geometry();
   check_files();

   smd_wsize(&nwud,&nwfd,&nws,&nwv,&nwvd);
   alloc_wud(nwud);
   alloc_wfd(nwfd+2);
   alloc_ws(nws+2);
   wsd_uses_ws();
   alloc_wv(nwv);
   alloc_wvd(nwvd);
   status=alloc_std_status();

   wfd=reserve_wfd(2);

   mdfs=mdflds();
   dfl=dfl_parms();
   nact=smd.nact;
   iact=smd.iact;

   alloc_arrays();
   act=(double**)(arrays.act[0].a);
   frc=(double**)(arrays.frc[0].a);
   phi=(double**)(arrays.phi[0].a);
   dphi=(double**)(arrays.dphi[0].a);
   dact0=(double**)(arrays.dact0[0].a);
   dact1=(double***)(arrays.dact1[0].a);
   dfrc=(double***)(arrays.dfrc[0].a);

   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      if (my_rank==0)
      {
         printf("Configuration no %d\n\n",icnfg);
         fflush(flog);
      }

      sprintf(cnfg_file,"%sn%d",nbase,icnfg);
      read_flds(iodat,cnfg_file,0x0,0x1);
      set_ud_phase();

      if (dfl.Ns)
      {
         dfl_modes(&ifail,status);

         if (my_rank==0)
         {
            printf("Deflation subspace generation\n");
            print_status("dfl_modes",&ifail,status);

            error_root(ifail<0,1,"main [check12.c]",
                       "Deflation subspace generation failed");
         }
      }

      random_mom();
      qact0=momentum_action(1);
      qact1=action0(1);
      atot=qact0.q[0]+qact1.q[0];
      datot=0.0;

      force0(1.0);
      assign_alg2alg(4*VOLUME_TRD,2,(*mdfs).frc,wfd[0]);
      nrm0=unorm_alg(4*VOLUME_TRD,3,wfd[0]);
      set_alg2zero(4*VOLUME_TRD,2,wfd[1]);

      if (my_rank==0)
      {
         printf("Momentum action:\n");
         printf("act = %.2e\n",qact0.q[0]);
         printf("Gauge action:\n");
         printf("act = %.2e\n",qact1.q[0]);
         printf("Gauge force:\n");
         printf("|frc|_oo = %.2e\n\n",nrm0);
         fflush(flog);
      }

      for (i=0;i<nact;i++)
      {
         k=iact[i];
         ap=action_parms(k);

         if (ap.action!=ACG)
         {
            if (my_rank==0)
            {
               printf("Action, pseudo-fermion field and force no %d\n",k);
               printf("-------------------------------------------");

               if (k<10)
                  printf("\n\n");
               else
                  printf("-\n\n");
            }

            if ((ap.action!=ACF_TM1)&&(ap.action!=ACF_TM1_EO)&&
                (ap.action!=ACF_TM1_EO_SDET))
               save_ranlux();

            MPI_Barrier(MPI_COMM_WORLD);
            wt1=MPI_Wtime();
            qact0=set_pf(k,1,status,&nrm0);
            MPI_Barrier(MPI_COMM_WORLD);
            wt2=MPI_Wtime();

            if (my_rank==0)
            {
               if ((ap.action!=ACF_TM1)&&(ap.action!=ACF_TM1_EO)&&
                   (ap.action!=ACF_TM1_EO_SDET))
               {
                  printf("Generation of the pseudo-fermion (action no %d):\n",
                         k);
                  printf("Precise solve: ");
                  print_status_pf(k,status,wt2-wt1);
               }
            }

            if (icnfg==first)
               set_val(nrm0,phi[i]);
            else
               add_val(nrm0,phi[i]);

            MPI_Barrier(MPI_COMM_WORLD);
            wt1=MPI_Wtime();
            qact1=get_action(k,1,status);
            MPI_Barrier(MPI_COMM_WORLD);
            wt2=MPI_Wtime();

            if (my_rank==0)
            {
               printf("Action no %d:\n",k);
               printf("Precise solve: ");
               print_status_act(k,status,wt2-wt1);
            }

            if (icnfg==first)
               set_val(qact1.q[0],act[i]);
            else
               add_val(qact1.q[0],act[i]);

            MPI_Barrier(MPI_COMM_WORLD);
            wt1=MPI_Wtime();
            set_force(k,1,status);
            MPI_Barrier(MPI_COMM_WORLD);
            wt2=MPI_Wtime();

            if (my_rank==0)
            {
               printf("Force no %d:\n",k);
               printf("Precise solve: ");
               print_status_frc(k,status,wt2-wt1);
               printf("\n");
            }

            muladd_assign_alg(4*VOLUME_TRD,2,1.0,(*mdfs).frc,wfd[0]);
            assign_alg2alg(4*VOLUME_TRD,2,(*mdfs).frc,(*mdfs).mom);
            nrm1=unorm_alg(4*VOLUME_TRD,3,(*mdfs).frc);

            if (icnfg==first)
               set_val(nrm1,frc[i]);
            else
               add_val(nrm1,frc[i]);

            MPI_Barrier(MPI_COMM_WORLD);
            wt1=MPI_Wtime();
            qact2=get_action(k,0,status);
            MPI_Barrier(MPI_COMM_WORLD);
            wt2=MPI_Wtime();

            qact2.q[0]=-qact2.q[0];
            qact2.q[1]=-qact2.q[1];
            add_qflt(qact1.q,qact2.q,qact2.q);
            dev=fabs(qact2.q[0]);
            atot+=qact1.q[0];
            datot+=dev;

            if (my_rank==0)
            {
               printf("Action no %d:\n",k);
               printf("Less precise solve: ");
               print_status_act(k,status,wt2-wt1);
               printf("act1 = %.2e, |dact1| = %.2e, |dact1|/act1 = %.2e\n",
                      qact1.q[0],dev,dev/qact1.q[0]);
               fflush(flog);
            }

            if (icnfg==first)
               set_val(dev,dact1[i][0]);
            else
               add_val(dev,dact1[i][0]);

            MPI_Barrier(MPI_COMM_WORLD);
            wt1=MPI_Wtime();
            set_force(k,0,status);
            MPI_Barrier(MPI_COMM_WORLD);
            wt2=MPI_Wtime();

            muladd_assign_alg(4*VOLUME_TRD,2,-1.0,(*mdfs).mom,(*mdfs).frc);
            muladd_assign_alg(4*VOLUME_TRD,2,1.0,(*mdfs).frc,wfd[1]);
            dev=unorm_alg(4*VOLUME_TRD,3,(*mdfs).frc);

            if (my_rank==0)
            {
               printf("Force no %d:\n",k);
               printf("Less precise solve: ");
               print_status_frc(k,status,wt2-wt1);
               printf("|frc|_oo = %.2e, |dfrc|_oo = %.2e, "
                      "|dfrc|_oo/|frc|_oo = %.2e\n",nrm1,dev,dev/nrm1);
               fflush(flog);
            }

            if (icnfg==first)
               set_val(dev,dfrc[i][0]);
            else
               add_val(dev,dfrc[i][0]);

            if ((ap.action!=ACF_TM1)&&(ap.action!=ACF_TM1_EO)&&
                (ap.action!=ACF_TM1_EO_SDET))
            {
               restore_ranlux();

               MPI_Barrier(MPI_COMM_WORLD);
               wt1=MPI_Wtime();
               qact2=set_pf(k,0,status,&dev);
               MPI_Barrier(MPI_COMM_WORLD);
               wt2=MPI_Wtime();

               if (my_rank==0)
               {
                  printf("Generation of the pseudo-fermion (action no %d):\n",k);
                  printf("Less precise solve: ");
                  print_status_pf(k,status,wt2-wt1);
                  printf("|phi|_oo = %.2e, |dphi|_oo = %.2e, "
                         "|dphi|_oo/|phi|_oo = %.2e\n",nrm0,dev,dev/nrm0);
                  fflush(flog);
               }

               if (icnfg==first)
                  set_val(dev,dphi[i]);
               else
                  add_val(dev,dphi[i]);

               qact2.q[0]=-qact2.q[0];
               qact2.q[1]=-qact2.q[1];
               add_qflt(qact0.q,qact2.q,qact2.q);
               dev=fabs(qact2.q[0]);

               if (my_rank==0)
               {
                  printf("act0 = %.2e, |dact0| = %.2e, |dact0|/act0 = %.2e\n\n",
                         qact0.q[0],dev,dev/qact0.q[0]);
                  fflush(flog);
               }

               if (icnfg==first)
                  set_val(dev,dact0[i]);
               else
                  add_val(dev,dact0[i]);

               MPI_Barrier(MPI_COMM_WORLD);
               wt1=MPI_Wtime();
               qact2=get_action(k,1,status);
               MPI_Barrier(MPI_COMM_WORLD);
               wt2=MPI_Wtime();

               qact2.q[0]=-qact2.q[0];
               qact2.q[1]=-qact2.q[1];
               add_qflt(qact1.q,qact2.q,qact2.q);
               dev=fabs(qact2.q[0]);

               if (my_rank==0)
               {
                  printf("Action no %d:\n",k);
                  printf("Precise solve, less precise phi: ");
                  print_status_act(k,status,wt2-wt1);
                  printf("act1 = %.2e, |dact1| = %.2e, |dact1|/act1 = %.2e\n\n",
                         qact1.q[0],dev,dev/qact1.q[0]);
                  fflush(flog);
               }

               if (icnfg==first)
                  set_val(dev,dact1[i][1]);
               else
                  add_val(dev,dact1[i][1]);

               MPI_Barrier(MPI_COMM_WORLD);
               wt1=MPI_Wtime();
               set_force(k,1,status);
               MPI_Barrier(MPI_COMM_WORLD);
               wt2=MPI_Wtime();

               muladd_assign_alg(4*VOLUME_TRD,2,-1.0,(*mdfs).mom,(*mdfs).frc);
               dev=unorm_alg(4*VOLUME_TRD,3,(*mdfs).frc);

               if (my_rank==0)
               {
                  printf("Force no %d:\n",k);
                  printf("Precise solve, less precise phi: ");
                  print_status_frc(k,status,wt2-wt1);
                  printf("|frc|_oo = %.2e, |dfrc|_oo = %.2e, "
                         "|dfrc|_oo/|frc|_oo = %.2e\n\n",nrm1,dev,dev/nrm1);
                  fflush(flog);
               }

               if (icnfg==first)
                  set_val(dev,dfrc[i][1]);
               else
                  add_val(dev,dfrc[i][1]);
            }
            else if (my_rank==0)
               printf("\n");
         }
      }

      qact0=scalar_prod_alg(4*VOLUME_TRD,3,wfd[0],wfd[1]);
      dev=fabs(qact0.q[0]);

      if (icnfg==first)
      {
         set_val(atot,act[nact]);
         set_val(datot,dact1[nact][0]);
         set_val(dev,dfrc[nact][0]);
      }
      else
      {
         add_val(atot,act[nact]);
         add_val(datot,dact1[nact][0]);
         add_val(dev,dfrc[nact][0]);
      }
   }

   if (my_rank==0)
   {
      printf("Test summary\n");
      printf("------------\n\n");

      if (last==first)
         ncnfg=1;
      else
         ncnfg=(last-first)/step+1;

      printf("Processed %d configurations\n",ncnfg);
      printf("The measured minimal, maximal and average values are:\n\n");

      for (i=0;i<nact;i++)
      {
         k=iact[i];
         ap=action_parms(k);

         if (ap.action!=ACG)
         {
            printf("Pseudo-fermion (action %2d): %.2e, %.2e, %.2e; ",
                   k,phi[i][0],phi[i][1],phi[i][2]/(double)(ncnfg));

            if ((ap.action!=ACF_TM1)&&(ap.action!=ACF_TM1_EO)&&
                (ap.action!=ACF_TM1_EO_SDET))
            {
               printf("Deviations: %.1e, %.1e, %.1e (field); ",
                      dphi[i][0],dphi[i][1],dphi[i][2]/(double)(ncnfg));
               printf("%.1e, %.1e, %.1e (action)\n",
                      dact0[i][0],dact0[i][1],dact0[i][2]/(double)(ncnfg));
            }
            else
               printf("\n");
         }
      }

      printf("\n");

      for (i=0;i<nact;i++)
      {
         k=iact[i];
         ap=action_parms(k);

         if (ap.action!=ACG)
         {
            printf("Action %2d: %.2e, %.2e, %.2e; ",
                   k,act[i][0],act[i][1],act[i][2]/(double)(ncnfg));
            printf("Deviations: %.1e, %.1e, %.1e (solver); ",dact1[i][0][0],
                   dact1[i][0][1],dact1[i][0][2]/(double)(ncnfg));

            if ((ap.action!=ACF_TM1)&&(ap.action!=ACF_TM1_EO)&&
                (ap.action!=ACF_TM1_EO_SDET))
            {
               printf("%.1e, %.1e, %.1e (phi)\n",dact1[i][1][0],dact1[i][1][1],
                      dact1[i][1][2]/(double)(ncnfg));
            }
            else
               printf("\n");
         }
      }

      printf("\n");

      for (i=0;i<nact;i++)
      {
         k=iact[i];
         ap=action_parms(k);

         if (ap.action!=ACG)
         {
            printf("Force %2d: %.2e, %.2e, %.2e; ",
                   k,frc[i][0],frc[i][1],frc[i][2]/(double)(ncnfg));
            printf("Deviation: %.1e, %.1e, %.1e (solver); ",
                   dfrc[i][0][0],dfrc[i][0][1],dfrc[i][0][2]/(double)(ncnfg));

            if ((ap.action!=ACF_TM1)&&(ap.action!=ACF_TM1_EO)&&
                (ap.action!=ACF_TM1_EO_SDET))
            {
               printf("Deviation: %.1e, %.1e, %.1e (phi)\n",dfrc[i][1][0],
                      dfrc[i][1][1],dfrc[i][1][2]/(double)(ncnfg));
            }
            else
               printf("\n");
         }
      }

      printf("\n");
      printf("Total action: %.12e, %.12e, %.12e\n",
             act[nact][0],act[nact][1],act[nact][2]/(double)(ncnfg));
      printf("Total absolute deviation: %.2e, %.2e, %.2e\n",
             dact1[nact][0][0],dact1[nact][0][1],
             dact1[nact][0][2]/(double)(ncnfg));
      printf("Accumulated field-induced deviation |(frc,dfrc)|: "
             "%.2e, %.2e, %.2e\n\n",
             dfrc[nact][0][0],dfrc[nact][0][1],
             dfrc[nact][0][2]/(double)(ncnfg));
      fclose(flog);
   }

   release_wfd();

   MPI_Finalize();
   exit(0);
}
