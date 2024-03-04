
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2005-2016, 2018, 2022 Martin Luescher, Filippo Palombi
*                                     Stefan Schaefer
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Reversibility of the MD evolution.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "archive.h"
#include "linalg.h"
#include "forces.h"
#include "dfl.h"
#include "update.h"
#include "global.h"

static int my_rank,first,last,step;
static iodat_t iodat[1];
static char nbase[NAME_SIZE],cnfg_file[NAME_SIZE];
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
                 "read_run_parms [check2.c]","Improper configuration range");
   }

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   read_lat_parms("Lattice parameters",0x3);
   read_bc_parms("Boundary conditions",0x3);
   read_hmc_parms("HMC parameters",0x3);

   read_all_mdint_parms();
   read_all_action_parms();
   read_all_force_parms();
   read_all_solver_parms(&isap,&idfl);

   if ((isap)||(idfl))
      read_sap_parms("SAP",0x1);

   if (idfl)
   {
      read_dfl_parms("Deflation subspace");
      read_dfl_pro_parms("Deflation projection");
      read_dfl_gen_parms("Deflation subspace generation");
      read_dfl_upd_parms("Deflation update scheme");
   }
}


static void check_files(void)
{
   error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
         "check_files [check2.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);
}


static void start_hmc(qflt *act0,su3_dble *uold)
{
   int i,n,nact,*iact;
   int ifail[2],status[NSTD_STATUS];
   double *mu;
   dfl_parms_t dfl;
   hmc_parms_t hmc;
   action_parms_t ap;

   hmc=hmc_parms();
   nact=hmc.nact;
   iact=hmc.iact;
   mu=hmc.mu;

   random_mom();
   assign_ud2ud(4*VOLUME_TRD,2,udfld(),uold);

   if (hmc.npf)
   {
      clear_counters();
      reset_chrono(0);
      set_ud_phase();
      dfl=dfl_parms();

      if (dfl.Ns)
      {
         dfl_modes2(ifail,status);

         if ((ifail[0]<-2)||(ifail[1]<0))
         {
            print_status("dfl_modes2",ifail,status);
            error_root(1,1,"start_hmc [check2.c]",
                       "Deflation subspace generation failed");
         }

         if (ifail[0]==0)
            add2counter("modes",0,status);
         else
         {
            n=1;
            add2counter("modes",0,status+2);
            add2counter("modes",2,&n);
         }

         start_dfl_upd();
      }
   }

   act0[0]=momentum_action(0);
   n=2;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if (ap.action==ACG)
         act0[1]=action0(0);
      else
      {
         set_sw_parms(sea_quark_mass(ap.im0));
         status[NSTD_STATUS-1]=0;

         if (ap.action==ACF_TM1)
            act0[n]=setpf1(mu[ap.imu[0]],ap.ipf,0);
         else if (ap.action==ACF_TM1_EO)
            act0[n]=setpf4(mu[ap.imu[0]],ap.ipf,0,0);
         else if (ap.action==ACF_TM1_EO_SDET)
            act0[n]=setpf4(mu[ap.imu[0]],ap.ipf,1,0);
         else if (ap.action==ACF_TM2)
         {
            act0[n]=setpf2(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[1],
                           0,status);
            add2counter("field",ap.ipf,status);
         }
         else if (ap.action==ACF_TM2_EO)
         {
            act0[n]=setpf5(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[1],
                           0,status);
            add2counter("field",ap.ipf,status);
         }
         else if (ap.action==ACF_RAT)
         {
            act0[n]=setpf3(ap.irat,ap.ipf,0,ap.isp[1],0,status);
            add2counter("field",ap.ipf,status);
         }
         else if (ap.action==ACF_RAT_SDET)
         {
            act0[n]=setpf3(ap.irat,ap.ipf,1,ap.isp[1],0,status);
            add2counter("field",ap.ipf,status);
         }
         else
            error(1,1,"start_hmc [check2.c]","Unknown action");

         if (status[NSTD_STATUS-1])
            add2counter("modes",2,status+NSTD_STATUS-1);
         n+=1;
      }
   }
}


static void end_hmc(qflt *act1)
{
   int i,n,ifr,nact,*iact;
   int status[NSTD_STATUS];
   double *mu;
   hmc_parms_t hmc;
   action_parms_t ap;
   force_parms_t fp;

   hmc=hmc_parms();
   nact=hmc.nact;
   iact=hmc.iact;
   mu=hmc.mu;

   act1[0]=momentum_action(0);
   n=2;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);
      ifr=matching_force(iact[i]);
      fp=force_parms(ifr);

      if (ap.action==ACG)
         act1[1]=action0(0);
      else
      {
         set_sw_parms(sea_quark_mass(ap.im0));

         if (ap.action==ACF_TM1)
            act1[n]=action1(mu[ap.imu[0]],ap.ipf,ap.isp[0],fp.icr[0],
                            0,status);
         else if (ap.action==ACF_TM1_EO)
            act1[n]=action4(mu[ap.imu[0]],ap.ipf,0,ap.isp[0],fp.icr[0],
                            0,status);
         else if (ap.action==ACF_TM1_EO_SDET)
            act1[n]=action4(mu[ap.imu[0]],ap.ipf,1,ap.isp[0],fp.icr[0],
                            0,status);
         else if (ap.action==ACF_TM2)
            act1[n]=action2(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[0],
                            fp.icr[0],0,status);
         else if (ap.action==ACF_TM2_EO)
            act1[n]=action5(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[0],
                            fp.icr[0],0,status);
         else if (ap.action==ACF_RAT)
            act1[n]=action3(ap.irat,ap.ipf,0,ap.isp[0],0,status);
         else if (ap.action==ACF_RAT_SDET)
            act1[n]=action3(ap.irat,ap.ipf,1,ap.isp[0],0,status);
         else
            error(1,1,"end_hmc [check2.c]","Unknown action");

         add2counter("action",iact[i],status);

         if (status[NSTD_STATUS-1])
            add2counter("modes",2,status+NSTD_STATUS-1);
         n+=1;
      }
   }
}


static void flip_mom(void)
{
   int n,ifail[2],status[NSTD_STATUS];
   su3_alg_dble *mom;
   mdflds_t *mdfs;
   dfl_parms_t dfl;

   mdfs=mdflds();
   mom=(*mdfs).mom;
   flip_assign_alg2alg(4*VOLUME_TRD,2,mom,mom);

   reset_chrono(0);
   dfl=dfl_parms();

   if (dfl.Ns)
   {
      dfl_modes2(ifail,status);

      if ((ifail[0]<-2)||(ifail[1]<0))
      {
         print_status("dfl_modes2",ifail,status);
         error_root(1,1,"flip_mom [check2.c]",
                    "Deflation subspace generation failed");
      }

      if (ifail[0]==0)
         add2counter("modes",0,status);
      else
      {
         n=1;
         add2counter("modes",0,status+2);
         add2counter("modes",2,&n);
      }

      start_dfl_upd();
   }
}


static double cmp_ud(su3_dble *u,su3_dble *v)
{
   int i;
   double r[18],dev,dmax;

   r[ 0]=(*u).c11.re-(*v).c11.re;
   r[ 1]=(*u).c11.im-(*v).c11.im;
   r[ 2]=(*u).c12.re-(*v).c12.re;
   r[ 3]=(*u).c12.im-(*v).c12.im;
   r[ 4]=(*u).c13.re-(*v).c13.re;
   r[ 5]=(*u).c13.im-(*v).c13.im;

   r[ 6]=(*u).c21.re-(*v).c21.re;
   r[ 7]=(*u).c21.im-(*v).c21.im;
   r[ 8]=(*u).c22.re-(*v).c22.re;
   r[ 9]=(*u).c22.im-(*v).c22.im;
   r[10]=(*u).c23.re-(*v).c23.re;
   r[11]=(*u).c23.im-(*v).c23.im;

   r[12]=(*u).c31.re-(*v).c31.re;
   r[13]=(*u).c31.im-(*v).c31.im;
   r[14]=(*u).c32.re-(*v).c32.re;
   r[15]=(*u).c32.im-(*v).c32.im;
   r[16]=(*u).c33.re-(*v).c33.re;
   r[17]=(*u).c33.im-(*v).c33.im;

   dmax=0.0;

   for (i=0;i<18;i+=2)
   {
      dev=r[i]*r[i]+r[i+1]*r[i+1];
      if (dev>dmax)
         dmax=dev;
   }

   return dmax;
}


static double max_dev_ud(su3_dble *v)
{
   int iph;
   double d,dmax;
   su3_dble *u,*um;

   if (query_flags(UD_PHASE_SET))
   {
      iph=1;
      unset_ud_phase();
   }
   else
      iph=0;

   u=udfld();
   um=u+4*VOLUME;
   dmax=0.0;

   for (;u<um;u++)
   {
      d=cmp_ud(u,v);

      if (d>dmax)
         dmax=d;

      v+=1;
   }

   if (NPROC>1)
   {
      d=dmax;
      MPI_Reduce(&d,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }

   if (iph)
      set_ud_phase();

   return sqrt(dmax);
}


static void sum_act(int nact,qflt *act0,qflt *act1,double *sm)
{
   int i;
   double *qsm[2];
   qflt act[2],da;

   for (i=0;i<2;i++)
   {
      act[i].q[0]=0.0;
      act[i].q[1]=0.0;
      qsm[i]=act[i].q;
   }

   for (i=0;i<=nact;i++)
   {
      add_qflt(act0[i].q,act[0].q,act[0].q);
      da.q[0]=-act0[i].q[0];
      da.q[1]=-act0[i].q[1];
      add_qflt(act1[i].q,da.q,da.q);
      add_qflt(da.q,act[1].q,act[1].q);
   }

   global_qsum(2,qsm,qsm);

   for (i=0;i<2;i++)
      sm[i]=act[i].q[0];
}


int main(int argc,char *argv[])
{
   int nc,icnfg,nact;
   int isap,idfl;
   int nwud,nws,nwv,nwvd;
   double sm[2],dud,dH;
   double dudmin,dudmax,dudavg,dHmin,dHmax,dHavg;
   qflt *act0,*act1,*act2;
   su3_dble **usv;
   hmc_parms_t hmc;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);
      fin=freopen("check2.in","r",stdin);

      printf("\n");
      printf("Reversibility of the MD evolution\n");
      printf("---------------------------------\n\n");

      print_lattice_sizes();
   }

   read_run_parms();

   if (my_rank==0)
      fclose(fin);

   hmc_wsize(&nwud,&nws,&nwv,&nwvd);
   alloc_wud(nwud);
   alloc_ws(nws);
   wsd_uses_ws();
   alloc_wv(nwv);
   alloc_wvd(nwvd);
   usv=reserve_wud(1);

   hmc=hmc_parms();
   nact=hmc.nact;
   act0=malloc(3*(nact+1)*sizeof(*act0));
   act1=act0+nact+1;
   act2=act1+nact+1;
   error(act0==NULL,1,"main [check2.c]","Unable to allocate action arrays");

   check_machine();
   print_lat_parms(0x3);
   print_bc_parms(0x3);
   print_hmc_parms();
   print_action_parms();
   print_rat_parms();
   print_mdint_parms();
   print_force_parms(0x0);
   print_solver_parms(&isap,&idfl);
   if (isap)
      print_sap_parms(0);
   if (idfl)
      print_dfl_parms(1);

   if (my_rank==0)
   {
      print_iodat("i",iodat);

      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   start_ranlux(0,1234);
   geometry();
   check_files();

   hmc_sanity_check();
   set_mdsteps();
   setup_counters();
   setup_chrono();
   print_msize(2);

   dudmin=0.0;
   dudmax=0.0;
   dudavg=0.0;
   dHmin=0.0;
   dHmax=0.0;
   dHavg=0.0;

   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      if (my_rank==0)
      {
         printf("Configuration no %d\n\n",icnfg);
         fflush(flog);
      }

      sprintf(cnfg_file,"%sn%d",nbase,icnfg);
      read_flds(iodat,cnfg_file,0x0,0x1);

      start_hmc(act0,usv[0]);
      dud=max_dev_ud(usv[0]);
      run_mdint();
      end_hmc(act1);
      sum_act(nact,act0,act1,sm);

      if (my_rank==0)
      {
         printf("start_hmc:\n");
         printf("max|U_ij-U'_ij| = %.1e\n",dud);
         printf("run_mdint:\n");
         printf("H = %.6e\n",sm[0]);
         printf("dH = %.2e\n",sm[1]);
         fflush(flog);
      }

      print_all_avgstat();

      flip_mom();
      run_mdint();
      end_hmc(act2);
      sum_act(nact,act0,act2,sm);

      dH=fabs(sm[1]);
      dud=max_dev_ud(usv[0]);

      if (my_rank==0)
      {
         printf("Flip momenta and run_mdint:\n");
         printf("H  = %.6e\n",sm[0]);
         printf("|dH| = % .2e\n",dH);
         printf("max|U_ij-U'_ij| = %.2e\n\n",dud);
         fflush(flog);
      }

      if (icnfg==first)
      {
         dudmin=dud;
         dudmax=dud;
         dudavg=dud;

         dHmin=dH;
         dHmax=dH;
         dHavg=dH;
      }
      else
      {
         if (dud<dudmin)
            dudmin=dud;
         if (dud>dudmax)
            dudmax=dud;
         dudavg+=dud;

         if (dH<dHmin)
            dHmin=dH;
         if (dH>dHmax)
            dHmax=dH;
         dHavg+=dH;
      }
   }

   if (my_rank==0)
   {
      nc=(last-first)/step+1;

      printf("Test summary\n");
      printf("------------\n\n");

      printf("Considered %d configurations in the range %d -> %d\n\n",
             nc,first,last);

      printf("The three figures quoted in each case are the minimal,\n");
      printf("maximal and average values\n\n");

      printf("max|U_ij-U'_ij| = %.2e, %.2e, %.2e\n",
             dudmin,dudmax,dudavg/(double)(nc));
      printf("|dH|            = %.2e, %.2e, %.2e\n\n",
             dHmin,dHmax,dHavg/(double)(nc));

      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
