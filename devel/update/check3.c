
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2005-2018, 2022 Martin Luescher, Filippo Palombi,
*                               Stefan Schaefer
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Conservation of the Hamilton function by the MD evolution.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "linalg.h"
#include "archive.h"
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
                 "read_run_parms [check3.c]","Improper configuration range");
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
         "check_files [check3.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);
}


static void set_nstep(int *nstep)
{
   int ilv,i;
   hmc_parms_t hmc;
   mdint_parms_t mdp;

   hmc=hmc_parms();
   ilv=hmc.nlv-1;
   mdp=mdint_parms(ilv);
   nstep[0]=mdp.nstep;

  /* for (i=1;i<4;i++)
   {
      if (mdp.integrator==OMF4)
         nstep[i]=2*nstep[i-1];
      else
      {
         if (i>=2)
            nstep[i]=2*nstep[i-2];
         else
            nstep[i]=(int)(sqrt(2.0)*(double)(nstep[i-1])+0.5);
      }
   }*/
   for (i=1;i<4;i++)
   {
      nstep[i] = 2*nstep[i-1];
   }
}


static void reset_toplevel(int nstep)
{
   int ilv;
   hmc_parms_t hmc;
   mdint_parms_t mdp;

   hmc=hmc_parms();
   ilv=hmc.nlv-1;
   mdp=mdint_parms(ilv);
   set_mdint_parms(ilv,mdp.integrator,mdp.lambda,nstep,mdp.nfr,mdp.ifr);
}


static void start_hmc(qflt *act0,su3_dble *uold,su3_alg_dble *mold)
{
   int i,n,nact,*iact;
   int ifail[2],status[NSTD_STATUS];
   double *mu;
   mdflds_t *mdfs;
   dfl_parms_t dfl;
   hmc_parms_t hmc;
   action_parms_t ap;

   hmc=hmc_parms();
   nact=hmc.nact;
   iact=hmc.iact;
   mu=hmc.mu;

   random_mom();
   mdfs=mdflds();
   assign_alg2alg(4*VOLUME_TRD,2,(*mdfs).mom,mold);
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
            error_root(1,1,"start_hmc [check3.c]",
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
            error(1,1,"start_hmc [check3.c]","Unknown action");

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
            error(1,1,"end_hmc [check3.c]","Unknown action");

         add2counter("action",iact[i],status);

         if (status[NSTD_STATUS-1])
            add2counter("modes",2,status+NSTD_STATUS-1);
         n+=1;
      }
   }
}


static void restart_hmc(su3_dble *uold,su3_alg_dble *mold)
{
   int n,ifail[2],status[NSTD_STATUS];
   mdflds_t *mdfs;
   hmc_parms_t hmc;
   dfl_parms_t dfl;

   assign_ud2ud(4*VOLUME_TRD,2,uold,udfld());
   set_flags(UPDATED_UD);
   set_flags(UNSET_UD_PHASE);

   mdfs=mdflds();
   assign_alg2alg(4*VOLUME_TRD,2,mold,(*mdfs).mom);

   hmc=hmc_parms();

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
            error_root(1,1,"restart_hmc [check3.c]",
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
}


static void sum_act(int nact,qflt *act0,qflt *act1,double *sm)
{
   int i;
   double *qsm[3];
   qflt act[3],da;

   for (i=0;i<3;i++)
   {
      act[i].q[0]=0.0;
      act[i].q[1]=0.0;
      qsm[i]=act[i].q;
   }

   for (i=0;i<=nact;i++)
   {
      add_qflt(act0[i].q,act[0].q,act[0].q);
      add_qflt(act1[i].q,act[1].q,act[1].q);
      da.q[0]=-act0[i].q[0];
      da.q[1]=-act0[i].q[1];
      add_qflt(act1[i].q,da.q,da.q);
      add_qflt(da.q,act[2].q,act[2].q);
   }

   global_qsum(3,qsm,qsm);

   for (i=0;i<3;i++)
      sm[i]=act[i].q[0];
}


int main(int argc,char *argv[])
{
   int icnfg,nact,i;
   int isap,idfl;
   int nwud,nws,nwv,nwvd;
   int nstep[4];
   double sm[3],dH[4];
   qflt *act0,*act1;
   su3_dble **usv;
   su3_alg_dble **fsv;
   hmc_parms_t hmc;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      fin=freopen("check3.in","r",stdin);

      printf("\n");
      printf("Conservation of the Hamilton function by the MD evolution\n");
      printf("---------------------------------------------------------\n\n");

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
   alloc_wfd(1);
   usv=reserve_wud(1);
   fsv=reserve_wfd(1);

   hmc=hmc_parms();
   nact=hmc.nact;
   act0=malloc(2*(nact+1)*sizeof(*act0));
   act1=act0+nact+1;
   error(act0==NULL,1,"main [check3.c]","Unable to allocate action arrays");
   set_nstep(nstep);

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
   setup_counters();
   setup_chrono();
   print_msize(2);

   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      if (my_rank==0)
      {
         printf("Configuration no %d\n\n",icnfg);
         fflush(flog);
      }

      sprintf(cnfg_file,"%sn%d",nbase,icnfg);
      read_flds(iodat,cnfg_file,0x0,0x1);

      for (i=0;i<4;i++)
      {
         if (i>0)
            reset_toplevel(nstep[i]);

         set_mdsteps();

         if (i==0)
            start_hmc(act0,usv[0],fsv[0]);
         else
            restart_hmc(usv[0],fsv[0]);

         run_mdint();
         end_hmc(act1);
         sum_act(nact,act0,act1,sm);
         dH[i]=fabs(sm[2]);

         if (my_rank==0)
         {
            if (i==0)
            {
               printf("start_hmc:\n");
               printf("H = %.6e\n",sm[0]);
               fflush(flog);
            }

            printf("\n");
            printf("run_md: nstep = %d, tau/nstep = %.2e\n",
                   nstep[i],hmc.tau/(double)(nstep[i]));

            print_all_avgstat();

            printf("H = %.6e, |dH| = %.2e\n",sm[1],dH[i]);
            fflush(flog);
         }
      }

      if (my_rank==0)
      {
         printf("\n");
         printf("nstep[0] = %d,            |dH| = %.2e\n",nstep[0],dH[0]);

         for (i=1;i<4;i++)
         {
            printf("nstep[%d]/nstep[%d] = %.2e, ",
                   i-1,i,(double)(nstep[i-1])/(double)(nstep[i]));
            printf("|dH| = %.2e, |dH[%d]|/|dH[%d]| = %.2e\n",
                   dH[i],i,i-1,dH[i]/dH[i-1]);
         }

         printf("\n");
         fflush(flog);
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
