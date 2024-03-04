
/*******************************************************************************
*
* File mdint.c
*
* Copyright (C) 2011-2018, 2020, 2022  Stefan Schaefer, Martin Luescher,
*                                      John Bulava
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Integration of the molecular-dynamics equations.
*
*   void run_mdint(void)
*     Integrates the molecular-dynamics equations.
*
* The integrator used is the one defined by the array of elementary operations
* returned by mdsteps() (see update/mdsteps.c). It is assumed that the fields,
* the integrator, the status counters and the chronological propagation of the
* the solutions of the Dirac equation have been properly initialized.
*
* In the course of the integration, the solver iteration numbers are added
* to the appropriate counters provided by the module update/counters.c. The
* deflation subspace is updated according to the parameter data base (see
* flags/dfl_parms.c).
*
* This program does not change the phase of the link variables. If phase-
* periodic boundary conditions are chosen, it is up to the calling program
* to ensure that the gauge field is in the proper phase-set condition (see
* uflds/uflds.c).
*
* Some debugging information is printed to stdout on MPI process 0 if the
* macro MDINT_DBG is defined at compilation time. The norm of the forces
* printed is the uniform norm.
*
* The program run_mdint() is assumed to be called by the OpenMP master thread
* on all MPI processes simultaneously.
*
*******************************************************************************/

#define MDINT_C

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "mdflds.h"
#include "linalg.h"
#include "forces.h"
#include "update.h"
#include "global.h"
#include "su3fcts.h"
#include "uflds.h"
su3_dble **ud_tmp=NULL;    /* temporary link field for Hessian-free force-gradient integrator */


#ifdef MDINT_DBG

static void print_force_step(mdstep_t *s,double wdt)
{
   int my_rank;
   double nrm,eps;
   force_parms_t fp;
   mdflds_t *mdfs;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   mdfs=mdflds();
   nrm=unorm_alg(4*VOLUME,1,(*mdfs).frc);
   fp=force_parms((*s).iop);
   eps=(*s).eps;

   if (my_rank==0)
   {
      if (fp.force==FRG)
         printf("Force FRG:              ");
      else if (fp.force==FRF_TM1)
         printf("Force FRF_TM1:          ");
      else if (fp.force==FRF_TM1_EO)
         printf("Force FRF_TM1_EO:       ");
      else if (fp.force==FRF_TM1_EO_SDET)
         printf("Force FRF_TM1_EO_SDET:  ");
      else if (fp.force==FRF_TM2)
         printf("Force FRF_TM2:          ");
      else if (fp.force==FRF_TM2_EO)
         printf("Force FRF_TM2_EO:       ");
      else if (fp.force==FRF_RAT)
         printf("Force FRF_RAT:          ");
      else if (fp.force==FRF_RAT_SDET)
         printf("Force FRF_RAT_SDET:     ");

      printf("|frc| = %.2e, eps = % .2e, |eps*frc| = %.2e, "
             "time = %.2e sec\n",nrm/fabs(eps),eps,nrm,wdt);
   }
}

#endif

static void mdint(double *mu)
{
   int nop,itu,iop;
   int status[NSTD_STATUS];
   double eps;
#ifdef MDINT_DBG
   double wt1,wt2;
#endif
   mdstep_t *s,*sm;
   force_parms_t fp;
   solver_parms_t sp;

   s=mdsteps(&nop,&itu);
   sm=s+nop;

   set_frc2zero();
    
   for (;s<sm;s++)
   {
      iop=(*s).iop;
      eps=(*s).eps;
       
      if (iop<itu-3)
      {
         fp=force_parms(iop);

#ifdef MDINT_DBG
         MPI_Barrier(MPI_COMM_WORLD);
         wt1=MPI_Wtime();
#endif
         if (fp.force==FRG)
              force0(eps);
         else
         {
            sp=solver_parms(fp.isp[0]);
            if (sp.solver==DFL_SAP_GCR)
               dfl_upd();
            set_sw_parms(sea_quark_mass(fp.im0));

            if (fp.force==FRF_TM1)
               force1(mu[fp.imu[0]],fp.ipf,fp.isp[0],fp.icr[0],
                      eps,status);
            else if (fp.force==FRF_TM1_EO)
               force4(mu[fp.imu[0]],fp.ipf,0,fp.isp[0],fp.icr[0],
                      eps,status);
            else if (fp.force==FRF_TM1_EO_SDET)
               force4(mu[fp.imu[0]],fp.ipf,1,fp.isp[0],fp.icr[0],
                      eps,status);
            else if (fp.force==FRF_TM2)
               force2(mu[fp.imu[0]],mu[fp.imu[1]],fp.ipf,fp.isp[0],fp.icr[0],
                      eps,status);
            else if (fp.force==FRF_TM2_EO)
               force5(mu[fp.imu[0]],mu[fp.imu[1]],fp.ipf,fp.isp[0],fp.icr[0],
                      eps,status);
            else if (fp.force==FRF_RAT)
               force3(fp.irat,fp.ipf,0,fp.isp[0],
                      eps,status);
            else if (fp.force==FRF_RAT_SDET)
               force3(fp.irat,fp.ipf,1,fp.isp[0],
                      eps,status);
            else
               error_root(1,1,"mdint [mdint.c]","Unknown force");

            add2counter("force",iop,status);
            if (status[NSTD_STATUS-1])
               add2counter("modes",2,status+NSTD_STATUS-1);
         }

#ifdef MDINT_DBG
         MPI_Barrier(MPI_COMM_WORLD);
         wt2=MPI_Wtime();
         print_force_step(s,wt2-wt1);
         update_mom();
         set_frc2zero();
#endif
      }
      else if (iop==(itu-3) )
      {
          update_mom();
          set_frc2zero();
      }
      else if (iop==itu)
      {
         update_ud(eps);
      }
      else if (iop==itu-2)
      {
          if (ud_tmp==NULL)
            ud_tmp=reserve_wud(1);
          assign_ud2ud(4*VOLUME_TRD,2,udfld(),ud_tmp[0]);
          fg_update_ud(eps);
          set_frc2zero();
      }
      else if (iop==itu-1)
      {
          update_mom();
          set_frc2zero();
          assign_ud2ud(4*VOLUME_TRD,2,ud_tmp[0],udfld());
          set_flags(UPDATED_UD);
	  release_wud();
          ud_tmp=NULL;
      }
      else
      {
          update_mom();
          set_frc2zero();
      }
   }
}


void run_mdint(void)
{
   hmc_parms_t hmc;
   smd_parms_t smd;

   hmc=hmc_parms();
   smd=smd_parms();

   if (hmc.nlv!=0)      
      mdint(hmc.mu);
   else if (smd.nlv!=0)
      mdint(smd.mu);
   else
      error_root(1,1,"run_mdint [mdint.c]",
                 "Simulation parameters are not set");
}
