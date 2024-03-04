
/*******************************************************************************
*
* File sap_gcr.c
*
* Copyright (C) 2005-2013, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* SAP+GCR solver for the Wilson-Dirac equation.
*
*   double sap_gcr(int nkv,int nmx,int istop,double res,double mu,
*                  spinor_dble *eta,spinor_dble *psi,int *ifail,int *status)
*     Obtains an approximate solution psi of the Wilson-Dirac equation for
*     given source eta using the SAP-preconditioned GCR algorithm.
*
* Depending on whether the twisted-mass flag is set or not, the program
* solves the equation
*
*   (Dw+i*mu*gamma_5*1e)*psi=eta  or  (Dw+i*mu*gamma_5)*psi=eta,
*
* respectively, where 1e is 1 on the even and 0 on the odd lattice sites.
* The twisted-mass flag is retrieved from the parameter data base (see
* flags/lat_parms.c).
*
* The program is based on the flexible GCR algorithm (see linsolv/fgcr.c).
* Before the solver is launched, the following parameter-setting programs
* must have been called:
*
*  set_lat_parms()        SW improvement coefficient.
*
*  set_bc_parms()         Boundary conditions and associated improvement
*                         coefficients.
*
*  set_sw_parms()         Bare quark mass.
*
*  set_sap_parms()        Parameters of the SAP preconditioner.
*
* See doc/parms.pdf and the relevant files in the modules/flags directory
* for further explanations.
*
* All other parameters are passed through the argument list:
*
*  nkv       Maximal number of Krylov vectors generated before the GCR
*            algorithm is restarted.
*
*  nmx       Maximal total number of Krylov vectors that may be generated.
*
*  istop     Stopping criterion (0: L_2 norm based, 1: uniform norm based).
*
*  res       Desired maximal relative residue of the calculated solution.
*
*  mu        Value of the twisted mass in the Dirac equation.
*
*  eta       Source field. eta is unchanged on exit unless psi=eta (which
*            is permissible).
*
*  psi       Calculated approximate solution of the Dirac equation.
*
* On exit
*
*  ifail[0]=0     The program completed successfully.
*
*  ifail[0]=-1    The solver did not converge.
*
*  ifail[0]=-3    The inversion of the SW term on the odd sites
*                 of the lattice was not safe.
*
*  status[0]      Total number of Krylov vectors that were generated
*                 by the algorithm.
*
* The program returns the norm of the residue of the calculated approximate
* solution if ifail[0]>=-1. Otherwise the program returns the norm of the
* source eta (which is unchanged) and sets psi to zero if psi!=eta.
*
* The fields eta and psi must be such that the Dirac operator can act on
* them (see main/README.global). Moreover, the source eta is assumed to
* respect the chosen boundary conditions (see doc/dirac.pdf).
*
* The SAP_BLOCKS blocks grid is automatically allocated and the SW term is
* recalculated when needed. The gauge and SW fields are then copied to the
* block grid if they are not in the proper condition.
*
* The required workspaces are
*
*  spinor              2*nkv+1
*  spinor_dble         2
*
* (see utils/wspace.c).
*
* The program sap_gcr() is assumed to be called by the OpenMP master thread
* on all MPI processes simultaneously.
*
* If SSE (AVX) instructions are used, the Dirac spinors must be aligned to
* a 16 (32) byte boundary.
*
*******************************************************************************/

#define SAP_GCR_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "flags.h"
#include "utils.h"
#include "block.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "linsolv.h"
#include "sap.h"
#include "global.h"

static double mud;
static sap_parms_t spr;


static void Dop(spinor_dble *s,spinor_dble *r)
{
   Dw_dble(mud,s,r);
}


static void Mop(int k,spinor *rho,spinor *phi,spinor *chi)
{
   int n;

   set_s2zero(VOLUME_TRD,2,phi);
   assign_s2s(VOLUME_TRD,2,rho,chi);

   for (n=0;n<spr.ncy;n++)
      sap((float)(mud),spr.isolv,spr.nmr,phi,chi);

   diff_s2s(VOLUME_TRD,2,rho,chi);
}


double sap_gcr(int nkv,int nmx,int istop,double res,double mu,
               spinor_dble *eta,spinor_dble *psi,int *ifail,int *status)
{
   int nb,isw;
   int swde,swdo,swu,swe,swo;
   double rho0,rho;
   qflt qnrm;
   spinor **ws;
   spinor_dble **wsd,*rsd;

   spr=sap_parms();
   error_root(spr.ncy==0,1,"sap_gcr [sap_gcr.c]","SAP parameters are not set");

   blk_list(SAP_BLOCKS,&nb,&isw);

   if (nb==0)
      alloc_bgr(SAP_BLOCKS);

   if (query_grid_flags(SAP_BLOCKS,UBGR_MATCH_UD)!=1)
      assign_ud2ubgr(SAP_BLOCKS);

   if (query_flags(SWD_UP2DATE)!=1)
      sw_term(NO_PTS);

   swde=query_flags(SWD_E_INVERTED);
   swdo=query_flags(SWD_O_INVERTED);

   swu=query_grid_flags(SAP_BLOCKS,SW_UP2DATE);
   swe=query_grid_flags(SAP_BLOCKS,SW_E_INVERTED);
   swo=query_grid_flags(SAP_BLOCKS,SW_O_INVERTED);
   ifail[0]=0;

   if (spr.isolv==0)
   {
      if ((swde==1)||(swdo==1))
         sw_term(NO_PTS);

      if ((swu!=1)||(swe==1)||(swo==1))
         assign_swd2swbgr(SAP_BLOCKS,NO_PTS);
   }
   else if (spr.isolv==1)
   {
      if ((swde!=1)&&(swdo==1))
      {
         if ((swu!=1)||(swe==1)||(swo!=1))
            assign_swd2swbgr(SAP_BLOCKS,NO_PTS);

         sw_term(NO_PTS);
      }
      else
      {
         if ((swde==1)||(swdo==1))
            sw_term(NO_PTS);

         if ((swu!=1)||(swe==1)||(swo!=1))
            ifail[0]=assign_swd2swbgr(SAP_BLOCKS,ODD_PTS);
      }
   }
   else
      error_root(1,1,"sap_gcr [sap_gcr.c]","Unknown block solver");

   status[0]=0;
   rho=0.0;

   if (ifail[0])
      ifail[0]=-3;
   else
   {
      rho0=unorm_dble(VOLUME_TRD,3,eta);

      if (rho0!=0.0)
      {
         ws=reserve_ws(2*nkv+1);
         wsd=reserve_wsd(2);
         rsd=wsd[1];

         mud=mu;
         assign_sd2sd(VOLUME_TRD,2,eta,rsd);
         scale_dble(VOLUME_TRD,2,1.0/rho0,rsd);

         rho=fgcr(VOLUME_TRD,Dop,Mop,ws,wsd,nkv,nmx,istop,res,rsd,
                  psi,status);

         scale_dble(VOLUME_TRD,2,rho0,psi);
         rho*=rho0;

         if (status[0]<0)
         {
            ifail[0]=-1;
            status[0]=nmx;
         }

         release_wsd();
         release_ws();
      }
      else
         set_sd2zero(VOLUME_TRD,2,psi);
   }

   if (ifail[0]<=-2)
   {
      if (psi!=eta)
         set_sd2zero(VOLUME_TRD,2,psi);

      if (istop)
         rho=unorm_dble(VOLUME_TRD,3,eta);
      else
      {
         qnrm=norm_square_dble(VOLUME_TRD,3,eta);
         rho=sqrt(qnrm.q[0]);
      }
   }

   return rho;
}
