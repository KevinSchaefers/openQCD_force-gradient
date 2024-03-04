
/*******************************************************************************
*
* File update.c
*
* Copyright (C) 2017, 2018, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Molecular-dynamics elementary integration steps.
*
*   void update_mom(void)
*     Subtracts the current force field from the momentum field (see the
*     notes). The operation is performed on the active links only.
*
*   void update_ud(double eps)
*     Replaces the gauge field variables ud by exp{eps*mom}*ud, where mom
*     is the current momentum field (see the notes). Only the active link
*     variables are updated and the molecular-dynamics time is advanced by
*     eps.
*
*   void start_dfl_upd(void)
*     Starts (or restarts) the deflation-subspace update cycle (see the
*     notes).
*
*   void dfl_upd(void)
*     Reads the molecular-dynamics time and updates the deflation subspace
*     if an update is due according to the chosen update scheme.
*
* The programs update_mom() and update_ud() act on the global fields, i.e.
* the gauge, momentum and force fields that can be accessed through udfld()
* [uflds/uflds.c] and mdflds() [mdflds/mdflds.c], respectively.
*
* The update scheme for the deflation subspace is defined by the parameter
* data base [flags/dfl_parms.c]. No subspace initialization is performed
* by start_dfl_upd(), only the update time is set to the current time. The
* program dfl_upd() does nothing if start_dfl_upd() has not been called.
*
* The program dfl_upd() assumes that the deflation subspace and the counters
* have been properly initialized. If phase-periodic boundary conditions are
* chosen, the calling program must ensure that the gauge field is phase-set.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define UPDATE_C

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "su3fcts.h"
#include "dfl.h"
#include "update.h"
#include "global.h"

#define N0 (NPROC0*L0)

static int nsm=0;
static double dtau=1.0,rtau=0.0;


void update_mom(void)
{
   int bc;
   int k,ofs,vol,ix,t,ifc;
   su3_alg_dble *mom,*frc;
   mdflds_t *mdfs;

   bc=bc_type();
   mdfs=mdflds();

#pragma omp parallel private(k,ofs,vol,ix,t,ifc,mom,frc)
   {
      k=omp_get_thread_num();

      vol=VOLUME_TRD/2;
      ofs=(VOLUME/2)+k*vol;

      mom=(*mdfs).mom+8*k*vol;
      frc=(*mdfs).frc+8*k*vol;

      for (ix=ofs;ix<(ofs+vol);ix++)
      {
         t=global_time(ix);

         if (t==0)
         {
            _su3_alg_sub_assign(mom[0],frc[0]);
            mom+=1;
            frc+=1;

            if (bc!=0)
            {
               _su3_alg_sub_assign(mom[0],frc[0]);
            }

            mom+=1;
            frc+=1;

            for (ifc=2;ifc<8;ifc++)
            {
               if (bc!=1)
               {
                  _su3_alg_sub_assign(mom[0],frc[0]);
               }

               mom+=1;
               frc+=1;
            }
         }
         else if (t==(N0-1))
         {
            if (bc!=0)
            {
               _su3_alg_sub_assign(mom[0],frc[0]);
            }

            mom+=1;
            frc+=1;

            for (ifc=1;ifc<8;ifc++)
            {
               _su3_alg_sub_assign(mom[0],frc[0]);
               mom+=1;
               frc+=1;
            }
         }
         else
         {
            for (ifc=0;ifc<8;ifc++)
            {
               _su3_alg_sub_assign(mom[0],frc[0]);
               mom+=1;
               frc+=1;
            }
         }
      }
   }
}

void fg_update_ud(double eps)
{
   int bc;
   int k,ofs,vol,ix,t,ifc;
   su3_dble *ud;
   su3_alg_dble *frc;
   mdflds_t *mdfs;

   bc=bc_type();
   (void)(udfld());
   mdfs=mdflds();
   chexp_init();

#pragma omp parallel private(k,ofs,vol,ix,t,ifc,ud,frc)
   {
      k=omp_get_thread_num();

      vol=VOLUME_TRD/2;
      ofs=(VOLUME/2)+k*vol;

      ud=udfld()+8*k*vol;
      frc=(*mdfs).frc+8*k*vol;

      for (ix=ofs;ix<(ofs+vol);ix++)
      {
         t=global_time(ix);

         if (t==0)
         {
            expXsu3(eps,frc,ud);
            ud+=1;
            frc+=1;

            if (bc!=0)
            {
                expXsu3(eps,frc,ud);
            }
            ud+=1;
            frc+=1;

            for (ifc=2;ifc<8;ifc++)
            {
               if (bc!=1)
               {
                   expXsu3(eps,frc,ud);
               }
               ud+=1;
               frc+=1;
            }
         }
         else if (t==(N0-1))
         {
            if (bc!=0)
            {
                expXsu3(eps,frc,ud);
            }
            ud+=1;
            frc+=1;

            for (ifc=1;ifc<8;ifc++)
            {
               expXsu3(eps,frc,ud);
               ud+=1;
               frc+=1;
            }
         }
         else
         {
            for (ifc=0;ifc<8;ifc++)
            {
               expXsu3(eps,frc,ud);
               ud+=1;
               frc+=1;
            }
         }
      }
   }

   /* step_mdtime(eps); muss vielleicht weg? */ 
   set_flags(UPDATED_UD);
}


void update_ud(double eps)
{
   int bc;
   int k,ofs,vol,ix,t,ifc;
   su3_dble *ud;
   su3_alg_dble *mom;
   mdflds_t *mdfs;

   bc=bc_type();
   (void)(udfld());
   mdfs=mdflds();
   chexp_init();

#pragma omp parallel private(k,ofs,vol,ix,t,ifc,ud,mom)
   {
      k=omp_get_thread_num();

      vol=VOLUME_TRD/2;
      ofs=(VOLUME/2)+k*vol;

      ud=udfld()+8*k*vol;
      mom=(*mdfs).mom+8*k*vol;

      for (ix=ofs;ix<(ofs+vol);ix++)
      {
         t=global_time(ix);

         if (t==0)
         {
            expXsu3(eps,mom,ud);
            ud+=1;
            mom+=1;

            if (bc!=0)
               expXsu3(eps,mom,ud);
            ud+=1;
            mom+=1;

            for (ifc=2;ifc<8;ifc++)
            {
               if (bc!=1)
                  expXsu3(eps,mom,ud);
               ud+=1;
               mom+=1;
            }
         }
         else if (t==(N0-1))
         {
            if (bc!=0)
               expXsu3(eps,mom,ud);
            ud+=1;
            mom+=1;

            for (ifc=1;ifc<8;ifc++)
            {
               expXsu3(eps,mom,ud);
               ud+=1;
               mom+=1;
            }
         }
         else
         {
            for (ifc=0;ifc<8;ifc++)
            {
               expXsu3(eps,mom,ud);
               ud+=1;
               mom+=1;
            }
         }
      }
   }

   step_mdtime(eps);
   set_flags(UPDATED_UD);
}


void start_dfl_upd(void)
{
   dfl_upd_parms_t dup;

   dup=dfl_upd_parms();
   dtau=dup.dtau;
   nsm=dup.nsm;
   rtau=mdtime();
}


void dfl_upd(void)
{
   int n,ifail[2],status[4];
   double tau;
   dfl_parms_t dfl;

   tau=mdtime();

   if ((nsm>0)&&((tau-rtau)>dtau))
   {
      dfl=dfl_parms();

      if (dfl.Ns)
      {
         dfl_update2(nsm,ifail,status);

         if ((ifail[0]<-2)||(ifail[1]<0))
         {
            print_status("dfl_update2",ifail,status);
            error_root(1,1,"dfl_upd [update.c]",
                       "Deflation subspace update failed");
         }

         if (ifail[0]==0)
            add2counter("modes",1,status);
         else
         {
            n=1;
            add2counter("modes",0,status+2);
            add2counter("modes",2,&n);
         }

         rtau=tau;
      }
   }
}
