
/*******************************************************************************
*
* File wflow.c
*
* Copyright (C) 2009-2013, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Integration of the Wilson flow.
*
*   void fwd_euler(int n,double eps)
*     Applies n forward Euler integration steps, with step size eps, to the
*     current gauge field.
*
*   void fwd_rk2(int n,double eps)
*     Applies n forward 2nd-order Runge-Kutta integration steps, with step
*     size eps, to the current gauge field.
*
*   void fwd_rk3(int n,double eps)
*     Applies n forward 3rd-order Runge-Kutta integration steps, with step
*     size eps, to the current gauge field.
*
* On lattices with periodic boundary conditions, the Wilson flow is defined
* through equations (1.3) and (1.4) in
*
*   M. Luescher: "Properties and uses of the Wilson flow in lattice QCD",
*   JHEP 1008 (2010) 071.
*
* The Runge-Kutta integrators used here are described in appendix C of this
* paper.
*
* In the case of open, SF and open-SF boundary conditions, the flow evolves
* the active link variables only, where the action on the right of the flow
* equation is taken to be the tree-level O(a)-improved plaquette action.
* O(a)-improvement moreover requires the derivative of the action on the
* space-like links at the boundaries with open boundary conditions to be
* multiplied by 2. See appendix B in
*
*   M. Luescher, S. Schaefer: "Lattice QCD with open boundary conditions
*   and twisted-mass reweighting", Comput.Phys.Commun. 184 (2013) 519,
*
* for the exact form of the flow equation in this case.
*
* The integration programs make use of the force field in the structure
* returned by mdflds [mdflds.c]. On exit the force field must therefore be
* expected to be changed.
*
* The Runge-Kutta integrators require a workspace of 1 force field. All
* programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define WFLOW_C

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "su3fcts.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "linalg.h"
#include "forces.h"
#include "wflow.h"
#include "global.h"

#define N0 (NPROC0*L0)


static void update_ud(double eps,su3_alg_dble *frc)
{
   int bc;
   int k,ofs,vol,ix,t,ifc;
   su3_dble *udb,*ud;
   su3_alg_dble *fd;

   bc=bc_type();
   udb=udfld();

#pragma omp parallel private(k,ofs,vol,ix,t,ifc,ud,fd)
   {
      k=omp_get_thread_num();

      vol=VOLUME_TRD/2;
      ofs=(VOLUME/2)+k*vol;
      ud=udb+8*k*vol;
      fd=frc+8*k*vol;

      for (ix=ofs;ix<(ofs+vol);ix++)
      {
         t=global_time(ix);

         if (t==0)
         {
            expXsu3(eps,fd,ud);
            fd+=1;
            ud+=1;;

            if (bc!=0)
               expXsu3(eps,fd,ud);
            fd+=1;
            ud+=1;;

            for (ifc=2;ifc<8;ifc++)
            {
               if ((bc==0)||(bc==2))
                  expXsu3(2.0*eps,fd,ud);
               else if (bc==3)
                  expXsu3(eps,fd,ud);
               fd+=1;
               ud+=1;;
            }
         }
         else if (t==(N0-1))
         {
            if (bc!=0)
               expXsu3(eps,fd,ud);
            fd+=1;
            ud+=1;;

            expXsu3(eps,fd,ud);
            fd+=1;
            ud+=1;;

            for (ifc=2;ifc<8;ifc++)
            {
               if (bc==0)
                  expXsu3(2.0*eps,fd,ud);
               else
                  expXsu3(eps,fd,ud);
               fd+=1;
               ud+=1;;
            }
         }
         else
         {
            for (ifc=0;ifc<8;ifc++)
            {
               expXsu3(eps,fd,ud);
               fd+=1;
               ud+=1;
            }
         }
      }
   }

   set_flags(UPDATED_UD);
}


static void update_fro1(double c,su3_alg_dble *frc,su3_alg_dble *fro)
{
   int k;
   su3_alg_dble *fdc,*fdo,*fdm;

#pragma omp parallel private(k,fdc,fdo,fdm)
   {
      k=omp_get_thread_num();

      fdo=fro+k*4*VOLUME_TRD;
      fdc=frc+k*4*VOLUME_TRD;
      fdm=fdc+4*VOLUME_TRD;

      for (;fdc<fdm;fdc++)
      {
         (*fdo).c1-=c*(*fdc).c1;
         (*fdo).c2-=c*(*fdc).c2;
         (*fdo).c3-=c*(*fdc).c3;
         (*fdo).c4-=c*(*fdc).c4;
         (*fdo).c5-=c*(*fdc).c5;
         (*fdo).c6-=c*(*fdc).c6;
         (*fdo).c7-=c*(*fdc).c7;
         (*fdo).c8-=c*(*fdc).c8;

         fdo+=1;
      }
   }
}


static void update_fro2(double c,su3_alg_dble *frc,su3_alg_dble *fro)
{
   int k;
   su3_alg_dble *fdc,*fdo,*fdm;

#pragma omp parallel private(k,fdc,fdo,fdm)
   {
      k=omp_get_thread_num();

      fdo=fro+k*4*VOLUME_TRD;
      fdc=frc+k*4*VOLUME_TRD;
      fdm=fdc+4*VOLUME_TRD;

      for (;fdc<fdm;fdc++)
      {
         (*fdo).c1=(*fdc).c1+c*(*fdo).c1;
         (*fdo).c2=(*fdc).c2+c*(*fdo).c2;
         (*fdo).c3=(*fdc).c3+c*(*fdo).c3;
         (*fdo).c4=(*fdc).c4+c*(*fdo).c4;
         (*fdo).c5=(*fdc).c5+c*(*fdo).c5;
         (*fdo).c6=(*fdc).c6+c*(*fdo).c6;
         (*fdo).c7=(*fdc).c7+c*(*fdo).c7;
         (*fdo).c8=(*fdc).c8+c*(*fdo).c8;

         fdo+=1;
      }
   }
}


void fwd_euler(int n,double eps)
{
   int iprms[1],k;
   double dprms[1];
   su3_alg_dble *frc;
   mdflds_t *mdfs;

   if (NPROC>1)
   {
      iprms[0]=n;
      dprms[0]=eps;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((iprms[0]!=n)||(dprms[0]!=eps),1,
            "fwd_euler [wflow.c]","Parameters are not global");
   }

   if (n>0)
   {
      chexp_init();
      mdfs=mdflds();
      frc=(*mdfs).frc;

      for (k=0;k<n;k++)
      {
         plaq_frc();
         update_ud(-eps,frc);
      }
   }
}


void fwd_rk2(int n,double eps)
{
   int iprms[1],k;
   double dprms[1];
   su3_alg_dble *frc,*fro,**fsv;
   mdflds_t *mdfs;

   if (NPROC>1)
   {
      iprms[0]=n;
      dprms[0]=eps;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((iprms[0]!=n)||(dprms[0]!=eps),1,
            "fwd_rk2 [wflow.c]","Parameters are not global");
   }

   if (n>0)
   {
      chexp_init();
      mdfs=mdflds();
      frc=(*mdfs).frc;
      fsv=reserve_wfd(1);
      fro=fsv[0];

      for (k=0;k<n;k++)
      {
         plaq_frc();
         assign_alg2alg(4*VOLUME_TRD,2,frc,fro);
         update_ud(-0.5*eps,frc);

         plaq_frc();
         update_fro2(-0.5,frc,fro);
         update_ud(-eps,fro);
      }

      release_wfd();
   }
}


void fwd_rk3(int n,double eps)
{
   int iprms[1],k;
   double dprms[1];
   su3_alg_dble *frc,*fro,**fsv;
   mdflds_t *mdfs;

   if (NPROC>1)
   {
      iprms[0]=n;
      dprms[0]=eps;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((iprms[0]!=n)||(dprms[0]!=eps),1,
            "fwd_rk3 [wflow.c]","Parameters are not global");
   }

   if (n>0)
   {
      chexp_init();
      mdfs=mdflds();
      frc=(*mdfs).frc;
      fsv=reserve_wfd(1);
      fro=fsv[0];

      for (k=0;k<n;k++)
      {
         plaq_frc();
         assign_alg2alg(4*VOLUME_TRD,2,frc,fro);
         update_ud(-0.25*eps,frc);

         plaq_frc();
         update_fro1(32.0/17.0,frc,fro);
         update_ud((17.0/36.0)*eps,fro);

         plaq_frc();
         update_fro2(17.0/27.0,frc,fro);
         update_ud(-0.75*eps,fro);
      }

      release_wfd();
   }
}
