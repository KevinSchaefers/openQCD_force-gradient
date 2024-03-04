
/*******************************************************************************
*
* File bcnds.c
*
* Copyright (C) 2005, 2010-2014, 2021 Martin Luescher, John Bulava
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs related to the boundary conditions in the time direction.
*
*   int *bnd_lks(int *n)
*     Returns the starting address of an array of length n whose elements
*     are the integer offsets of the time-like link variables on the local
*     lattice at global time NPROC0*L0-1.
*
*   int *bnd_pts(int *n)
*     Returns the starting address of an array of length n whose elements
*     are the indices of the points on the local lattice at global time 0
*     (boundary conditions type 0,1 or 2) and time NPROC0*L0-1 (boundary
*     conditions type 0). The ordering of the indices is such that the n/2
*     even points come first.
*
*   void set_bc(void)
*     Sets the double-precision link variables at time 0 and T to the
*     values required by the chosen boundary conditions (see the notes).
*
*   int check_bc(double tol)
*     Returns 1 if the double-precision gauge field has the proper boundary
*     values and if no active link variables are equal to zero. Otherwise
*     the program returns 0. The parameter tol>=0.0 sets an upper bound on
*     the tolerated difference of the boundary values of the gauge field from
*     the expected ones in the case of SF and open-SF boundary conditions.
*
*   void bnd_s2zero(ptset_t set,spinor *s)
*     Sets the components of the single-precision spinor field s on the
*     specified set of points at global time 0 (boundary conditions type
*     0,1 or 2) and time NPROC0*L0-1 (boundary conditions type 0) to zero.
*
*   void bnd_sd2zero(ptset_t set,spinor_dble *sd)
*     Sets the components of the double-precision spinor field sd on the
*     specified set of points at global time 0 (boundary conditions type
*     0,1 or 2) and time NPROC0*L0-1 (boundary conditions type 0) to zero.
*
* The time extent T of the lattice is
*
*  NPROC0*L0-1      for open boundary conditions,
*
*  NPROC0*L0        for SF, open-SF and periodic boundary conditions.
*
* Note that in the latter cases the points at time T are not in the local
* lattice and are omitted in the programs bnd_pts(), bnd_s2zero() and
* bnd_sd2zero().
*
* The action performed by set_bc() is the following:
*
*  Open bc:         Set all link variables U(x,0) at time T to zero.
*
*  SF bc:           Reads the boundary values of the gauge field from the
*                   parameter data base and assigns them to the link variables
*                   at time 0 and T. The spatial link variables at time T are
*                   stored in the buffers appended to the local field on the
*                   MPI processes where cpr[0]=NPROC0-1.
*
*  Open-SF bc:      Same as SF bc, but omitting the assignment of the link
*                   variables at time 0.
*
*  Periodic bc:     No action is performed.
*
* Then the program checks whether any active link variables are equal to
* zero and, if some are found, aborts the program with an error message.
* An error occurs if set_bc() or check_bc() is called when the gauge field
* is phase-set (see set_ud_phase() [uflds.c]).
*
* The programs in this module act globally and must be called simultaneously
* by the OpenMP master thread on all MPI processes.
*
*******************************************************************************/

#define BCNDS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "uflds.h"
#include "lattice.h"
#include "global.h"

#define N0 (NPROC0*L0)

typedef union
{
   su3_dble u;
   double r[18];
} umat_t;

static int init0=0,nlks,*lks;
static int init1=0,npts,*pts;
static int init2=0;
static int (*ofs_lks)[2];
static int (*ofs_all_pts)[2],(*ofs_even_pts)[2],(*ofs_odd_pts)[2];
static const su3_dble ud0={{0.0}};
static const spinor s0={{{0.0f}}};
static const spinor_dble sd0={{{0.0}}};
static su3_dble ubnd[2][3];


static int *alloc_lks(void)
{
   int k,*a,*b;

   error(ipt==NULL,1,"alloc_lks [bcnds.c]",
         "Geometry arrays are not set");

   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      if (NPROC0>1)
         nlks=(L1*L2*L3)/2;
      else
         nlks=L1*L2*L3;

      lks=malloc(nlks*sizeof(*lks));
      ofs_lks=malloc(NTHREAD*sizeof(*ofs_lks));
      a=malloc(2*NTHREAD*sizeof(*a));
   }
   else
   {
      nlks=0;
      lks=NULL;
      ofs_lks=NULL;
      a=NULL;
   }

   error((nlks>0)&&((lks==NULL)||(ofs_lks==NULL)||(a==NULL)),1,
         "alloc_lks [bcnds.c]","Unable to allocate index array");

   if (nlks>0)
   {
      b=a+NTHREAD;
      divide_range(nlks,NTHREAD,a,b);

      for (k=0;k<NTHREAD;k++)
      {
         ofs_lks[k][0]=a[k];
         ofs_lks[k][1]=b[k]-a[k];
      }
   }

   return a;
}


static void set_lks(void)
{
   int k,l,n,ix,t;
   int *lk,*a;

   a=alloc_lks();

   if (nlks>0)
   {
#pragma omp parallel private(k,l,n,ix,t,lk)
      {
         k=omp_get_thread_num();
         n=0;

         for (ix=(k*(VOLUME_TRD/2));ix<((k+1)*(VOLUME_TRD/2));ix++)
         {
            t=global_time(ix+(VOLUME/2));

            if ((t==0)||(t==(N0-1)))
               n+=1;
         }

         a[k]=n;

#pragma omp barrier
         if (a[k]>0)
         {
            lk=lks;

            for (l=0;l<k;l++)
               lk+=a[l];

            for (ix=(k*(VOLUME_TRD/2));ix<((k+1)*(VOLUME_TRD/2));ix++)
            {
               t=global_time(ix+(VOLUME/2));

               if (t==0)
               {
                  (*lk)=8*ix+1;
                  lk+=1;
               }
               else if (t==(N0-1))
               {
                  (*lk)=8*ix;
                  lk+=1;
               }
            }
         }
      }

      free(a);
   }

   init0=1;
}


static int *alloc_pts(void)
{
   int bc,k,*a,*b;

   error(ipt==NULL,1,"alloc_pts [bcnds.c]",
         "Geometry arrays are not set");
   bc=bc_type();

   if (((cpr[0]==0)&&(bc!=3))||((cpr[0]==(NPROC0-1))&&(bc==0)))
   {
      if ((NPROC0==1)&&(bc==0))
         npts=2*L1*L2*L3;
      else
         npts=L1*L2*L3;

      pts=malloc(npts*sizeof(*pts));
      ofs_all_pts=malloc(3*NTHREAD*sizeof(*ofs_all_pts));
      a=malloc(2*NTHREAD*sizeof(*a));
   }
   else
   {
      npts=0;
      pts=NULL;
      ofs_all_pts=NULL;
      a=NULL;
   }

   error((npts>0)&&((pts==NULL)||(ofs_all_pts==NULL)||(a==NULL)),1,
         "alloc_pts [bcnds.c]","Unable to allocate index array");

   if (npts>0)
   {
      ofs_even_pts=ofs_all_pts+NTHREAD;
      ofs_odd_pts=ofs_even_pts+NTHREAD;
      b=a+NTHREAD;

      divide_range(npts,NTHREAD,a,b);

      for (k=0;k<NTHREAD;k++)
      {
         ofs_all_pts[k][0]=a[k];
         ofs_all_pts[k][1]=b[k]-a[k];
      }

      divide_range(npts/2,NTHREAD,a,b);

      for (k=0;k<NTHREAD;k++)
      {
         ofs_even_pts[k][0]=a[k];
         ofs_even_pts[k][1]=b[k]-a[k];

         ofs_odd_pts[k][0]=a[k]+npts/2;
         ofs_odd_pts[k][1]=b[k]-a[k];
      }
   }

   return a;
}


static void set_pts(void)
{
   int bc,*a,*b;
   int k,l,n,m,ix,t,*pt;

   a=alloc_pts();

   if (npts>0)
   {
      b=a+NTHREAD;
      bc=bc_type();

#pragma omp parallel private(k,l,n,m,ix,t,pt)
      {
         k=omp_get_thread_num();
         n=0;
         m=0;

         for (ix=(k*(VOLUME_TRD/2));ix<((k+1)*(VOLUME_TRD/2));ix++)
         {
            t=global_time(ix);

            if ((t==0)||((t==(N0-1))&&(bc==0)))
               n+=1;

            t=global_time(ix+(VOLUME/2));

            if ((t==0)||((t==(N0-1))&&(bc==0)))
               m+=1;
         }

         a[k]=n;
         b[k]=m;

#pragma omp barrier
         if (a[k]>0)
         {
            pt=pts;

            for (l=0;l<k;l++)
               pt+=a[l];

            for (ix=(k*(VOLUME_TRD/2));ix<((k+1)*(VOLUME_TRD/2));ix++)
            {
               t=global_time(ix);

               if ((t==0)||((t==(N0-1))&&(bc==0)))
               {
                  (*pt)=ix;
                  pt+=1;
               }
            }
         }

         if (b[k]>0)
         {
            pt=pts+(npts/2);

            for (l=0;l<k;l++)
               pt+=b[l];

            for (ix=(k*(VOLUME_TRD/2));ix<((k+1)*(VOLUME_TRD/2));ix++)
            {
               t=global_time(ix+(VOLUME/2));

               if ((t==0)||((t==(N0-1))&&(bc==0)))
               {
                  (*pt)=ix+(VOLUME/2);
                  pt+=1;
               }
            }
         }
      }

      free(a);
   }

   init1=1;
}


int *bnd_lks(int *n)
{
   if (init0==0)
      set_lks();

   (*n)=nlks;

   return lks;
}


int *bnd_pts(int *n)
{
   if (init1==0)
      set_pts();

   (*n)=npts;

   return pts;
}


static int is_zero(su3_dble *u)
{
   int i,it;
   umat_t *um;

   um=(umat_t*)(u);
   it=1;

   for (i=0;i<18;i++)
      it&=((*um).r[i]==0.0);

   return it;
}


static int is_equal(double tol,su3_dble *u,su3_dble *v)
{
   int i,it;
   umat_t *um,*vm;

   um=(umat_t*)(u);
   vm=(umat_t*)(v);
   it=1;

   for (i=0;i<18;i++)
      it&=(fabs((*um).r[i]-(*vm).r[i])<=tol);

   return it;
}


static int check_zero(int bc)
{
   int it,k,ix,t,ifc;
   su3_dble *ub,*ud;

   ub=udfld();
   it=1;

#pragma omp parallel private(k,ix,t,ifc,ud) reduction(& : it)
   {
      k=omp_get_thread_num();
      ud=ub+8*k*(VOLUME_TRD/2);

      for (ix=(k*(VOLUME_TRD/2));ix<((k+1)*(VOLUME_TRD/2));ix++)
      {
         t=global_time(ix+(VOLUME/2));

         if ((bc==0)&&(t==0))
         {
            it&=(0x1^is_zero(ud));
            ud+=1;
            it&=is_zero(ud);
            ud+=1;
         }
         else if ((bc==0)&&(t==(N0-1)))
         {
            it&=is_zero(ud);
            ud+=1;
            it&=(0x1^is_zero(ud));
            ud+=1;
         }
         else
         {
            it&=(0x1^is_zero(ud));
            ud+=1;
            it&=(0x1^is_zero(ud));
            ud+=1;
         }

         for (ifc=2;ifc<8;ifc++)
         {
            it&=(0x1^is_zero(ud));
            ud+=1;
         }
      }
   }

   return it;
}


static void set_ubnd(void)
{
   int i,k;
   double s[3];
   bc_parms_t bcp;

   bcp=bc_parms();
   s[0]=(double)(NPROC1*L1);
   s[1]=(double)(NPROC2*L2);
   s[2]=(double)(NPROC3*L3);

   for (i=0;i<2;i++)
   {
      for (k=0;k<3;k++)
      {
         ubnd[i][k]=ud0;
         ubnd[i][k].c11.re=cos(bcp.phi[i][0]/s[k]);
         ubnd[i][k].c11.im=sin(bcp.phi[i][0]/s[k]);
         ubnd[i][k].c22.re=cos(bcp.phi[i][1]/s[k]);
         ubnd[i][k].c22.im=sin(bcp.phi[i][1]/s[k]);
         ubnd[i][k].c33.re=cos(bcp.phi[i][2]/s[k]);
         ubnd[i][k].c33.im=sin(bcp.phi[i][2]/s[k]);
      }
   }

   init2=1;
}


static void open_bc(void)
{
   int k,*lk,*lkm;
   su3_dble *ub;

   if (init0==0)
      set_lks();

   ub=udfld();

   if (nlks>0)
   {
#pragma omp parallel private(k,lk,lkm)
      {
         k=omp_get_thread_num();

         lk=lks+ofs_lks[k][0];
         lkm=lk+ofs_lks[k][1];

         for (;lk<lkm;lk++)
            ub[*lk]=ud0;
      }
   }

   set_flags(UPDATED_UD);
}


static void SF_bc(void)
{
   int k,*pt,*ptm;
   su3_dble *ub,*ud;

   if (init1==0)
      set_pts();
   if (init2==0)
      set_ubnd();

   ub=udfld();

   if (npts>0)
   {
#pragma omp parallel private(k,pt,ptm,ud)
      {
         k=omp_get_thread_num();

         pt=pts+ofs_odd_pts[k][0];
         ptm=pt+ofs_odd_pts[k][1];

         for (;pt<ptm;pt++)
         {
            ud=ub+8*(pt[0]-(VOLUME/2));

            for (k=0;k<3;k++)
            {
               ud[2+2*k]=ubnd[0][k];
               ud[3+2*k]=ubnd[0][k];
            }
         }
      }
   }

   if (cpr[0]==(NPROC0-1))
   {
      ud=ub+4*VOLUME+7*(BNDRY/4);

      for (k=0;k<3;k++)
         ud[k]=ubnd[1][k];
   }

   set_flags(UPDATED_UD);
}


static void openSF_bc(void)
{
   int k;
   su3_dble *ub,*ud;

   if (init2==0)
      set_ubnd();

   ub=udfld();

   if (cpr[0]==(NPROC0-1))
   {
      ud=ub+4*VOLUME+7*(BNDRY/4);

      for (k=0;k<3;k++)
         ud[k]=ubnd[1][k];
   }

   set_flags(UPDATED_UD);
}


void set_bc(void)
{
   int bc,it;

   error_root(query_flags(UD_PHASE_SET)==1,1,"set_bc [bcnds.c]",
              "Gauge configuration must not be phase-set");
   bc=bc_type();

   if (bc==0)
      open_bc();
   else if (bc==1)
      SF_bc();
   else if (bc==2)
      openSF_bc();

   it=check_zero(bc);
   error(it!=1,1,"set_bc [bcnds.c]",
         "Link variables vanish on an incorrect set of links");
}


static int check_SF(double tol)
{
   int it,k,*pt,*ptm;
   su3_dble *ub,*ud;

   if (init1==0)
      set_pts();
   if (init2==0)
      set_ubnd();

   ub=udfld();
   it=1;

   if (npts>0)
   {
#pragma omp parallel private(k,pt,ptm,ud) reduction(& : it)
      {
         k=omp_get_thread_num();

         pt=pts+ofs_odd_pts[k][0];
         ptm=pt+ofs_odd_pts[k][1];

         for (;pt<ptm;pt++)
         {
            ud=ub+8*(pt[0]-(VOLUME/2));

            for (k=0;k<3;k++)
            {
               it&=is_equal(tol,ud+2+2*k,ubnd[0]+k);
               it&=is_equal(tol,ud+3+2*k,ubnd[0]+k);
            }
         }
      }
   }

   if (cpr[0]==(NPROC0-1))
   {
      ud=ub+4*VOLUME+7*(BNDRY/4);

      for (k=0;k<3;k++)
         it&=is_equal(tol,ud+k,ubnd[1]+k);
   }

   return it;
}


static int check_openSF(double tol)
{
   int it,k;
   su3_dble *ub,*ud;

   if (init2==0)
      set_ubnd();

   ub=udfld();
   it=1;

   if (cpr[0]==(NPROC0-1))
   {
      ud=ub+4*VOLUME+7*(BNDRY/4);

      for (k=0;k<3;k++)
         it&=is_equal(tol,ud+k,ubnd[1]+k);
   }

   return it;
}


int check_bc(double tol)
{
   int bc,it,is;
   double dprms[1];

   error_root(query_flags(UD_PHASE_SET)==1,1,"check_bc [bcnds.c]",
              "Gauge configuration must not be phase-set");

   if (NPROC>1)
   {
      dprms[0]=tol;
      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      error(dprms[0]!=tol,1,"check_bc [bcnds.c]","Parameter is not global");
   }

   bc=bc_type();
   it=check_zero(bc);

   if (bc==1)
      it&=check_SF(tol);
   else if (bc==2)
      it&=check_openSF(tol);

   if (NPROC>1)
   {
      is=it;
      MPI_Allreduce(&is,&it,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
   }

   return it;
}


void bnd_s2zero(ptset_t set,spinor *s)
{
   int k,*pt,*ptm;

   if (init1==0)
      set_pts();

   if (npts>0)
   {
#pragma omp parallel private(k,pt,ptm)
      {
         k=omp_get_thread_num();

         if (set==ALL_PTS)
         {
            pt=pts+ofs_all_pts[k][0];
            ptm=pt+ofs_all_pts[k][1];
         }
         else if (set==EVEN_PTS)
         {
            pt=pts+ofs_even_pts[k][0];
            ptm=pt+ofs_even_pts[k][1];
         }
         else if (set==ODD_PTS)
         {
            pt=pts+ofs_odd_pts[k][0];
            ptm=pt+ofs_odd_pts[k][1];
         }
         else
         {
            pt=pts;
            ptm=pt;
         }

         for (;pt<ptm;pt++)
            s[*pt]=s0;
      }
   }
}


void bnd_sd2zero(ptset_t set,spinor_dble *sd)
{
   int k,*pt,*ptm;

   if (init1==0)
      set_pts();

   if (npts>0)
   {
#pragma omp parallel private(k,pt,ptm)
      {
         k=omp_get_thread_num();

         if (set==ALL_PTS)
         {
            pt=pts+ofs_all_pts[k][0];
            ptm=pt+ofs_all_pts[k][1];
         }
         else if (set==EVEN_PTS)
         {
            pt=pts+ofs_even_pts[k][0];
            ptm=pt+ofs_even_pts[k][1];
         }
         else if (set==ODD_PTS)
         {
            pt=pts+ofs_odd_pts[k][0];
            ptm=pt+ofs_odd_pts[k][1];
         }
         else
         {
            pt=pts;
            ptm=pt;
         }

         for (;pt<ptm;pt++)
            sd[*pt]=sd0;
      }
   }
}
