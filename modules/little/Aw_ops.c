
/*******************************************************************************
*
* File Aw_ops.c
*
* Copyright (C) 2011-2013, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Allocation and computation of the little Dirac operator.
*
*   Aw_t Awop(void)
*     Returns a structure containing the matrices that describe the
*     single-precision little Dirac operator.
*
*   Aw_t Awophat(void)
*     Returns a structure containing the matrices that describe the
*     single-precision even-odd preconditioned little Dirac operator.
*
*   Aw_dble_t Awop_dble(void)
*     Returns a structure containing the matrices that describe the
*     double-precision little Dirac operator.
*
*   Aw_dble_t Awophat_dble(void)
*     Returns a structure containing the matrices that describe the
*     double-precision even-odd preconditioned little Dirac operator.
*
*   void set_Aw(double mu)
*     Computes the single- and the double-precision little Dirac operator.
*     The SW term is updated if needed and the twisted mass is set to mu.
*     If the twisted-mass flag is set, the twisted-mass term is switched
*     off on the odd sites of the lattice.
*
*   int set_Awhat(double mu)
*     Computes the single- and the double-precision even-odd preconditioned
*     little Dirac operator. The program calls set_Aw(mu) and thus updates
*     the operator w/o even-odd preconditioning too. The little modes are
*     updated as well (see ltl_modes.c). On exit the program returns 0 if
*     all matrix inversions were safe and 1 if not.
*
* For a description of the little Dirac operator and the associated data
* structures see README.Aw. The twisted-mass flag is retrieved from the
* parameter data base (see flags/lat_parms.c).
*
* The inversion of a double-precision complex matrix is considered to be
* safe if and only if its Frobenius condition number is less than 10^6.
* The program set_Awhat() calls set_ltl_modes() [ltl_modes.c], which
* requires a workspace of 2 double-precision complex vector fields.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define AW_OPS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "vflds.h"
#include "linalg.h"
#include "block.h"
#include "dfl.h"
#include "little.h"
#include "global.h"

#define MAX_FROBENIUS 1.0e6

static int Ns=0,nb,old_eo;
static double old_m0,old_mu;
static Aw_t Aws={0,0,NULL,NULL},Awshat={0,0,NULL,NULL};
static Aw_dble_t Awd={0,0,NULL,NULL},Awdhat={0,0,NULL,NULL};


static void check_setup(void)
{
   int isw;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   Ns=dfl.Ns;
   error(Ns==0,1,"check_setup [Aw_ops.c]",
         "The deflation parameters are not set");

   (void)(blk_list(DFL_BLOCKS,&nb,&isw));
   error(nb==0,1,"check_setup [Aw_ops.c]",
         "The DFL_BLOCKS block grid is not allocated");
}


static void alloc_Awd(Aw_dble_t *Aw)
{
   int n,nmat;
   complex_dble **ww,*w;

   if (Ns==0)
      check_setup();

   nmat=Ns*Ns;
   ww=malloc(9*nb*sizeof(*ww));
   w=amalloc(9*nb*nmat*sizeof(*w),6);
   error((ww==NULL)||(w==NULL),1,"alloc_Awd [Aw_ops.c]",
         "Unable to allocate matrix arrays");

   for (n=0;n<(9*nb);n++)
   {
      ww[n]=w;
      w+=nmat;
   }

   (*Aw).Ns=Ns;
   (*Aw).nb=nb;
   (*Aw).Ablk=ww;
   (*Aw).Ahop=ww+nb;
}


static void alloc_Aws(Aw_t *Aw)
{
   int n,nmat;
   complex **ww,*w;

   if (Ns==0)
      check_setup();

   nmat=Ns*Ns;
   ww=malloc(9*nb*sizeof(*ww));
   w=amalloc(9*nb*nmat*sizeof(*w),6);
   error((ww==NULL)||(w==NULL),1,"alloc_Aws [Aw_ops.c]",
         "Unable to allocate matrix arrays");

   for (n=0;n<(9*nb);n++)
   {
      ww[n]=w;
      w+=nmat;
   }

   (*Aw).Ns=Ns;
   (*Aw).nb=nb;
   (*Aw).Ablk=ww;
   (*Aw).Ahop=ww+nb;
}


Aw_dble_t Awop_dble(void)
{
   if (Awd.Ns==0)
      alloc_Awd(&Awd);

   return Awd;
}


Aw_dble_t Awophat_dble(void)
{
   if (Awdhat.Ns==0)
      alloc_Awd(&Awdhat);

   return Awdhat;
}


Aw_t Awop(void)
{
   if (Aws.Ns==0)
      alloc_Aws(&Aws);

   return Aws;
}


Aw_t Awophat(void)
{
   if (Awshat.Ns==0)
      alloc_Aws(&Awshat);

   return Awshat;
}


void set_Aw(double mu)
{
   int n;
   double dprms[1];

   if (NPROC>1)
   {
      dprms[0]=mu;
      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      error(dprms[0]!=mu,1,
            "set_Aw [Aw_ops.c]","Parameters are not global");
   }

   if (query_flags(AW_UP2DATE)==1)
   {
      if (update_Awblk(mu))
      {
         n=(nb*Ns*Ns)/NTHREAD;
         assign_vd2v(n,2,Awd.Ablk[0],Aws.Ablk[0]);
      }
   }
   else
   {
      set_Awblk(mu);
      set_Awhop();

      if (Aws.Ns==0)
         alloc_Aws(&Aws);

      n=(nb*Ns*Ns)/NTHREAD;
      assign_vd2v(9*n,2,Awd.Ablk[0],Aws.Ablk[0]);
   }

   set_flags(COMPUTED_AW);
}


int set_Awhat(double mu)
{
   int nbt,eo,ifail;
   int k,n,ifc;
   double m0,cn;
   complex_dble **Ab0,**Ab1,**Ah0,**Ah1;
   sw_parms_t swp;
   tm_parms_t tm;
   cmat_wsp_t *cwsp;

   set_Aw(mu);

   tm=tm_parms();
   swp=sw_parms();
   eo=tm.eoflg;
   m0=swp.m0;

   if (query_flags(AWHAT_UP2DATE)==1)
   {
      if ((eo==old_eo)&&(m0==old_m0)&&(mu==old_mu))
         return 0;
   }
   else
   {
      if (Awdhat.Ns==0)
         alloc_Awd(&Awdhat);
      if (Awshat.Ns==0)
         alloc_Aws(&Awshat);
   }

   nbt=nb/NTHREAD;
   ifail=0;

#pragma omp parallel private(k,n,ifc,cn,cwsp,Ab0,Ab1,Ah0,Ah1) \
   reduction(| : ifail)
   {
      cwsp=alloc_cmat_wsp(Ns);

      if (cwsp==NULL)
         ifail=1;
      else
      {
         k=omp_get_thread_num();
         n=k*nbt;
         Ab0=Awd.Ablk+n;
         Ab1=Awdhat.Ablk+n;
         Ah0=Awd.Ahop+8*n;
         Ah1=Awdhat.Ahop+8*n;

         for (n=0;n<nbt;n++)
         {
            ifail|=cmat_inv_dble(Ns,*Ab0,cwsp,*Ab1,&cn);
            ifail|=(cn>MAX_FROBENIUS);

            for (ifc=0;ifc<8;ifc++)
            {
               cmat_mul_dble(Ns,*Ab1,*Ah0,*Ah1);
               Ah0+=1;
               Ah1+=1;
            }

            Ab0+=1;
            Ab1+=1;
         }

         free_cmat_wsp(cwsp);
      }
   }

   n=(nbt*Ns*Ns);
   assign_vd2v(9*n,2,Awdhat.Ablk[0],Awshat.Ablk[0]);

   ifail|=set_ltl_modes();

   if (NPROC>1)
   {
      n=ifail;
      MPI_Allreduce(&n,&ifail,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
   }

   old_eo=eo;
   old_m0=m0;
   old_mu=mu;

   set_flags(COMPUTED_AWHAT);

   return ifail;
}
