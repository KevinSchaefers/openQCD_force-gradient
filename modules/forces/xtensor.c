
/*******************************************************************************
*
* File xtensor.c
*
* Copyright (C) 2011-2013, 2018, 2021 Martin Luescher, Antonio Rago
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Spin parts of the quark forces.
*
*   u3_alg_dble **xtensor(void)
*     Returns the pointers xt[0],..,xt[5] to the X tensor field components
*     with Lorentz indices (0,1),(0,2),(0,3),(2,3),(3,1),(1,2). The arrays
*     are automatically allocated and initialized to zero if they are not
*     already allocated.
*
*   void set_xt2zero(void)
*     Sets the X tensor field to zero.
*
*   int add_det2xt(double c,ptset_t set)
*     Computes the spin part of the SW force deriving from the action
*     -Tr{ln(D)}, where D=D_ee,D_oo,D_ee+D_oo or 1 when set=EVEN_PTS,
*     ODD_PTS,ALL_PTS or NO_PTS (see the notes). The calculated matrices
*     are then multiplied by c and are added to the X tensor field. When
*     needed, the program recomputes and inverts the SW term. The program
*     returns 0 if all inversions were safe and a non-zero value otherwise.
*
*   void add_prod2xt(double c,spinor_dble *r,spinor_dble *s)
*     Computes the spin part of the SW force deriving from the "action"
*     -2*Re(r,gamma_5*Dw*s), where Dw denotes the lattice Dirac operator
*     (see the notes). The calculated matrices are then multiplied by c
*     and are added to the X tensor field.
*
*   su3_dble *xvector(void)
*     Returns the pointer xv to the X vector field. The components of
*     field are stored in memory in the same order as the link variables.
*     The array automatically allocated and initialized to zero if it is
*     not already allocated.
*
*   void set_xv2zero(void)
*     Sets the X vector field to zero.
*
*   void add_prod2xv(double c,spinor_dble *r,spinor_dble *s)
*     Computes the spin part of the force deriving from the hopping terms
*     in the "action" -2*Re(r,gamma_5*Dw*s), where Dw denotes the lattice
*     Dirac operator (see the notes). The calculated matrices are then
*     multiplied by c and are added to the X vector field.
*
* The computation of the quark forces is described in "Molecular-dynamics
* quark forces" [doc/forces.pdf]. For unexplained notation concerning the
* SW term see "Implementation of the lattice Dirac operator" [doc/dirac.pdf].
*
* In the case of the traditional form of the SW term, the contribution of the
* determinant of the diagonal blocks D_ee,D_oo of the Dirac operator to the
* n'th component of the X tensor at the point x is given by
*
*  X[n]=i*tr{sigma_{mu,nu}*M(x)^(-1)},
*
* where M(x) is the 12x12 matrix representing the SW term at x and n labels
* the (mu,nu)=(0,1),(0,2),(0,3),(2,3),(3,1),(1,2) index pairs. Similarly, for
* given spinor fields r and s, the associated X tensor is defined by
*
*  X[n]=i*tr{[gamma_5*sigma_{mu,nu}*s(x) x r^dag(x)]+(s<->r)}.
*
* If the "exponential" variant of the SW term is chosen, the determinant of
* the diagonal is independent of the gauge field and does not contribute to
* the X tensor (thus add_det2xt() does nothing). The program add_prod2xt()
* adds 6 terms to the X tensor in this case, given by
*
*  i*tr{[gamma_5*sigma_{mu,nu}*s_k(x) x r_k^dag(x)]+(s_k<->r_k)}, k=0,..,5,
*
* where s_k and r_k are obtained from s and r and the Pauli matrices at the
* point x (see sw_term/swexp.c and sw_term/sw_term.c).
*
* The contribution of the fields r,s to the X vector component on the link
* (x,x+mu) is given by
*
*  X=tr{[gamma_5*(1-gamma_mu)*s(x+mu) x r^dag(x)]+(s<->r)}
*
* In all cases, the trace is taken over the Dirac indices only.
*
* The components of the X tensor field are of type u3_alg_dble. As in the
* case of symmetric gauge-field tensor, the field array includes additional
* space for the field components on the boundaries of the local lattice
* (see tcharge/ftensor.c and lattice/README.ftidx). The type u3_alg_dble
* is explained in the module su3fcts/su3prod.c.
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously. Once they have previously been
* called, the programs xtensor() and xvector() are thread-safe.
*
*******************************************************************************/

#define XTENSOR_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"
#include "flags.h"
#include "lattice.h"
#include "su3fcts.h"
#include "sw_term.h"
#include "sflds.h"
#include "linalg.h"
#include "tcharge.h"
#include "forces.h"
#include "global.h"

#define N0 (NPROC0*L0)

typedef union
{
   spinor_dble s;
   weyl_dble w[2];
} spin_dble;

static su3_dble *xvs=NULL,*xv;
static u3_alg_dble **xts=NULL,**xt;


static void alloc_xts(void)
{
   int n,nt,nxt[6];
   u3_alg_dble **pp,*p;
   ftidx_t *idx;

   idx=ftidx();
   nt=0;

   for (n=0;n<6;n++)
   {
      nxt[n]=VOLUME+idx[n].nft[0]+idx[n].nft[1];
      nt+=nxt[n];
   }

   pp=malloc(12*sizeof(*pp));
   p=amalloc(nt*sizeof(*p),5);
   error((pp==NULL)||(p==NULL),1,"alloc_xts [xtensor.c]",
         "Unable to allocate field arrays");

   xts=pp;
   xt=pp+6;

   for (n=0;n<6;n++)
   {
      (*pp)=p;
      pp+=1;
      p+=nxt[n];
   }

   set_xt2zero();
}


u3_alg_dble **xtensor(void)
{
   int n;

   if (xts==NULL)
      alloc_xts();

   for (n=0;n<6;n++)
      xt[n]=xts[n];

   return xt;
}


void set_xt2zero(void)
{
   int n;

   if (xts==NULL)
      alloc_xts();
   else
   {
      for (n=0;n<6;n++)
         set_ualg2zero(VOLUME_TRD,2,xts[n]);
   }
}


static void add_det2xt_part(int ofs,int vol,double c)
{
   int n;
   u3_alg_dble X[6],*xtr[6];
   pauli_dble *m,*mm;

   for (n=0;n<6;n++)
      xtr[n]=xts[n]+ofs;

   m=swdfld()+2*ofs;
   mm=m+2*vol;

   for (;m<mm;m+=2)
   {
      det2xt(m,X);

      for (n=0;n<6;n++)
      {
         _u3_alg_mul_add_assign(xtr[n][0],c,X[n]);
         xtr[n]+=1;
      }
   }
}


int add_det2xt(double c,ptset_t set)
{
   int k,ofs,vol,ifail;
   sw_parms_t swp;

   swp=sw_parms();

   if ((set==NO_PTS)||(swp.isw))
      return 0;

   ifail=sw_term(set);

   if (ifail!=0)
      return ifail;

   if (xts==NULL)
      alloc_xts();

#pragma omp parallel private(k,ofs,vol)
   {
      k=omp_get_thread_num();

      if (set==EVEN_PTS)
      {
         vol=(VOLUME_TRD/2);
         ofs=k*vol;
      }
      else if (set==ODD_PTS)
      {
         vol=(VOLUME_TRD/2);
         ofs=(VOLUME/2)+k*vol;
      }
      else
      {
         vol=VOLUME_TRD;
         ofs=k*vol;
      }

      add_det2xt_part(ofs,vol,c);
   }

   return 0;
}


static void set_rksk0(pauli_dble *ms,spinor_dble *r,spinor_dble *s,
                      weyl_dble (*rk)[6],spin_dble *wk)
{
   int k;
   spin_dble *rs;

   rs=(spin_dble*)(r);
   rk[0][0]=(*rs).w[0];
   rk[1][0]=(*rs).w[1];

   for (k=1;k<6;k++)
   {
      mul_pauli_dble(0.0,ms  ,rk[0]+k-1,rk[0]+k);
      mul_pauli_dble(0.0,ms+1,rk[1]+k-1,rk[1]+k);
   }

   rs=(spin_dble*)(s);
   wk[1].w[0]=(*rs).w[0];
   wk[1].w[1]=(*rs).w[1];
}


static void set_wk(int k,double (*qs)[36],pauli_dble *ms,weyl_dble (*rk)[6],
                   spin_dble *wk)
{
   cmb6weyl(qs[0]+6*k,rk[0],wk[0].w);
   cmb6weyl(qs[1]+6*k,rk[1],wk[0].w+1);

   if (k>0)
   {
      mul_pauli_dble(0.0,ms  ,wk[1].w  ,wk[1].w  );
      mul_pauli_dble(0.0,ms+1,wk[1].w+1,wk[1].w+1);
   }
}


static void add_prod2xt0_part(int ofs,int vol,double c,spinor_dble *r,
                              spinor_dble *s)
{
   int n;
   u3_alg_dble X[6],*xtr[6];
   spinor_dble *rm;

   for (n=0;n<6;n++)
      xtr[n]=xts[n]+ofs;

   r+=ofs;
   s+=ofs;
   rm=r+vol;

   for (;r<rm;r++)
   {
      prod2xt(r,s,X);

      for (n=0;n<6;n++)
      {
         _u3_alg_mul_add_assign(xtr[n][0],c,X[n]);
         xtr[n]+=1;
      }

      s+=1;
   }
}


static void add_prod2xt0(double c,spinor_dble *r,spinor_dble *s)
{
   int k,ofs,vol;

   if (xts==NULL)
      alloc_xts();

#pragma omp parallel private(k,ofs,vol)
   {
      k=omp_get_thread_num();
      vol=VOLUME_TRD;
      ofs=k*vol;
      add_prod2xt0_part(ofs,vol,c,r,s);
   }
}


static int add_prod2xt1_part(int ofs,int vol,double c,spinor_dble *r,
                             spinor_dble *s)
{
   int N,bc,ix,t,n,l;
   double c0,c1,c2[2],qs[2][36];
   u3_alg_dble X[6],*xtr[6],*ftr[6],**fts;
   pauli_dble ms[2] ALIGNED32;
   weyl_dble rk[2][6] ALIGNED32;
   spin_dble wk[2] ALIGNED32;
   sw_parms_t swp;
   swexp_wsp_t *swsp;

   swp=sw_parms();

   c1=-0.5*swp.csw/(4.0+swp.m0);
   c2[0]=c*(1.0+(swp.cF[0]-1.0)/(4.0+swp.m0));
   c2[1]=c*(1.0+(swp.cF[1]-1.0)/(4.0+swp.m0));

   bc=bc_type();
   N=sw_order();
   swsp=alloc_swexp_wsp(N);
   if (swsp==NULL)
      return 1;

   fts=ftensor();

   for (n=0;n<6;n++)
   {
      xtr[n]=xts[n]+ofs;
      ftr[n]=fts[n]+ofs;
   }

   r+=ofs;
   s+=ofs;

   for (ix=ofs;ix<(ofs+vol);ix++)
   {
      t=global_time(ix);

      if ((t==1)&&(bc!=3))
         c0=c2[0];
      else if (((t==(N0-2))&&(bc==0))||((t==(N0-1))&&((bc==1)||(bc==2))))
         c0=c2[1];
      else
         c0=c;

      pauli_term(c1,ftr,ms);
      sw_dexp(ms  ,c0,swsp,qs[0]);
      sw_dexp(ms+1,c0,swsp,qs[1]);
      set_rksk0(ms,r,s,rk,wk);

      for (l=0;l<6;l++)
      {
         set_wk(l,qs,ms,rk,wk);
         prod2xt(&(wk[0].s),&(wk[1].s),X);

         for (n=0;n<6;n++)
         {
            _u3_alg_add_assign(xtr[n][0],X[n]);
         }
      }

      for (n=0;n<6;n++)
      {
         ftr[n]+=1;
         xtr[n]+=1;
      }

      r+=1;
      s+=1;
   }

   free_swexp_wsp(swsp);

   return 0;
}


static void add_prod2xt1(double c,spinor_dble *r,spinor_dble *s)
{
   int k,ofs,vol,ie;

   if (xts==NULL)
      alloc_xts();

   (void)(ftensor());
   ie=0;

#pragma omp parallel private(k,ofs,vol) reduction (| : ie)
   {
      k=omp_get_thread_num();
      vol=VOLUME_TRD;
      ofs=k*vol;
      ie=add_prod2xt1_part(ofs,vol,c,r,s);
   }

   error_loc(ie,1,"add_prod2xt1() [xtensor.c]",
             "Unable to allocate swexp workspace");
}


void add_prod2xt(double c,spinor_dble *r,spinor_dble *s)
{
   sw_parms_t swp;

   if (xts==NULL)
      alloc_xts();

   swp=sw_parms();

   if (swp.isw==0)
      add_prod2xt0(c,r,s);
   else
      add_prod2xt1(c,r,s);
}


static void alloc_xvs(void)
{
   xvs=amalloc(4*VOLUME*sizeof(*xv),5);
   error(xvs==NULL,1,"alloc_xvs [xtensor.c]",
         "Unable to allocate field array");
   set_xv2zero();
}


su3_dble *xvector(void)
{
   if (xvs==NULL)
      alloc_xvs();

   return xvs;
}


void set_xv2zero(void)
{
   int k,ofs,vol;

   if (xvs==NULL)
      alloc_xvs();
   else
   {
#pragma omp parallel private(k,ofs,vol)
      {
         k=omp_get_thread_num();
         vol=4*VOLUME_TRD;
         ofs=k*vol;
         cm3x3_zero(vol,xvs+ofs);
      }
   }
}


static void add_prod2xv_part(int ofs,int vol,double c,spinor_dble *r,
                             spinor_dble *s)
{
   int mu,*piup,*pidn;
   su3_dble ws ALIGNED16;
   su3_dble *xvr,*xvm;
   spinor_dble *ro,*so;

   piup=iup[(VOLUME/2)+ofs];
   pidn=idn[(VOLUME/2)+ofs];

   ro=r+(VOLUME/2)+ofs;
   so=s+(VOLUME/2)+ofs;

   xvr=xvs+8*ofs;
   xvm=xvr+8*vol;

   while (xvr<xvm)
   {
      for (mu=0;mu<4;mu++)
      {
         prod2xv[mu](ro,r+(*piup),so,s+(*piup),&ws);
         cm3x3_mulr_add(&c,&ws,xvr);

         piup+=1;
         xvr+=1;

         prod2xv[mu](r+(*pidn),ro,s+(*pidn),so,&ws);
         cm3x3_mulr_add(&c,&ws,xvr);

         pidn+=1;
         xvr+=1;
      }

      ro+=1;
      so+=1;
   }
}


void add_prod2xv(double c,spinor_dble *r,spinor_dble *s)
{
   int k,ofs,vol;

   if (xvs==NULL)
      alloc_xvs();

   cpsd_int_bnd(0x1,r);
   cpsd_int_bnd(0x1,s);

#pragma omp parallel private(k,ofs,vol)
   {
      k=omp_get_thread_num();
      vol=VOLUME_TRD/2;
      ofs=k*vol;
      add_prod2xv_part(ofs,vol,c,r,s);
   }
}
