
/*******************************************************************************
*
* File cgne.c
*
* Copyright (C) 2005-2011, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic CG solver program for the lattice Dirac equation.
*
*   double cgne(int vol,void (*Dop)(spinor *s,spinor *r),
*               void (*Dop_dble)(spinor_dble *s,spinor_dble *r),
*               spinor **ws,spinor_dble **wsd,int nmx,int istop,double res,
*               spinor_dble *eta,spinor_dble *psi,int *status)
*     Solution of the (normal) Dirac equation D^dag*D*psi=eta for given
*     source eta, using the CG algorithm. See the notes for the explanation
*     of the parameters of the program.
*
* This program uses single-precision arithmetic to reduce the execution
* time but obtains the solution with double-precision accuracy.
*
* The programs for the single- and double-precision implementations of the
* Dirac operator are assumed to have the following properties:
*
*   void Dop(spinor *s,spinor *r)
*     Application the (global) operator D or its Hermitian conjugate D^dag
*     to the single-precision Dirac field s and assignment of the result
*     to r. D and D^dag are applied alternatingly, i.e. the first call of
*     the program applies D, the next call D^dag, then D again and so on.
*     In all cases the source field s is unchanged.
*
*   void Dop_dble(spinor_dble *s,spinor_dble *r)
*     Double-precision version of Dop().
*
* The other parameters of the program cgne() are:
*
*   vol     Number of elements per OpenMP thread of the spinor fields
*           on which Dop() and Dop_dble() act.
*
*   nmx     Maximal total number of CG iterations that may be applied.
*
*   istop   Stopping criterion (0: L_2 norm based, 1: uniform norm based).
*
*   res     Desired maximal relative residue |eta-D^dag*D*psi|/|eta| of
*           the calculated solution.
*
*   ws      Array of at least 5 single-precision spinor fields (used
*           as work space).
*
*   wsd     Array of at least 2 double-precision spinor fields (used
*           as work space).
*
*   eta     Source field (unchanged on exit).
*
*   psi     Calculated approximate solution of the normal Dirac equation
*           D^dag*D*psi=eta.
*
*   status  If the program terminates normally, status reports the total
*           number of CG iterations that were required for the solution of
*           the Dirac equation. Otherwise status is set to -1.
*
* Independently of whether the program succeeds in solving the equation to
* the desired accuracy, the program assigns the last calculated approximate
* solution to psi and returns the norm of the residue of that field.
*
* The fields eta and psi as well as the fields in the workspaces are assumed
* to be such that the Dirac operators can act on them.
*
* Some debugging output is printed to stdout on process 0 if the macro
* CGNE_DBG is defined.
*
* The program cgne() is assumed to be called by the OpenMP master thread
* and on all MPI processes simultaneously.
*
* If SSE (AVX) instructions are used, the Dirac spinors must be aligned to
* a 16 (32) byte boundary.
*
*******************************************************************************/

#define CGNE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "mpi.h"
#include "utils.h"
#include "sflds.h"
#include "linalg.h"
#include "linsolv.h"
#include "global.h"

#define PRECISION_LIMIT ((double)(512.0f*FLT_EPSILON))

static float rsq,rsq_old,ai,bi;
static spinor *psx,*psr,*psp,*psap,*psw;
static spinor_dble *pdb,*pdx,*pdw,*pdv;

#if (defined x64)
#include "sse2.h"

static void loc_update_g(int ofs,int vol)
{
   float c;
   spinor *r,*s,*sm;

   c=-ai;

   __asm__ __volatile__ ("movss %0, %%xmm6 \n\t"
                         "shufps $0x0, %%xmm6, %%xmm6 \n\t"
                         "movaps %%xmm6, %%xmm7 \n\t"
                         "movaps %%xmm6, %%xmm8"
                         :
                         :
                         "m" (c)
                         :
                         "xmm6", "xmm7", "xmm8");

   r=psr+ofs;
   s=psap+ofs;
   sm=s+vol;

   for (;s<sm;)
   {
      _sse_spinor_load(*s);

      s+=4;
      _prefetch_spinor(s);
      s-=3;

      __asm__ __volatile__ ("mulps %%xmm6, %%xmm0 \n\t"
                            "mulps %%xmm7, %%xmm1 \n\t"
                            "mulps %%xmm8, %%xmm2 \n\t"
                            "mulps %%xmm6, %%xmm3 \n\t"
                            "mulps %%xmm7, %%xmm4 \n\t"
                            "mulps %%xmm8, %%xmm5 \n\t"
                            "addps %0, %%xmm0 \n\t"
                            "addps %2, %%xmm1 \n\t"
                            "addps %4, %%xmm2"
                            :
                            :
                            "m" ((*r).c1.c1),
                            "m" ((*r).c1.c2),
                            "m" ((*r).c1.c3),
                            "m" ((*r).c2.c1),
                            "m" ((*r).c2.c2),
                            "m" ((*r).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      r+=4;
      _prefetch_spinor(r);
      r-=4;

      __asm__ __volatile__ ("addps %0, %%xmm3 \n\t"
                            "addps %2, %%xmm4 \n\t"
                            "addps %4, %%xmm5"
                            :
                            :
                            "m" ((*r).c3.c1),
                            "m" ((*r).c3.c2),
                            "m" ((*r).c3.c3),
                            "m" ((*r).c4.c1),
                            "m" ((*r).c4.c2),
                            "m" ((*r).c4.c3)
                            :
                            "xmm3", "xmm4", "xmm5");

      _sse_spinor_store(*r);

      r+=1;
   }
}


static void loc_update_xp(int ofs,int vol)
{
   spinor *r,*s,*t,*tm;

   __asm__ __volatile__ ("movss %0, %%xmm6 \n\t"
                         "movss %1, %%xmm9 \n\t"
                         "shufps $0x0, %%xmm6, %%xmm6 \n\t"
                         "shufps $0x0, %%xmm9, %%xmm9 \n\t"
                         "movaps %%xmm6, %%xmm7 \n\t"
                         "movaps %%xmm9, %%xmm10 \n\t"
                         "movaps %%xmm6, %%xmm8 \n\t"
                         "movaps %%xmm9, %%xmm11"
                         :
                         :
                         "m" (ai),
                         "m" (bi)
                         :
                         "xmm6", "xmm7", "xmm8", "xmm9",
                         "xmm10", "xmm11");

   r=psr+ofs;
   s=psp+ofs;
   t=psx+ofs;
   tm=t+vol;

   for (;t<tm;t++)
   {
      _sse_spinor_load(*s);

      s+=4;
      _prefetch_spinor(s);
      s-=4;

      __asm__ __volatile__ ("mulps %%xmm6, %%xmm0 \n\t"
                            "mulps %%xmm7, %%xmm1 \n\t"
                            "mulps %%xmm8, %%xmm2 \n\t"
                            "mulps %%xmm6, %%xmm3 \n\t"
                            "mulps %%xmm7, %%xmm4 \n\t"
                            "mulps %%xmm8, %%xmm5 \n\t"
                            "addps %0, %%xmm0 \n\t"
                            "addps %2, %%xmm1 \n\t"
                            "addps %4, %%xmm2"
                            :
                            :
                            "m" ((*t).c1.c1),
                            "m" ((*t).c1.c2),
                            "m" ((*t).c1.c3),
                            "m" ((*t).c2.c1),
                            "m" ((*t).c2.c2),
                            "m" ((*t).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      t+=4;
      _prefetch_spinor(t);
      t-=4;

      __asm__ __volatile__ ("addps %0, %%xmm3 \n\t"
                            "addps %2, %%xmm4 \n\t"
                            "addps %4, %%xmm5"
                            :
                            :
                            "m" ((*t).c3.c1),
                            "m" ((*t).c3.c2),
                            "m" ((*t).c3.c3),
                            "m" ((*t).c4.c1),
                            "m" ((*t).c4.c2),
                            "m" ((*t).c4.c3)
                            :
                            "xmm3", "xmm4", "xmm5");

      _sse_spinor_store(*t);
      _sse_spinor_load(*s);

      r+=4;
      _prefetch_spinor(r);
      r-=4;

      __asm__ __volatile__ ("mulps %%xmm9, %%xmm0 \n\t"
                            "mulps %%xmm10, %%xmm1 \n\t"
                            "mulps %%xmm11, %%xmm2 \n\t"
                            "mulps %%xmm9, %%xmm3 \n\t"
                            "mulps %%xmm10, %%xmm4 \n\t"
                            "mulps %%xmm11, %%xmm5 \n\t"
                            "addps %0, %%xmm0 \n\t"
                            "addps %2, %%xmm1 \n\t"
                            "addps %4, %%xmm2"
                            :
                            :
                            "m" ((*r).c1.c1),
                            "m" ((*r).c1.c2),
                            "m" ((*r).c1.c3),
                            "m" ((*r).c2.c1),
                            "m" ((*r).c2.c2),
                            "m" ((*r).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      __asm__ __volatile__ ("addps %0, %%xmm3 \n\t"
                            "addps %2, %%xmm4 \n\t"
                            "addps %4, %%xmm5"
                            :
                            :
                            "m" ((*r).c3.c1),
                            "m" ((*r).c3.c2),
                            "m" ((*r).c3.c3),
                            "m" ((*r).c4.c1),
                            "m" ((*r).c4.c2),
                            "m" ((*r).c4.c3)
                            :
                            "xmm3", "xmm4", "xmm5");

      _sse_spinor_store(*s);

      r+=1;
      s+=1;
   }
}

#else

static void loc_update_g(int ofs,int vol)
{
   float c;
   spinor *r,*s,*sm;

   c=-ai;
   r=psr+ofs;
   s=psap+ofs;
   sm=s+vol;

   for (;s<sm;s++)
   {
      _vector_mulr_assign((*r).c1,c,(*s).c1);
      _vector_mulr_assign((*r).c2,c,(*s).c2);
      _vector_mulr_assign((*r).c3,c,(*s).c3);
      _vector_mulr_assign((*r).c4,c,(*s).c4);

      r+=1;
   }
}


static void loc_update_xp(int ofs,int vol)
{
   spinor *r,*s,*t,*tm;

   r=psr+ofs;
   s=psp+ofs;
   t=psx+ofs;
   tm=t+vol;

   for (;t<tm;t++)
   {
      _vector_mulr_assign((*t).c1,ai,(*s).c1);
      _vector_mulr_assign((*t).c2,ai,(*s).c2);
      _vector_mulr_assign((*t).c3,ai,(*s).c3);
      _vector_mulr_assign((*t).c4,ai,(*s).c4);

      _vector_mulr_add((*s).c1,bi,(*r).c1);
      _vector_mulr_add((*s).c2,bi,(*r).c2);
      _vector_mulr_add((*s).c3,bi,(*r).c3);
      _vector_mulr_add((*s).c4,bi,(*r).c4);

      r+=1;
      s+=1;
   }
}

#endif

static void update_g(int vol)
{
   int k;

#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_update_g(k*vol,vol);
      }
}


static void update_xp(int vol)
{
   int k;

#pragma omp parallel private(k)
   {
      k=omp_get_thread_num();
      loc_update_xp(k*vol,vol);
   }
}


static void cg_init(int vol,spinor **ws,spinor_dble **wsd,
                    spinor_dble *eta,spinor_dble *psi)
{
   psx=ws[0];
   psr=ws[1];
   psp=ws[2];
   psap=ws[3];
   psw=ws[4];

   pdb=eta;
   pdx=psi;
   pdw=wsd[0];
   pdv=wsd[1];

   set_s2zero(vol,2,psx);
   assign_sd2s(vol,2,pdb,psr);
   assign_s2s(vol,2,psr,psp);
   set_sd2zero(vol,2,pdx);

   rsq=norm_square(vol,3,psr);
}


static void cg_step(int vol,void (*Dop)(spinor *s,spinor *r))
{
   (*Dop)(psp,psw);
   (*Dop)(psw,psap);

   ai=rsq/norm_square(vol,3,psw);
   update_g(vol);

   rsq_old=rsq;
   rsq=norm_square(vol,3,psr);
   bi=rsq/rsq_old;
   update_xp(vol);
}


static void cg_reset(int vol,void (*Dop)(spinor *s,spinor *r),
                     void (*Dop_dble)(spinor_dble *s,spinor_dble *r))
{
   float r;
   complex z;

   (*Dop_dble)(pdx,pdw);
   (*Dop_dble)(pdw,pdv);

   diff_sd2s(vol,2,pdb,pdv,psr);
   rsq=norm_square(vol,3,psr);

   assign_s2s(vol,2,psp,psw);
   assign_s2s(vol,2,psr,psp);

   z=spinor_prod(vol,3,psr,psw);
   z.re=-z.re/rsq;
   z.im=-z.im/rsq;
   mulc_spinor_add(vol,2,psw,psr,z);

   (*Dop)(psw,psx);
   (*Dop)(psx,psap);

   r=norm_square(vol,3,psx);
   z=spinor_prod(vol,3,psap,psr);

   if ((z.re*z.re+z.im*z.im)<(2.0f*r*r))
   {
      z.re=-z.re/r;
      z.im=-z.im/r;
      mulc_spinor_add(vol,2,psp,psw,z);
   }

   set_s2zero(vol,2,psx);
}


double cgne(int vol,void (*Dop)(spinor *s,spinor *r),
            void (*Dop_dble)(spinor_dble *s,spinor_dble *r),
            spinor **ws,spinor_dble **wsd,int nmx,int istop,double res,
            spinor_dble *eta,spinor_dble *psi,int *status)
{
   int ncg,iprms[3];
   double xn,rn,tol,dprms[1];

   error_loc((vol<=0)||(nmx<1)||(res<=DBL_EPSILON)||(istop<0)||(istop>1),1,
             "cgne [cgne.c]","Parameters are out of range");

   if (NPROC>1)
   {
      iprms[0]=vol;
      iprms[1]=nmx;
      iprms[2]=istop;
      dprms[0]=res;

      MPI_Bcast(iprms,3,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((iprms[0]!=vol)||(iprms[1]!=nmx)||(iprms[2]!=istop)||
            (dprms[0]!=res),1,"cgne [cgne.c]","Parameters are not global");
   }

   cg_init(vol,ws,wsd,eta,psi);
   xn=0.0;

   if (istop)
      rn=(double)(unorm(vol,3,psr));
   else
      rn=sqrt((double)(rsq));

   tol=res*rn;
   status[0]=0;

#ifdef CGNE_DBG
   message("\n");
   message("[cgne]: New call, res = %.1e, tol = %.1e, ||eta|| = %.1e\n",
           res,tol,rn);
#endif

   while (rn>tol)
   {
      ncg=0;

      while (1)
      {
         cg_step(vol,Dop);
         ncg+=1;
         status[0]+=1;

         if (istop)
         {
            xn=(double)(unorm(vol,3,psx));
            rn=(double)(unorm(vol,3,psr));
         }
         else
         {
            xn=(double)(norm_square(vol,3,psx));
            xn=sqrt(xn);
            rn=sqrt((double)(rsq));
         }

#ifdef CGNE_DBG
         message("[cgne]: ncg = %d, status = %d, rn = %.1e\n",
                 ncg,status[0],rn);
#endif

         if ((rn<=tol)||(rn<=(PRECISION_LIMIT*xn))||(ncg>=100)||
             (status[0]>=nmx))
            break;
      }

      add_s2sd(vol,2,psx,pdx);
      cg_reset(vol,Dop,Dop_dble);

      if (istop)
         rn=(double)(unorm(vol,3,psr));
      else
         rn=sqrt((double)(rsq));

#ifdef CGNE_DBG
      if (istop)
         xn=unorm_dble(vol,3,pdx);
      else
      {
         assign_sd2s(vol,2,pdx,psx);
         xn=(double)(norm_square(vol,3,psx));
         xn=sqrt(xn);
         set_s2zero(vol,2,psx);
         rn=sqrt((double)(rsq));
      }

      message("[cgne]: status = %d, ||psi|| = %.1e, ||rho|| = %.1e\n",
              status[0],xn,rn);
#endif

      if ((status[0]>=nmx)&&(rn>tol))
      {
         status[0]=-1;
         return rn;
      }
   }

   return rn;
}
