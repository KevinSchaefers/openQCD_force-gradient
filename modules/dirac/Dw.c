
/*******************************************************************************
*
* File Dw.c
*
* Copyright (C) 2005, 2011-2013, 2016, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Application of the O(a)-improved Wilson-Dirac operator D (single-
* precision programs).
*
*   void Dw(float mu,spinor *s,spinor *r)
*     Depending on whether the twisted-mass flag is set or not, this
*     program applies D+i*mu*gamma_5*1e or D+i*mu*gamma_5 to the field
*     s and assigns the result to the field r.
*
*   void Dwee(float mu,spinor *s,spinor *r)
*     Applies D_ee+i*mu*gamma_5 to the field s on the even points of the
*     lattice and assigns the result to the field r.
*
*   void Dwoo(float mu,spinor *s,spinor *r)
*     Depending on whether the twisted-mass flag is set or not, this
*     program applies D_oo or D_oo+i*mu*gamma_5 to the field s on the
*     odd points of the lattice and assigns the result to the field r.
*
*   void Dwoe(spinor *s,spinor *r)
*     Applies D_oe to the field s and assigns the result to the field r.
*
*   void Dweo(spinor *s,spinor *r)
*     Applies D_eo to the field s and *subtracts* the result from the
*     field r.
*
*   void Dwhat(float mu,spinor *s,spinor *r)
*     Applies Dhat+i*mu*gamma_5 to the field s and assigns the result to
*     the field r.
*
* The following programs operate on the fields in the n'th block b of the
* specified block grid:
*
*   void Dw_blk(blk_grid_t grid,int n,float mu,int k,int l)
*     Depending on whether the twisted-mass flag is set or not, this
*     program applies D+i*mu*gamma_5*1e or D+i*mu*gamma_5 to the field
*     b.s[k] and assigns the result to the field b.s[l].
*
*   void Dwee_blk(blk_grid_t grid,int n,float mu,int k,int l)
*     Applies D_ee+i*mu*gamma_5 to the field b.s[k] on the even points and
*     assigns the result to the field b.s[l].
*
*   void Dwoo_blk(blk_grid_t grid,int n,float mu,int k,int l)
*     Depending on whether the twisted-mass flag is set or not, this
*     program applies D_oo or D_oo+i*mu*gamma_5 to the field b.s[k] on
*     the odd points and assigns the result to the field b.s[l].
*
*   void Dwoe_blk(blk_grid_t grid,int n,int k,int l)
*     Applies D_oe to the field b.s[k] and assigns the result to the field
*     b.s[l].
*
*   void Dweo_blk(blk_grid_t grid,int n,int k,int l)
*     Applies D_eo to the field b.s[k] and *subtracts* the result from the
*     field b.s[l].
*
*   void Dwhat_blk(blk_grid_t grid,int n,float mu,int k,int l)
*     Applies Dhat+i*mu*gamma_5 to the field b.s[k] and assigns the result
*     to the field b.s[l].
*
* The notation and normalization conventions are specified in the notes
* "Implementation of the lattice Dirac operator" (file doc/dirac.pdf).
*
* In all these programs, it is assumed that the SW term is in the proper
* condition and that the global spinor fields have NSPIN elements. The
* programs check whether the twisted-mass flag (see flags/lat_parms.c) is
* set and turn off the twisted-mass term on the odd lattice sites if it is.
* The input and output fields may not coincide in the case of the programs
* Dw(), Dwhat(), Dw_blk() and Dwhat_blk().
*
* When the input and output fields are different, the input field is not
* changed except possibly at the points at global time 0 and NPROC0*L0-1,
* where both fields are set to zero if so required by the chosen boundary
* conditions. Depending on the operator considered, the fields are zeroed
* only on the even or odd points at these times.
*
* The programs Dw(),..,Dwhat() are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
* The programs Dw_blk(),..,Dwhat_blk() are thread-safe, but assume the grid
* "grid" is allocated and that the other parameters are not out of range.
*
* If SSE (AVX) instructions are used, the Dirac spinors must be aligned to
* a 16 (32) byte boundary.
*
*******************************************************************************/

#define DW_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "lattice.h"
#include "uflds.h"
#include "sflds.h"
#include "sw_term.h"
#include "block.h"
#include "dirac.h"
#include "global.h"

#define N0 (NPROC0*L0)

typedef union
{
   spinor s;
   weyl w[2];
} spin_t;

static const spinor s0={{{0.0f,0.0f}}};

#if (defined AVX)
#include "avx.h"

#define _load_cst(c) \
__asm__ __volatile__ ("vbroadcastss %0, %%ymm15 \n\t" \
                      : \
                      : \
                      "m" (c) \
                      : \
                      "xmm15")

#define _mul_cst() \
__asm__ __volatile__ ("vmulps %%ymm15, %%ymm0, %%ymm0 \n\t" \
                      "vmulps %%ymm15, %%ymm1, %%ymm1 \n\t" \
                      "vmulps %%ymm15, %%ymm2, %%ymm2" \
                      : \
                      : \
                      : \
                      "xmm0", "xmm1", "xmm2")


static void doe(float coe,int *piup,int *pidn,su3 *u,spinor *pk,spin_t *rs)
{
   spinor *sp,*sm;

/******************************** direction 0 *********************************/

   sp=pk+piup[0];
   sm=pk+pidn[0];

   _avx_spinor_pair_load34(*sp,*sm);

   sp=pk+piup[1];
   sm=pk+pidn[1];
   _prefetch_spinor(sp);
   _prefetch_spinor(sm);

   _avx_spinor_mul_up(_avx_sgn_add);
   _avx_spinor_add();

   _avx_su3_pair_mixed_multiply(u[0],u[1]);

   _avx_spinor_split();
   _avx_spinor_unsplit();
   _avx_spinor_store_up((*rs).s);

/******************************** direction 1 *********************************/

   _avx_spinor_pair_load43(*sp,*sm);

   sp=pk+piup[2];
   sm=pk+pidn[2];
   _prefetch_spinor(sp);
   _prefetch_spinor(sm);

   _avx_spinor_imul_up(_avx_sgn_i_add);
   _avx_spinor_add();

   _avx_su3_pair_mixed_multiply(u[2],u[3]);

   _avx_spinor_split();
   _avx_spinor_load((*rs).s);
   _avx_weyl_xch_imul(_sse_sgn24);
   _avx_spinor_unsplit();
   _avx_spinor_add();
   _avx_spinor_store((*rs).s);

/******************************** direction 2 *********************************/

   _avx_spinor_pair_load43(*sp,*sm);

   sp=pk+piup[3];
   sm=pk+pidn[3];
   _prefetch_spinor(sp);
   _prefetch_spinor(sm);

   _avx_spinor_mul_up(_avx_sgn_addsub);
   _avx_spinor_add();

   _avx_su3_pair_mixed_multiply(u[4],u[5]);

   _avx_spinor_split();
   _avx_spinor_load((*rs).s);
   _avx_weyl_xch();
   _avx_weyl_mul(_sse_sgn12);
   _avx_spinor_unsplit();
   _avx_spinor_add();
   _avx_spinor_store((*rs).s);

/******************************** direction 3 *********************************/

   _avx_spinor_pair_load34(*sp,*sm);
   _avx_spinor_imul_up(_avx_sgn_i_addsub);
   _avx_spinor_add();

   _avx_su3_pair_mixed_multiply(u[6],u[7]);

   _avx_spinor_split();
   _avx_spinor_load((*rs).s);
   _avx_weyl_imul(_sse_sgn23);
   _avx_spinor_unsplit();
   _load_cst(coe);
   _avx_spinor_add();
   _mul_cst();
   _avx_weyl_pair_store12((*rs).w[0],(*rs).w[1]);

   _avx_zeroupper();
}


static void deo(float ceo,int *piup,int *pidn,su3 *u,spin_t *rs,spinor *pl)
{
   spinor *sp,*sm;

   _load_cst(ceo);
   _avx_spinor_load((*rs).s);
   _mul_cst();
   _avx_spinor_store((*rs).s);

/******************************** direction 0 *********************************/

   sm=pl+pidn[0];
   sp=pl+piup[0];

   _prefetch_spinor(sm);
   _prefetch_spinor(sp);

   _avx_spinor_load_dup((*rs).s);
   _avx_spinor_mul_up(_avx_sgn_add);
   _avx_spinor_add();

   _avx_su3_pair_mixed_multiply(u[1],u[0]);

   _avx_weyl_pair_load12(*sm,*sp);
   _avx_spinor_add();
   _avx_weyl_pair_store12(*sm,*sp);

   _avx_weyl_pair_load34(*sm,*sp);
   _avx_spinor_mul_up(_avx_sgn_add);
   _avx_spinor_add();
   _avx_weyl_pair_store34(*sm,*sp);

/******************************** direction 1 *********************************/

   sm=pl+pidn[1];
   sp=pl+piup[1];

   _prefetch_spinor(sm);
   _prefetch_spinor(sp);

   _avx_spinor_load_dup((*rs).s);
   _avx_spinor_xch_imul_up(_avx_sgn_i_add);
   _avx_spinor_add();

   _avx_su3_pair_mixed_multiply(u[3],u[2]);

   _avx_weyl_pair_load12(*sm,*sp);
   _avx_spinor_add();
   _avx_weyl_pair_store12(*sm,*sp);

   _avx_weyl_pair_load34(*sm,*sp);
   _avx_spinor_xch_imul_up(_avx_sgn_i_add);
   _avx_spinor_sub();
   _avx_weyl_pair_store34(*sm,*sp);

/******************************** direction 2 *********************************/

   sm=pl+pidn[2];
   sp=pl+piup[2];

   _prefetch_spinor(sm);
   _prefetch_spinor(sp);

   _avx_spinor_load_dup((*rs).s);
   _avx_spinor_xch_up();
   _avx_spinor_mul_up(_avx_sgn_addsub);
   _avx_spinor_add();

   _avx_su3_pair_mixed_multiply(u[5],u[4]);

   _avx_weyl_pair_load12(*sm,*sp);
   _avx_spinor_add();
   _avx_weyl_pair_store12(*sm,*sp);

   _avx_weyl_pair_load34(*sm,*sp);
   _avx_spinor_xch_up();
   _avx_spinor_mul_up(_avx_sgn_addsub);
   _avx_spinor_sub();
   _avx_weyl_pair_store34(*sm,*sp);

/******************************** direction 3 *********************************/

   sm=pl+pidn[3];
   sp=pl+piup[3];

   _prefetch_spinor(sm);
   _prefetch_spinor(sp);

   _avx_spinor_load_dup((*rs).s);
   _avx_spinor_imul_up(_avx_sgn_i_addsub);
   _avx_spinor_add();

   _avx_su3_pair_mixed_multiply(u[7],u[6]);

   _avx_weyl_pair_load12(*sm,*sp);
   _avx_spinor_add();
   _avx_weyl_pair_store12(*sm,*sp);

   _avx_weyl_pair_load34(*sm,*sp);
   _avx_spinor_imul_up(_avx_sgn_i_addsub);
   _avx_spinor_sub();
   _avx_weyl_pair_store34(*sm,*sp);

   _avx_zeroupper();
}

#elif (defined x64)
#include "sse2.h"

#define _load_cst(c) \
__asm__ __volatile__ ("movss %0, %%xmm15 \n\t" \
                      "shufps $0x0, %%xmm15, %%xmm15" \
                      : \
                      : \
                      "m" (c) \
                      : \
                      "xmm15")

#define _mul_cst() \
__asm__ __volatile__ ("mulps %%xmm15, %%xmm0 \n\t" \
                      "mulps %%xmm15, %%xmm1 \n\t" \
                      "mulps %%xmm15, %%xmm2" \
                      : \
                      : \
                      : \
                      "xmm0", "xmm1", "xmm2")

#define _mul_cst_up() \
__asm__ __volatile__ ("mulps %%xmm15, %%xmm3 \n\t" \
                      "mulps %%xmm15, %%xmm4 \n\t" \
                      "mulps %%xmm15, %%xmm5" \
                      : \
                      : \
                      : \
                      "xmm3", "xmm4", "xmm5")


static void doe(float coe,int *piup,int *pidn,su3 *u,spinor *pk,spin_t *rs)
{
   spinor *sp,*sm;

/******************************* direction +0 *********************************/

   sp=pk+(*(piup++));

   _sse_pair_load((*sp).c1,(*sp).c2);
   _sse_pair_load_up((*sp).c3,(*sp).c4);

   sm=pk+(*(pidn++));
   _prefetch_spinor(sm);

   _sse_vector_add();
   sp=pk+(*(piup++));
   _prefetch_spinor(sp);
   _sse_su3_multiply(*u);

   _sse_weyl_store_up((*rs).w[0]);
   _sse_weyl_store_up((*rs).w[1]);

/******************************* direction -0 *********************************/

   _sse_pair_load((*sm).c1,(*sm).c2);
   _sse_pair_load_up((*sm).c3,(*sm).c4);

   u+=2;
   _prefetch_su3_dble(u);
   u-=1;
   _sse_vector_sub();
   sm=pk+(*(pidn++));
   _prefetch_spinor(sm);
   _sse_su3_inverse_multiply(*u);

   _sse_weyl_load((*rs).w[0]);
   _sse_vector_add();
   _sse_weyl_store((*rs).w[0]);

   _sse_weyl_load((*rs).w[1]);
   _sse_vector_sub();
   _sse_weyl_store((*rs).w[1]);

/******************************* direction +1 *********************************/

   _sse_pair_load((*sp).c1,(*sp).c2);
   _sse_pair_load_up((*sp).c4,(*sp).c3);

   _sse_vector_i_add();
   sp=pk+(*(piup++));
   _prefetch_spinor(sp);
   u+=1;
   _sse_su3_multiply(*u);

   _sse_weyl_load((*rs).w[0]);
   _sse_vector_add();
   _sse_weyl_store((*rs).w[0]);

   _sse_weyl_load((*rs).w[1]);
   _sse_vector_xch_i_sub();
   _sse_weyl_store((*rs).w[1]);

/******************************* direction -1 *********************************/

   _sse_pair_load((*sm).c1,(*sm).c2);
   _sse_pair_load_up((*sm).c4,(*sm).c3);

   u+=2;
   _prefetch_su3_dble(u);
   u-=1;
   _sse_vector_i_sub();
   sm=pk+(*(pidn++));
   _prefetch_spinor(sm);
   _sse_su3_inverse_multiply(*u);

   _sse_weyl_load((*rs).w[0]);
   _sse_vector_add();
   _sse_weyl_store((*rs).w[0]);

   _sse_weyl_load((*rs).w[1]);
   _sse_vector_xch_i_add();
   _sse_weyl_store((*rs).w[1]);

/******************************* direction +2 *********************************/

   _sse_pair_load((*sp).c1,(*sp).c2);
   _sse_pair_load_up((*sp).c4,(*sp).c3);

   _sse_vector_addsub();

   u+=1;
   _sse_su3_multiply(*u);
   sp=pk+(*(piup));
   _prefetch_spinor(sp);
   _sse_weyl_load((*rs).w[0]);
   _sse_vector_add();
   _sse_weyl_store((*rs).w[0]);

   _sse_weyl_load((*rs).w[1]);
   _sse_vector_xch();
   _sse_vector_subadd();
   _sse_weyl_store((*rs).w[1]);

/******************************* direction -2 *********************************/

   _sse_pair_load((*sm).c1,(*sm).c2);
   _sse_pair_load_up((*sm).c4,(*sm).c3);

   u+=2;
   _prefetch_su3_dble(u);
   u-=1;
   _sse_vector_subadd();
   sm=pk+(*(pidn));
   _prefetch_spinor(sm);
   _sse_su3_inverse_multiply(*u);

   _sse_weyl_load((*rs).w[0]);
   _sse_vector_add();
   _sse_weyl_store((*rs).w[0]);

   _sse_weyl_load((*rs).w[1]);
   _sse_vector_xch();
   _sse_vector_addsub();
   _sse_weyl_store((*rs).w[1]);

/******************************* direction +3 *********************************/

   _sse_pair_load((*sp).c1,(*sp).c2);
   _sse_pair_load_up((*sp).c3,(*sp).c4);

   _sse_vector_i_addsub();
   u+=1;
   _sse_su3_multiply(*u);

   _sse_weyl_load((*rs).w[0]);
   _sse_vector_add();
   _sse_weyl_store((*rs).w[0]);

   _sse_weyl_load((*rs).w[1]);
   _sse_vector_i_subadd();
   _sse_weyl_store((*rs).w[1]);

/******************************* direction -3 *********************************/

   _sse_pair_load((*sm).c1,(*sm).c2);
   _sse_pair_load_up((*sm).c3,(*sm).c4);

   u+=2;
   _prefetch_su3_dble(u);
   u-=1;
   _sse_vector_i_subadd();
   _sse_su3_inverse_multiply(*u);

   _load_cst(coe);
   _sse_weyl_load((*rs).w[0]);
   _sse_vector_add();
   _mul_cst();
   _sse_pair_store((*rs).s.c1,(*rs).s.c2);

   _sse_weyl_load((*rs).w[1]);
   _sse_vector_i_addsub();
   _mul_cst();
   _sse_pair_store((*rs).s.c3,(*rs).s.c4);
}


static void deo(float ceo,int *piup,int *pidn,su3 *u,spin_t *rs,spinor *pl)
{
   spinor *sp,*sm;

/******************************* direction +0 *********************************/

   sp=pl+(*(piup++));
   _prefetch_spinor(sp);

   _load_cst(ceo);
   _sse_pair_load((*rs).s.c1,(*rs).s.c2);
   _sse_pair_load_up((*rs).s.c3,(*rs).s.c4);
   _mul_cst();
   _mul_cst_up();
   _sse_weyl_store((*rs).w[0]);
   _sse_weyl_store_up((*rs).w[1]);

   sm=pl+(*(pidn++));
   _prefetch_spinor(sm);
   _sse_vector_sub();
   _sse_su3_inverse_multiply(*u);

   _sse_pair_load((*sp).c1,(*sp).c2);
   _sse_vector_add();
   _sse_pair_store((*sp).c1,(*sp).c2);

   _sse_pair_load((*sp).c3,(*sp).c4);
   _sse_vector_sub();
   _sse_pair_store((*sp).c3,(*sp).c4);

/******************************* direction -0 *********************************/

   _sse_weyl_load((*rs).w[0]);
   _sse_weyl_load_up((*rs).w[1]);

   sp=pl+(*(piup++));
   _prefetch_spinor(sp);
   _sse_vector_add();
   u+=1;
   _sse_su3_multiply(*u);

   _sse_pair_load((*sm).c1,(*sm).c2);
   _sse_vector_add();
   _sse_pair_store((*sm).c1,(*sm).c2);

   _sse_pair_load((*sm).c3,(*sm).c4);
   _sse_vector_add();
   _sse_pair_store((*sm).c3,(*sm).c4);

/******************************* direction +1 *********************************/

   _sse_weyl_load((*rs).w[0]);
   _sse_weyl_load_up((*rs).w[1]);

   sm=pl+(*(pidn++));
   _prefetch_spinor(sm);
   _sse_vector_xch_i_sub();
   u+=1;
   _sse_su3_inverse_multiply(*u);

   _sse_pair_load((*sp).c1,(*sp).c2);
   _sse_vector_add();
   _sse_pair_store((*sp).c1,(*sp).c2);

   _sse_pair_load((*sp).c3,(*sp).c4);
   _sse_vector_xch_i_add();
   _sse_pair_store((*sp).c3,(*sp).c4);

/******************************* direction -1 *********************************/

   _sse_weyl_load((*rs).w[0]);
   _sse_weyl_load_up((*rs).w[1]);

   sp=pl+(*(piup++));
   _prefetch_spinor(sp);
   _sse_vector_xch_i_add();
   u+=1;
   _sse_su3_multiply(*u);

   _sse_pair_load((*sm).c1,(*sm).c2);
   _sse_vector_add();
   _sse_pair_store((*sm).c1,(*sm).c2);

   _sse_pair_load((*sm).c3,(*sm).c4);
   _sse_vector_xch_i_sub();
   _sse_pair_store((*sm).c3,(*sm).c4);

/******************************* direction +2 *********************************/

   _sse_weyl_load((*rs).w[0]);
   _sse_weyl_load_up((*rs).w[1]);

   sm=pl+(*(pidn++));
   _prefetch_spinor(sm);
   _sse_vector_xch();
   _sse_vector_subadd();
   u+=1;
   _sse_su3_inverse_multiply(*u);

   _sse_pair_load((*sp).c1,(*sp).c2);
   _sse_vector_add();
   _sse_pair_store((*sp).c1,(*sp).c2);

   _sse_pair_load((*sp).c3,(*sp).c4);
   _sse_vector_xch();
   _sse_vector_addsub();
   _sse_pair_store((*sp).c3,(*sp).c4);

/******************************* direction -2 *********************************/

   _sse_weyl_load((*rs).w[0]);
   _sse_weyl_load_up((*rs).w[1]);

   sp=pl+(*(piup));
   _prefetch_spinor(sp);
   _sse_vector_xch();
   _sse_vector_addsub();
   u+=1;
   _sse_su3_multiply(*u);

   _sse_pair_load((*sm).c1,(*sm).c2);
   _sse_vector_add();
   _sse_pair_store((*sm).c1,(*sm).c2);

   _sse_pair_load((*sm).c3,(*sm).c4);
   _sse_vector_xch();
   _sse_vector_subadd();
   _sse_pair_store((*sm).c3,(*sm).c4);

/******************************* direction +3 *********************************/

   _sse_weyl_load((*rs).w[0]);
   _sse_weyl_load_up((*rs).w[1]);

   sm=pl+(*(pidn));
   _prefetch_spinor(sm);
   _sse_vector_i_subadd();
   u+=1;
   _sse_su3_inverse_multiply(*u);

   _sse_pair_load((*sp).c1,(*sp).c2);
   _sse_vector_add();
   _sse_pair_store((*sp).c1,(*sp).c2);

   _sse_pair_load((*sp).c3,(*sp).c4);
   _sse_vector_i_addsub();
   _sse_pair_store((*sp).c3,(*sp).c4);

/******************************* direction -3 *********************************/

   _sse_weyl_load((*rs).w[0]);
   _sse_weyl_load_up((*rs).w[1]);

   _sse_vector_i_addsub();
   u+=1;
   _sse_su3_multiply(*u);

   _sse_pair_load((*sm).c1,(*sm).c2);
   _sse_vector_add();
   _sse_pair_store((*sm).c1,(*sm).c2);

   _sse_pair_load((*sm).c3,(*sm).c4);
   _sse_vector_i_subadd();
   _sse_pair_store((*sm).c3,(*sm).c4);
}

#else

#define _vector_mul_assign(r,c) \
   (r).c1.re*=(c); \
   (r).c1.im*=(c); \
   (r).c2.re*=(c); \
   (r).c2.im*=(c); \
   (r).c3.re*=(c); \
   (r).c3.im*=(c)


static void doe(float coe,int *piup,int *pidn,su3 *u,spinor *pk,spin_t *rs)
{
   spinor *sp,*sm;
   su3_vector psi,chi;

/******************************* direction +0 *********************************/

   sp=pk+(*(piup++));

   _vector_add(psi,(*sp).c1,(*sp).c3);
   _su3_multiply((*rs).s.c1,*u,psi);
   (*rs).s.c3=(*rs).s.c1;

   _vector_add(psi,(*sp).c2,(*sp).c4);
   _su3_multiply((*rs).s.c2,*u,psi);
   (*rs).s.c4=(*rs).s.c2;

/******************************* direction -0 *********************************/

   sm=pk+(*(pidn++));
   u+=1;

   _vector_sub(psi,(*sm).c1,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c1,chi);
   _vector_sub_assign((*rs).s.c3,chi);

   _vector_sub(psi,(*sm).c2,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c2,chi);
   _vector_sub_assign((*rs).s.c4,chi);

/******************************* direction +1 *********************************/

   sp=pk+(*(piup++));
   u+=1;

   _vector_i_add(psi,(*sp).c1,(*sp).c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c1,chi);
   _vector_i_sub_assign((*rs).s.c4,chi);

   _vector_i_add(psi,(*sp).c2,(*sp).c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c2,chi);
   _vector_i_sub_assign((*rs).s.c3,chi);

/******************************* direction -1 *********************************/

   sm=pk+(*(pidn++));
   u+=1;

   _vector_i_sub(psi,(*sm).c1,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c1,chi);
   _vector_i_add_assign((*rs).s.c4,chi);

   _vector_i_sub(psi,(*sm).c2,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c2,chi);
   _vector_i_add_assign((*rs).s.c3,chi);

/******************************* direction +2 *********************************/

   sp=pk+(*(piup++));
   u+=1;

   _vector_add(psi,(*sp).c1,(*sp).c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c1,chi);
   _vector_add_assign((*rs).s.c4,chi);

   _vector_sub(psi,(*sp).c2,(*sp).c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c2,chi);
   _vector_sub_assign((*rs).s.c3,chi);

/******************************* direction -2 *********************************/

   sm=pk+(*(pidn++));
   u+=1;

   _vector_sub(psi,(*sm).c1,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c1,chi);
   _vector_sub_assign((*rs).s.c4,chi);

   _vector_add(psi,(*sm).c2,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c2,chi);
   _vector_add_assign((*rs).s.c3,chi);

/******************************* direction +3 *********************************/

   sp=pk+(*(piup));
   u+=1;

   _vector_i_add(psi,(*sp).c1,(*sp).c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c1,chi);
   _vector_i_sub_assign((*rs).s.c3,chi);

   _vector_i_sub(psi,(*sp).c2,(*sp).c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c2,chi);
   _vector_i_add_assign((*rs).s.c4,chi);

/******************************* direction -3 *********************************/

   sm=pk+(*(pidn));
   u+=1;

   _vector_i_sub(psi,(*sm).c1,(*sm).c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c1,chi);
   _vector_i_add_assign((*rs).s.c3,chi);

   _vector_i_add(psi,(*sm).c2,(*sm).c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*rs).s.c2,chi);
   _vector_i_sub_assign((*rs).s.c4,chi);

   _vector_mul_assign((*rs).s.c1,coe);
   _vector_mul_assign((*rs).s.c2,coe);
   _vector_mul_assign((*rs).s.c3,coe);
   _vector_mul_assign((*rs).s.c4,coe);
}


static void deo(float ceo,int *piup,int *pidn,su3 *u,spin_t *rs,spinor *pl)
{
   spinor *sp,*sm;
   su3_vector psi,chi;

   _vector_mul_assign((*rs).s.c1,ceo);
   _vector_mul_assign((*rs).s.c2,ceo);
   _vector_mul_assign((*rs).s.c3,ceo);
   _vector_mul_assign((*rs).s.c4,ceo);

/******************************* direction +0 *********************************/

   sp=pl+(*(piup++));

   _vector_sub(psi,(*rs).s.c1,(*rs).s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_sub_assign((*sp).c3,chi);

   _vector_sub(psi,(*rs).s.c2,(*rs).s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_sub_assign((*sp).c4,chi);

/******************************* direction -0 *********************************/

   sm=pl+(*(pidn++));
   u+=1;

   _vector_add(psi,(*rs).s.c1,(*rs).s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_add_assign((*sm).c3,chi);

   _vector_add(psi,(*rs).s.c2,(*rs).s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_add_assign((*sm).c4,chi);

/******************************* direction +1 *********************************/

   sp=pl+(*(piup++));
   u+=1;

   _vector_i_sub(psi,(*rs).s.c1,(*rs).s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_i_add_assign((*sp).c4,chi);

   _vector_i_sub(psi,(*rs).s.c2,(*rs).s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_i_add_assign((*sp).c3,chi);

/******************************* direction -1 *********************************/

   sm=pl+(*(pidn++));
   u+=1;

   _vector_i_add(psi,(*rs).s.c1,(*rs).s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_i_sub_assign((*sm).c4,chi);

   _vector_i_add(psi,(*rs).s.c2,(*rs).s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_i_sub_assign((*sm).c3,chi);

/******************************* direction +2 *********************************/

   sp=pl+(*(piup++));
   u+=1;

   _vector_sub(psi,(*rs).s.c1,(*rs).s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_sub_assign((*sp).c4,chi);

   _vector_add(psi,(*rs).s.c2,(*rs).s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_add_assign((*sp).c3,chi);

/******************************* direction -2 *********************************/

   sm=pl+(*(pidn++));
   u+=1;

   _vector_add(psi,(*rs).s.c1,(*rs).s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_add_assign((*sm).c4,chi);

   _vector_sub(psi,(*rs).s.c2,(*rs).s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_sub_assign((*sm).c3,chi);

/******************************* direction +3 *********************************/

   sp=pl+(*(piup));
   u+=1;

   _vector_i_sub(psi,(*rs).s.c1,(*rs).s.c3);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c1,chi);
   _vector_i_add_assign((*sp).c3,chi);

   _vector_i_add(psi,(*rs).s.c2,(*rs).s.c4);
   _su3_inverse_multiply(chi,*u,psi);
   _vector_add_assign((*sp).c2,chi);
   _vector_i_sub_assign((*sp).c4,chi);

/******************************* direction -3 *********************************/

   sm=pl+(*(pidn));
   u+=1;

   _vector_i_add(psi,(*rs).s.c1,(*rs).s.c3);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c1,chi);
   _vector_i_sub_assign((*sm).c3,chi);

   _vector_i_sub(psi,(*rs).s.c2,(*rs).s.c4);
   _su3_multiply(chi,*u,psi);
   _vector_add_assign((*sm).c2,chi);
   _vector_i_add_assign((*sm).c4,chi);
}

#endif

void Dw(float mu,spinor *s,spinor *r)
{
   int bc;
   int k,ix,t,ofs,vol,isb;
   int *piup,*pidn;
   float muo;
   su3 *ub,*u;
   pauli *swb,*sw;
   spin_t *rs,*so,*ro;
   tm_parms_t tm;

   cps_int_bnd(0x1,s);

   bc=bc_type();
   tm=tm_parms();
   if (tm.eoflg==1)
      muo=0.0f;
   else
      muo=mu;

   ub=ufld();
   swb=swfld();

#pragma omp parallel private(k,ix,t,vol,ofs,isb,piup,pidn,u,sw,rs,so,ro)
   {
      k=omp_get_thread_num();

      vol=(BNDRY/2)/NTHREAD;
      ofs=k*vol;

      if (k==(NTHREAD-1))
         vol=(BNDRY/2)-ofs;

      set_s2zero(vol,0,r+VOLUME+ofs);

      vol=VOLUME_TRD/2;
      ofs=k*vol;
      apply_sw(vol,mu,swb+2*ofs,s+ofs,r+ofs);

      u=ub+8*ofs;
      ofs+=VOLUME/2;
      piup=iup[ofs];
      pidn=idn[ofs];
      sw=swb+2*ofs;
      so=(spin_t*)(s+ofs);
      ro=(spin_t*)(r+ofs);
      rs=amalloc(sizeof(*rs),5);

      if (((cpr[0]==0)&&(bc!=3))||((cpr[0]==(NPROC0-1))&&(bc==0)))
      {
         for (isb=0;isb<16;isb++)
         {
#pragma omp barrier
            ofs=(VOLUME/2)+k*(VOLUME_TRD/2)+sbofs[isb]/2;
            vol=sbvol[isb]/2;

            for (ix=ofs;ix<(ofs+vol);ix++)
            {
               t=global_time(ix);

               if ((t>0)&&((t<(N0-1))||(bc!=0)))
               {
                  doe(-0.5f,piup,pidn,u,s,rs);

                  mul_pauli2(muo,sw,&((*so).s),&((*ro).s));

                  _vector_add_assign((*ro).s.c1,(*rs).s.c1);
                  _vector_add_assign((*ro).s.c2,(*rs).s.c2);
                  _vector_add_assign((*ro).s.c3,(*rs).s.c3);
                  _vector_add_assign((*ro).s.c4,(*rs).s.c4);
                  (*rs)=(*so);

                  deo(-0.5f,piup,pidn,u,rs,r);
               }
               else
               {
                  (*so).s=s0;
                  (*ro).s=s0;
               }

               piup+=4;
               pidn+=4;
               u+=8;
               sw+=2;
               so+=1;
               ro+=1;
            }
         }
      }
      else
      {
         for (isb=0;isb<16;isb++)
         {
#pragma omp barrier
            ofs=(VOLUME/2)+k*(VOLUME_TRD/2)+sbofs[isb]/2;
            vol=sbvol[isb]/2;

            for (ix=ofs;ix<(ofs+vol);ix++)
            {
               doe(-0.5f,piup,pidn,u,s,rs);

               mul_pauli2(muo,sw,&((*so).s),&((*ro).s));

               _vector_add_assign((*ro).s.c1,(*rs).s.c1);
               _vector_add_assign((*ro).s.c2,(*rs).s.c2);
               _vector_add_assign((*ro).s.c3,(*rs).s.c3);
               _vector_add_assign((*ro).s.c4,(*rs).s.c4);
               (*rs)=(*so);

               deo(-0.5f,piup,pidn,u,rs,r);

               piup+=4;
               pidn+=4;
               u+=8;
               sw+=2;
               so+=1;
               ro+=1;
            }
         }
      }

      afree(rs);
   }

   cps_ext_bnd(0x1,r);
}


void Dwee(float mu,spinor *s,spinor *r)
{
   int bc;
   int k,ix,t,ofs,vol;
   pauli *swb,*sw;
   spin_t *se,*re;

   bc=bc_type();
   swb=swfld();

#pragma omp parallel private(k,ix,t,vol,ofs,sw,se,re)
   {
      k=omp_get_thread_num();

      vol=VOLUME_TRD/2;
      ofs=k*vol;
      sw=swb+2*ofs;
      se=(spin_t*)(s+ofs);
      re=(spin_t*)(r+ofs);

      if (((cpr[0]==0)&&(bc!=3))||((cpr[0]==(NPROC0-1))&&(bc==0)))
      {
         for (ix=ofs;ix<(ofs+vol);ix++)
         {
            t=global_time(ix);

            if ((t>0)&&((t<(N0-1))||(bc!=0)))
               mul_pauli2(mu,sw,&((*se).s),&((*re).s));
            else
            {
               (*se).s=s0;
               (*re).s=s0;
            }

            sw+=2;
            se+=1;
            re+=1;
         }
      }
      else
      {
         for (ix=ofs;ix<(ofs+vol);ix++)
         {
            mul_pauli2(mu,sw,&((*se).s),&((*re).s));

            sw+=2;
            se+=1;
            re+=1;
         }
      }
   }
}


void Dwoo(float mu,spinor *s,spinor *r)
{
   int bc;
   int k,ix,t,ofs,vol;
   pauli *swb,*sw;
   spin_t *so,*ro;
   tm_parms_t tm;

   bc=bc_type();
   tm=tm_parms();
   if (tm.eoflg==1)
      mu=0.0f;
   swb=swfld();

#pragma omp parallel private(k,ix,t,vol,ofs,sw,so,ro)
   {
      k=omp_get_thread_num();

      vol=VOLUME_TRD/2;
      ofs=VOLUME/2+k*vol;
      sw=swb+2*ofs;
      so=(spin_t*)(s+ofs);
      ro=(spin_t*)(r+ofs);

      if (((cpr[0]==0)&&(bc!=3))||((cpr[0]==(NPROC0-1))&&(bc==0)))
      {
         for (ix=ofs;ix<(ofs+vol);ix++)
         {
            t=global_time(ix);

            if ((t>0)&&((t<(N0-1))||(bc!=0)))
               mul_pauli2(mu,sw,&((*so).s),&((*ro).s));
            else
            {
               (*so).s=s0;
               (*ro).s=s0;
            }

            sw+=2;
            so+=1;
            ro+=1;
         }
      }
      else
      {
         for (ix=ofs;ix<(ofs+vol);ix++)
         {
            mul_pauli2(mu,sw,&((*so).s),&((*ro).s));

            sw+=2;
            so+=1;
            ro+=1;
         }
      }
   }
}


void Dwoe(spinor *s,spinor *r)
{
   int bc;
   int k,ix,t,ofs,vol,isb;
   int *piup,*pidn;
   su3 *ub,*u;
   spin_t *rs,*ro;

   cps_int_bnd(0x1,s);

   bc=bc_type();
   ub=ufld();

#pragma omp parallel private(k,ix,t,vol,ofs,isb,piup,pidn,u,rs,ro)
   {
      k=omp_get_thread_num();

      ofs=k*VOLUME_TRD/2;
      u=ub+8*ofs;
      ofs+=VOLUME/2;
      piup=iup[ofs];
      pidn=idn[ofs];
      ro=(spin_t*)(r+ofs);
      rs=amalloc(sizeof(*rs),5);

      if (((cpr[0]==0)&&(bc!=3))||((cpr[0]==(NPROC0-1))&&(bc==0)))
      {
         for (isb=0;isb<16;isb++)
         {
#pragma omp barrier
            ofs=(VOLUME/2)+k*(VOLUME_TRD/2)+sbofs[isb]/2;
            vol=sbvol[isb]/2;

            for (ix=ofs;ix<(ofs+vol);ix++)
            {
               t=global_time(ix);

               if ((t>0)&&((t<(N0-1))||(bc!=0)))
               {
                  doe(-0.5f,piup,pidn,u,s,rs);
                  (*ro)=(*rs);
               }
               else
                  (*ro).s=s0;

               piup+=4;
               pidn+=4;
               u+=8;
               ro+=1;
            }
         }
      }
      else
      {
         for (isb=0;isb<16;isb++)
         {
#pragma omp barrier
            ofs=(VOLUME/2)+k*(VOLUME_TRD/2)+sbofs[isb]/2;
            vol=sbvol[isb]/2;

            for (ix=ofs;ix<(ofs+vol);ix++)
            {
               doe(-0.5f,piup,pidn,u,s,rs);
               (*ro)=(*rs);

               piup+=4;
               pidn+=4;
               u+=8;
               ro+=1;
            }
         }
      }

      afree(rs);
   }
}


void Dweo(spinor *s,spinor *r)
{
   int bc;
   int k,ix,t,ofs,vol,isb;
   int *piup,*pidn;
   su3 *ub,*u;
   spin_t *rs,*so;

   bc=bc_type();
   ub=ufld();

#pragma omp parallel private(k,ix,t,vol,ofs,isb,piup,pidn,u,rs,so)
   {
      k=omp_get_thread_num();

      vol=(BNDRY/2)/NTHREAD;
      ofs=k*vol;

      if (k==(NTHREAD-1))
         vol=(BNDRY/2)-ofs;

      set_s2zero(vol,0,r+VOLUME+ofs);

     vol=VOLUME_TRD/2;
      ofs=k*vol;
      u=ub+8*ofs;
      ofs+=VOLUME/2;
      piup=iup[ofs];
      pidn=idn[ofs];
      so=(spin_t*)(s+ofs);
      rs=amalloc(sizeof(*rs),5);

      if (((cpr[0]==0)&&(bc!=3))||((cpr[0]==(NPROC0-1))&&(bc==0)))
      {
         for (isb=0;isb<16;isb++)
         {
#pragma omp barrier
            ofs=(VOLUME/2)+k*(VOLUME_TRD/2)+sbofs[isb]/2;
            vol=sbvol[isb]/2;

            for (ix=ofs;ix<(ofs+vol);ix++)
            {
               t=global_time(ix);

               if ((t>0)&&((t<(N0-1))||(bc!=0)))
               {
                  (*rs)=(*so);
                  deo(0.5f,piup,pidn,u,rs,r);
               }
               else
                  (*so).s=s0;

               piup+=4;
               pidn+=4;
               u+=8;
               so+=1;
            }
         }
      }
      else
      {
         for (isb=0;isb<16;isb++)
         {
#pragma omp barrier
            ofs=(VOLUME/2)+k*(VOLUME_TRD/2)+sbofs[isb]/2;
            vol=sbvol[isb]/2;

            for (ix=ofs;ix<(ofs+vol);ix++)
            {
               (*rs)=(*so);
               deo(0.5f,piup,pidn,u,rs,r);

               piup+=4;
               pidn+=4;
               u+=8;
               so+=1;
            }
         }
      }

      afree(rs);
   }

   cps_ext_bnd(0x1,r);
}


void Dwhat(float mu,spinor *s,spinor *r)
{
   int bc;
   int k,ix,t,ofs,vol,isb;
   int *piup,*pidn;
   su3 *ub,*u;
   pauli *swb,*sw;
   spin_t *rs;

   cps_int_bnd(0x1,s);

   bc=bc_type();
   ub=ufld();
   swb=swfld();

#pragma omp parallel private(k,ix,t,vol,ofs,isb,piup,pidn,u,sw,rs)
   {
      k=omp_get_thread_num();

      vol=(BNDRY/2)/NTHREAD;
      ofs=k*vol;

      if (k==(NTHREAD-1))
         vol=(BNDRY/2)-ofs;

      set_s2zero(vol,0,r+VOLUME+ofs);

      vol=VOLUME_TRD/2;
      ofs=k*vol;
      apply_sw(vol,mu,swb+2*ofs,s+ofs,r+ofs);

      u=ub+8*ofs;
      ofs+=VOLUME/2;
      piup=iup[ofs];
      pidn=idn[ofs];
      sw=swb+2*ofs;
      rs=amalloc(sizeof(*rs),5);

      if (((cpr[0]==0)&&(bc!=3))||((cpr[0]==(NPROC0-1))&&(bc==0)))
      {
         for (isb=0;isb<16;isb++)
         {
#pragma omp barrier
            ofs=(VOLUME/2)+k*(VOLUME_TRD/2)+sbofs[isb]/2;
            vol=sbvol[isb]/2;

            for (ix=ofs;ix<(ofs+vol);ix++)
            {
               t=global_time(ix);

               if ((t>0)&&((t<(N0-1))||(bc!=0)))
               {
                  doe(-0.5f,piup,pidn,u,s,rs);

                  mul_pauli2(0.0f,sw,&((*rs).s),&((*rs).s));

                  deo(0.5f,piup,pidn,u,rs,r);
               }

               piup+=4;
               pidn+=4;
               u+=8;
               sw+=2;
            }
         }
      }
      else
      {
         for (isb=0;isb<16;isb++)
         {
#pragma omp barrier
            ofs=(VOLUME/2)+k*(VOLUME_TRD/2)+sbofs[isb]/2;
            vol=sbvol[isb]/2;

            for (ix=ofs;ix<(ofs+vol);ix++)
            {
               doe(-0.5f,piup,pidn,u,s,rs);

               mul_pauli2(0.0f,sw,&((*rs).s),&((*rs).s));

               deo(0.5f,piup,pidn,u,rs,r);

               piup+=4;
               pidn+=4;
               u+=8;
               sw+=2;
            }
         }
      }

      afree(rs);
   }

   cps_ext_bnd(0x1,r);
}


void Dw_blk(blk_grid_t grid,int n,float mu,int k,int l)
{
   int nb,isw,vol,volh,ibu,ibd;
   int *piup,*pidn,*ibp,*ibm;
   su3 *u,*um;
   pauli *sw;
   spinor *s,*r;
   spin_t *rs,*so,*ro;
   block_t *b;
   tm_parms_t tm;

   b=blk_list(grid,&nb,&isw)+n;

   vol=(*b).vol;
   volh=vol/2;
   s=(*b).s[k];
   r=(*b).s[l];
   so=(spin_t*)(s+volh);
   ro=(spin_t*)(r+volh);

   s[vol]=s0;
   r[vol]=s0;
   sw=(*b).sw;
   apply_sw(volh,mu,sw,s,r);
   tm=tm_parms();
   if (tm.eoflg==1)
      mu=0.0f;

   piup=(*b).iup[volh];
   pidn=(*b).idn[volh];
   sw+=vol;
   u=(*b).u;
   um=u+4*vol;
   rs=amalloc(sizeof(*rs),5);

   if ((*b).nbp)
   {
      ibp=(*b).ibp;
      ibm=ibp+(*b).nbp/2;

      for (;ibp<ibm;ibp++)
         s[*ibp]=s0;

      ibu=((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0)&&(bc_type()==0));
      ibd=((cpr[0]==0)&&((*b).bo[0]==0));

      for (;u<um;u+=8)
      {
         if (((pidn[0]<vol)||(!ibd))&&((piup[0]<vol)||(!ibu)))
         {
            doe(-0.5f,piup,pidn,u,s,rs);

            mul_pauli2(mu,sw,&((*so).s),&((*ro).s));

            _vector_add_assign((*ro).s.c1,(*rs).s.c1);
            _vector_add_assign((*ro).s.c2,(*rs).s.c2);
            _vector_add_assign((*ro).s.c3,(*rs).s.c3);
            _vector_add_assign((*ro).s.c4,(*rs).s.c4);
            (*rs)=(*so);

            deo(-0.5f,piup,pidn,u,rs,r);
         }
         else
         {
            (*so).s=s0;
            (*ro).s=s0;
         }

         piup+=4;
         pidn+=4;
         sw+=2;
         ro+=1;
         so+=1;
      }

      ibp=(*b).ibp;
      ibm=ibp+(*b).nbp/2;

      for (;ibp<ibm;ibp++)
         r[*ibp]=s0;
   }
   else
   {
      for (;u<um;u+=8)
      {
         doe(-0.5f,piup,pidn,u,s,rs);

         mul_pauli2(mu,sw,&((*so).s),&((*ro).s));

         _vector_add_assign((*ro).s.c1,(*rs).s.c1);
         _vector_add_assign((*ro).s.c2,(*rs).s.c2);
         _vector_add_assign((*ro).s.c3,(*rs).s.c3);
         _vector_add_assign((*ro).s.c4,(*rs).s.c4);
         (*rs)=(*so);

         deo(-0.5f,piup,pidn,u,rs,r);

         piup+=4;
         pidn+=4;
         sw+=2;
         ro+=1;
         so+=1;
      }
   }

   afree(rs);
}


void Dwee_blk(blk_grid_t grid,int n,float mu,int k,int l)
{
   int nb,isw,vol,ibu,ibd;
   int *piup,*pidn;
   pauli *sw,*swm;
   spin_t *se,*re;
   block_t *b;

   b=blk_list(grid,&nb,&isw)+n;

   vol=(*b).vol;
   se=(spin_t*)((*b).s[k]);
   re=(spin_t*)((*b).s[l]);
   sw=(*b).sw;
   swm=sw+vol;

   if ((*b).nbp)
   {
      piup=(*b).iup[0];
      pidn=(*b).idn[0];

      ibu=((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0)&&(bc_type()==0));
      ibd=((cpr[0]==0)&&((*b).bo[0]==0));

      for (;sw<swm;sw+=2)
      {
         if (((pidn[0]<vol)||(!ibd))&&((piup[0]<vol)||(!ibu)))
            mul_pauli2(mu,sw,&((*se).s),&((*re).s));
         else
         {
            (*se).s=s0;
            (*re).s=s0;
         }

         piup+=4;
         pidn+=4;
         se+=1;
         re+=1;
      }
   }
   else
   {
      for (;sw<swm;sw+=2)
      {
         mul_pauli2(mu,sw,&((*se).s),&((*re).s));

         se+=1;
         re+=1;
      }
   }
}


void Dwoo_blk(blk_grid_t grid,int n,float mu,int k,int l)
{
   int nb,isw,vol,volh,ibu,ibd;
   int *piup,*pidn;
   pauli *sw,*swm;
   spin_t *so,*ro;
   block_t *b;
   tm_parms_t tm;

   b=blk_list(grid,&nb,&isw)+n;

   vol=(*b).vol;
   volh=vol/2;
   so=(spin_t*)((*b).s[k]+volh);
   ro=(spin_t*)((*b).s[l]+volh);
   tm=tm_parms();
   if (tm.eoflg==1)
      mu=0.0f;

   sw=(*b).sw+vol;
   swm=sw+vol;

   if ((*b).nbp)
   {
      piup=(*b).iup[volh];
      pidn=(*b).idn[volh];

      ibu=((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0)&&(bc_type()==0));
      ibd=((cpr[0]==0)&&((*b).bo[0]==0));

      for (;sw<swm;sw+=2)
      {
         if (((pidn[0]<vol)||(!ibd))&&((piup[0]<vol)||(!ibu)))
            mul_pauli2(mu,sw,&((*so).s),&((*ro).s));
         else
         {
            (*so).s=s0;
            (*ro).s=s0;
         }

         piup+=4;
         pidn+=4;
         so+=1;
         ro+=1;
      }
   }
   else
   {
      for (;sw<swm;sw+=2)
      {
         mul_pauli2(mu,sw,&((*so).s),&((*ro).s));

         so+=1;
         ro+=1;
      }
   }
}


void Dwoe_blk(blk_grid_t grid,int n,int k,int l)
{
   int nb,isw,vol,volh,ibu,ibd;
   int *piup,*pidn,*ibp,*ibm;
   su3 *u,*um;
   spinor *s;
   spin_t *rs,*ro;
   block_t *b;

   b=blk_list(grid,&nb,&isw)+n;

   vol=(*b).vol;
   volh=vol/2;
   s=(*b).s[k];
   ro=(spin_t*)((*b).s[l]+volh);
   s[vol]=s0;

   piup=(*b).iup[volh];
   pidn=(*b).idn[volh];
   u=(*b).u;
   um=u+4*vol;
   rs=amalloc(sizeof(*rs),5);

   if ((*b).nbp)
   {
      ibp=(*b).ibp;
      ibm=ibp+(*b).nbp/2;

      for (;ibp<ibm;ibp++)
         s[*ibp]=s0;

      ibu=((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0)&&(bc_type()==0));
      ibd=((cpr[0]==0)&&((*b).bo[0]==0));

      for (;u<um;u+=8)
      {
         if (((pidn[0]<vol)||(!ibd))&&((piup[0]<vol)||(!ibu)))
         {
            doe(-0.5f,piup,pidn,u,s,rs);
            (*ro)=(*rs);
         }
         else
            (*ro).s=s0;

         piup+=4;
         pidn+=4;
         ro+=1;
      }
   }
   else
   {
      for (;u<um;u+=8)
      {
         doe(-0.5f,piup,pidn,u,s,rs);
         (*ro)=(*rs);

         piup+=4;
         pidn+=4;
         ro+=1;
      }
   }

   afree(rs);
}


void Dweo_blk(blk_grid_t grid,int n,int k,int l)
{
   int nb,isw,vol,volh,ibu,ibd;
   int *piup,*pidn,*ibp,*ibm;
   su3 *u,*um;
   spinor *r;
   spin_t *rs,*so;
   block_t *b;

   b=blk_list(grid,&nb,&isw)+n;

   vol=(*b).vol;
   volh=vol/2;
   so=(spin_t*)((*b).s[k]+volh);
   r=(*b).s[l];
   r[vol]=s0;

   piup=(*b).iup[volh];
   pidn=(*b).idn[volh];
   u=(*b).u;
   um=u+4*vol;
   rs=amalloc(sizeof(*rs),5);

   if ((*b).nbp)
   {
      ibu=((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0)&&(bc_type()==0));
      ibd=((cpr[0]==0)&&((*b).bo[0]==0));

      for (;u<um;u+=8)
      {
         if (((pidn[0]<vol)||(!ibd))&&((piup[0]<vol)||(!ibu)))
         {
            (*rs)=(*so);
            deo(0.5f,piup,pidn,u,rs,r);
         }
         else
            (*so).s=s0;

         piup+=4;
         pidn+=4;
         so+=1;
      }

      ibp=(*b).ibp;
      ibm=ibp+(*b).nbp/2;

      for (;ibp<ibm;ibp++)
         r[*ibp]=s0;
   }
   else
   {
      for (;u<um;u+=8)
      {
         (*rs)=(*so);
         deo(0.5f,piup,pidn,u,rs,r);

         piup+=4;
         pidn+=4;
         so+=1;
      }
   }

   afree(rs);
}


void Dwhat_blk(blk_grid_t grid,int n,float mu,int k,int l)
{
   int nb,isw,vol,volh,ibu,ibd;
   int *piup,*pidn,*ibp,*ibm;
   su3 *u,*um;
   pauli *sw;
   spinor *s,*r;
   spin_t *rs;
   block_t *b;

   b=blk_list(grid,&nb,&isw)+n;

   vol=(*b).vol;
   volh=vol/2;
   s=(*b).s[k];
   r=(*b).s[l];

   s[vol]=s0;
   r[vol]=s0;
   sw=(*b).sw;
   apply_sw(volh,mu,sw,s,r);

   piup=(*b).iup[volh];
   pidn=(*b).idn[volh];
   sw+=vol;
   u=(*b).u;
   um=u+4*vol;
   rs=amalloc(sizeof(*rs),5);

   if ((*b).nbp)
   {
      ibp=(*b).ibp;
      ibm=ibp+(*b).nbp/2;

      for (;ibp<ibm;ibp++)
         s[*ibp]=s0;

      ibu=((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0)&&(bc_type()==0));
      ibd=((cpr[0]==0)&&((*b).bo[0]==0));

      for (;u<um;u+=8)
      {
         if (((pidn[0]<vol)||(!ibd))&&((piup[0]<vol)||(!ibu)))
         {
            doe(-0.5f,piup,pidn,u,s,rs);

            mul_pauli2(0.0f,sw,&((*rs).s),&((*rs).s));

            deo(0.5f,piup,pidn,u,rs,r);
         }

         piup+=4;
         pidn+=4;
         sw+=2;
      }

      ibp=(*b).ibp;
      ibm=ibp+(*b).nbp/2;

      for (;ibp<ibm;ibp++)
         r[*ibp]=s0;
   }
   else
   {
      for (;u<um;u+=8)
      {
         doe(-0.5f,piup,pidn,u,s,rs);

         mul_pauli2(0.0f,sw,&((*rs).s),&((*rs).s));

         deo(0.5f,piup,pidn,u,rs,r);

         piup+=4;
         pidn+=4;
         sw+=2;
      }
   }

   afree(rs);
}
