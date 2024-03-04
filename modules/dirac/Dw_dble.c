
/*******************************************************************************
*
* File Dw_dble.c
*
* Copyright (C) 2005, 2011-2013, 2016, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Application of the O(a)-improved Wilson-Dirac operator D (double-
* precision programs).
*
*   void Dw_dble(double mu,spinor_dble *s,spinor_dble *r)
*     Depending on whether the twisted-mass flag is set or not, this
*     program applies D+i*mu*gamma_5*1e or D+i*mu*gamma_5 to the field
*     s and assigns the result to the field r.
*
*   void Dwee_dble(double mu,spinor_dble *s,spinor_dble *r)
*     Applies D_ee+i*mu*gamma_5 to the field s on the even points of the
*     lattice and assigns the result to the field r.
*
*   void Dwoo_dble(double mu,spinor_dble *s,spinor_dble *r)
*     Depending on whether the twisted-mass flag is set or not, this
*     program applies D_oo or D_oo+i*mu*gamma_5 to the field s on the
*     odd points of the lattice and assigns the result to the field r.
*
*   void Dwoe_dble(spinor_dble *s,spinor_dble *r)
*     Applies D_oe to the field s and assigns the result to the field r.
*
*   void Dweo_dble(spinor_dble *s,spinor_dble *r)
*     Applies D_eo to the field s and *subtracts* the result from the
*     field r.
*
*   void Dwhat_dble(double mu,spinor_dble *s,spinor_dble *r)
*     Applies Dhat+i*mu*gamma_5 to the field s and assigns the result to
*     the field r.
*
* The following programs operate on the fields in the n'th block b of the
* specified block grid:
*
*   void Dw_blk_dble(blk_grid_t grid,int n,double mu,int k,int l)
*     Depending on whether the twisted-mass flag is set or not, this
*     program applies D+i*mu*gamma_5*1e or D+i*mu*gamma_5 to the field
*     b.sd[k] and assigns the result to the field b.sd[l].
*
*   void Dwee_blk_dble(blk_grid_t grid,int n,double mu,int k,int l)
*     Applies D_ee+i*mu*gamma_5 to the field b.sd[k] on the even points and
*     assigns the result to the field b.sd[l].
*
*   void Dwoo_blk_dble(blk_grid_t grid,int n,double mu,int k,int l)
*     Depending on whether the twisted-mass flag is set or not, this
*     program applies D_oo or D_oo+i*mu*gamma_5 to the field b.sd[k] on
*     the odd points and assigns the result to the field b.sd[l].
*
*   void Dwoe_blk_dble(blk_grid_t grid,int n,int k,int l)
*     Applies D_oe to the field b.sd[k] and assigns the result to the field
*     b.sd[l].
*
*   void Dweo_blk_dble(blk_grid_t grid,int n,int k,int l)
*     Applies D_eo to the field b.sd[k] and *subtracts* the result from the
*     field b.sd[l].
*
*   void Dwhat_blk_dble(blk_grid_t grid,int n,double mu,int k,int l)
*     Applies Dhat+i*mu*gamma_5 to the field b.sd[k] and assigns the result
*     to the field b.sd[l].
*
* The notation and normalization conventions are specified in the notes
* "Implementation of the lattice Dirac operator" (file doc/dirac.pdf).
*
* In all these programs, it is assumed that the SW term is in the proper
* condition and that the global spinor fields have NSPIN elements. The
* programs check whether the twisted-mass flag (see flags/lat_parms.c) is
* set and turn off the twisted-mass term on the odd lattice sites if it is.
* The input and output fields may not coincide in the case of the programs
* Dw_dble(), Dwhat_dble(), Dw_blk_dble() and Dwhat_blk_dble().
*
* When the input and output fields are different, the input field is not
* changed except possibly at the points at global time 0 and NPROC0*L0-1,
* where both fields are set to zero if so required by the chosen boundary
* conditions. Depending on the operator considered, the fields are zeroed
* only on the even or odd points at these times.
*
* The programs Dw_dble(),..,Dwhat_dble() are assumed to be called by the
* OpenMP master thread on all MPI processes simultaneously.
*
* The programs Dw_blk_dble(),..,Dwhat_blk_dble() are thread-safe, but assume
* the grid "grid" is allocated and that the other parameters are not out of
* range.
*
* If SSE (AVX) instructions are used, the Dirac spinors must be aligned to
* a 16 (32) byte boundary.
*
*******************************************************************************/

#define DW_DBLE_C

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
#include "dirac.h"
#include "global.h"

#define N0 (NPROC0*L0)

typedef union
{
   spinor_dble s;
   weyl_dble w[2];
} spin_t;

static const spinor_dble sd0={{{0.0,0.0}}};

#if (defined AVX)
#include "avx.h"

#define _load_cst(c) \
__asm__ __volatile__ ("vbroadcastsd %0, %%ymm15 \n\t" \
                      : \
                      : \
                      "m" (c) \
                      : \
                      "xmm15")

#define _mul_cst() \
__asm__ __volatile__ ("vmulpd %%ymm15, %%ymm0, %%ymm0 \n\t" \
                      "vmulpd %%ymm15, %%ymm1, %%ymm1 \n\t" \
                      "vmulpd %%ymm15, %%ymm2, %%ymm2" \
                      : \
                      : \
                      : \
                      "xmm0", "xmm1", "xmm2")

#define _mul_cst_up() \
__asm__ __volatile__ ("vmulpd %%ymm15, %%ymm3, %%ymm3 \n\t" \
                      "vmulpd %%ymm15, %%ymm4, %%ymm4 \n\t" \
                      "vmulpd %%ymm15, %%ymm5, %%ymm5" \
                      : \
                      : \
                      : \
                      "xmm3", "xmm4", "xmm5")


static void doe(double coe,int *piup,int *pidn,
                su3_dble *u,spinor_dble *pk,spin_t *rs)
{
   spinor_dble *sp,*sm;

/******************************* direction +0 *********************************/

   sp=pk+(*(piup++));

   _avx_pair_load_dble((*sp).c1,(*sp).c2);
   _avx_pair_load_up_dble((*sp).c3,(*sp).c4);

   sm=pk+(*(pidn++));
   _prefetch_spinor_dble(sm);

   _avx_vector_add_dble();
   sp=pk+(*(piup++));
   _prefetch_spinor_dble(sp);
   _avx_su3_multiply_pair_dble(*u);

   _avx_weyl_store_up_dble((*rs).w[0]);
   _avx_weyl_store_up_dble((*rs).w[1]);

/******************************* direction -0 *********************************/

   _avx_pair_load_dble((*sm).c1,(*sm).c2);
   _avx_pair_load_up_dble((*sm).c3,(*sm).c4);

   _avx_vector_sub_dble();
   sm=pk+(*(pidn++));
   _prefetch_spinor_dble(sm);
   u+=1;
   _avx_su3_inverse_multiply_pair_dble(*u);

   _avx_weyl_load_dble((*rs).w[0]);
   _avx_vector_add_dble();
   _avx_weyl_store_dble((*rs).w[0]);

   _avx_weyl_load_dble((*rs).w[1]);
   _avx_vector_sub_dble();
   _avx_weyl_store_dble((*rs).w[1]);

/******************************* direction +1 *********************************/

   _avx_pair_load_dble((*sp).c1,(*sp).c2);
   _avx_pair_load_up_dble((*sp).c4,(*sp).c3);

   _avx_vector_i_add_dble();
   sp=pk+(*(piup++));
   _prefetch_spinor_dble(sp);
   u+=1;
   _avx_su3_multiply_pair_dble(*u);

   _avx_weyl_load_dble((*rs).w[0]);
   _avx_vector_add_dble();
   _avx_weyl_store_dble((*rs).w[0]);

   _avx_weyl_load_dble((*rs).w[1]);
   _avx_vector_xch_i_sub_dble();
   _avx_weyl_store_dble((*rs).w[1]);

/******************************* direction -1 *********************************/

   _avx_pair_load_dble((*sm).c1,(*sm).c2);
   _avx_pair_load_up_dble((*sm).c4,(*sm).c3);

   _avx_vector_i_sub_dble();
   sm=pk+(*(pidn++));
   _prefetch_spinor_dble(sm);
   u+=1;
   _avx_su3_inverse_multiply_pair_dble(*u);

   _avx_weyl_load_dble((*rs).w[0]);
   _avx_vector_add_dble();
   _avx_weyl_store_dble((*rs).w[0]);

   _avx_weyl_load_dble((*rs).w[1]);
   _avx_vector_xch_i_add_dble();
   _avx_weyl_store_dble((*rs).w[1]);

/******************************* direction +2 *********************************/

   _avx_pair_load_dble((*sp).c1,(*sp).c2);
   _avx_pair_load_up_dble((*sp).c4,(*sp).c3);

   _avx_vector_addsub_dble();
   u+=1;
   _avx_su3_multiply_pair_dble(*u);
   sp=pk+(*(piup));
   _prefetch_spinor_dble(sp);
   _avx_weyl_load_dble((*rs).w[0]);
   _avx_vector_add_dble();
   _avx_weyl_store_dble((*rs).w[0]);

   _avx_weyl_load_dble((*rs).w[1]);
   _avx_vector_xch_dble();
   _avx_vector_subadd_dble();
   _avx_weyl_store_dble((*rs).w[1]);

/******************************* direction -2 *********************************/

   _avx_pair_load_dble((*sm).c1,(*sm).c2);
   _avx_pair_load_up_dble((*sm).c4,(*sm).c3);

   _avx_vector_subadd_dble();
   sm=pk+(*(pidn));
   _prefetch_spinor_dble(sm);
   u+=1;
   _avx_su3_inverse_multiply_pair_dble(*u);

   _avx_weyl_load_dble((*rs).w[0]);
   _avx_vector_add_dble();
   _avx_weyl_store_dble((*rs).w[0]);

   _avx_weyl_load_dble((*rs).w[1]);
   _avx_vector_xch_dble();
   _avx_vector_addsub_dble();
   _avx_weyl_store_dble((*rs).w[1]);

/******************************* direction +3 *********************************/

   _avx_pair_load_dble((*sp).c1,(*sp).c2);
   _avx_pair_load_up_dble((*sp).c3,(*sp).c4);

   _avx_vector_i_addsub_dble();
   u+=1;
   _avx_su3_multiply_pair_dble(*u);

   _avx_weyl_load_dble((*rs).w[0]);
   _avx_vector_add_dble();
   _avx_weyl_store_dble((*rs).w[0]);

   _avx_weyl_load_dble((*rs).w[1]);
   _avx_vector_i_subadd_dble();
   _avx_weyl_store_dble((*rs).w[1]);

/******************************* direction -3 *********************************/

   _avx_pair_load_dble((*sm).c1,(*sm).c2);
   _avx_pair_load_up_dble((*sm).c3,(*sm).c4);

   _avx_vector_i_subadd_dble();
   u+=1;
   _avx_su3_inverse_multiply_pair_dble(*u);

   _load_cst(coe);
   _avx_weyl_load_dble((*rs).w[0]);
   _avx_vector_add_dble();
   _mul_cst();
   _avx_pair_store_dble((*rs).s.c1,(*rs).s.c2);

   _avx_weyl_load_dble((*rs).w[1]);
   _avx_vector_i_addsub_dble();
   _mul_cst();
   _avx_pair_store_dble((*rs).s.c3,(*rs).s.c4);

   _avx_zeroupper();
}


static void deo(double ceo,int *piup,int *pidn,
                su3_dble *u,spin_t *rs,spinor_dble *pl)
{
   spinor_dble *sp,*sm;

/******************************* direction +0 *********************************/

   sp=pl+(*(piup++));
   _prefetch_spinor_dble(sp);

   _load_cst(ceo);
   _avx_pair_load_dble((*rs).s.c1,(*rs).s.c2);
   _avx_pair_load_up_dble((*rs).s.c3,(*rs).s.c4);
   _mul_cst();
   _mul_cst_up();
   _avx_weyl_store_dble((*rs).w[0]);
   _avx_weyl_store_up_dble((*rs).w[1]);

   sm=pl+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _avx_vector_sub_dble();
   _avx_su3_inverse_multiply_pair_dble(*u);

   _avx_pair_load_dble((*sp).c1,(*sp).c2);
   _avx_vector_add_dble();
   _avx_pair_store_dble((*sp).c1,(*sp).c2);

   _avx_pair_load_dble((*sp).c3,(*sp).c4);
   _avx_vector_sub_dble();
   _avx_pair_store_dble((*sp).c3,(*sp).c4);

/******************************* direction -0 *********************************/

   _avx_weyl_load_dble((*rs).w[0]);
   _avx_weyl_load_up_dble((*rs).w[1]);

   sp=pl+(*(piup++));
   _prefetch_spinor_dble(sp);
   _avx_vector_add_dble();
   u+=1;
   _avx_su3_multiply_pair_dble(*u);

   _avx_pair_load_dble((*sm).c1,(*sm).c2);
   _avx_vector_add_dble();
   _avx_pair_store_dble((*sm).c1,(*sm).c2);

   _avx_pair_load_dble((*sm).c3,(*sm).c4);
   _avx_vector_add_dble();
   _avx_pair_store_dble((*sm).c3,(*sm).c4);

/******************************* direction +1 *********************************/

   _avx_weyl_load_dble((*rs).w[0]);
   _avx_weyl_load_up_dble((*rs).w[1]);

   sm=pl+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _avx_vector_xch_i_sub_dble();
   u+=1;
   _avx_su3_inverse_multiply_pair_dble(*u);

   _avx_pair_load_dble((*sp).c1,(*sp).c2);
   _avx_vector_add_dble();
   _avx_pair_store_dble((*sp).c1,(*sp).c2);

   _avx_pair_load_dble((*sp).c3,(*sp).c4);
   _avx_vector_xch_i_add_dble();
   _avx_pair_store_dble((*sp).c3,(*sp).c4);

/******************************* direction -1 *********************************/

   _avx_weyl_load_dble((*rs).w[0]);
   _avx_weyl_load_up_dble((*rs).w[1]);

   sp=pl+(*(piup++));
   _prefetch_spinor_dble(sp);
   _avx_vector_xch_i_add_dble();
   u+=1;
   _avx_su3_multiply_pair_dble(*u);

   _avx_pair_load_dble((*sm).c1,(*sm).c2);
   _avx_vector_add_dble();
   _avx_pair_store_dble((*sm).c1,(*sm).c2);

   _avx_pair_load_dble((*sm).c3,(*sm).c4);
   _avx_vector_xch_i_sub_dble();
   _avx_pair_store_dble((*sm).c3,(*sm).c4);

/******************************* direction +2 *********************************/

   _avx_weyl_load_dble((*rs).w[0]);
   _avx_weyl_load_up_dble((*rs).w[1]);

   sm=pl+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _avx_vector_xch_dble();
   _avx_vector_subadd_dble();
   u+=1;
   _avx_su3_inverse_multiply_pair_dble(*u);

   _avx_pair_load_dble((*sp).c1,(*sp).c2);
   _avx_vector_add_dble();
   _avx_pair_store_dble((*sp).c1,(*sp).c2);

   _avx_pair_load_dble((*sp).c3,(*sp).c4);
   _avx_vector_xch_dble();
   _avx_vector_addsub_dble();
   _avx_pair_store_dble((*sp).c3,(*sp).c4);

/******************************* direction -2 *********************************/

   _avx_weyl_load_dble((*rs).w[0]);
   _avx_weyl_load_up_dble((*rs).w[1]);

   sp=pl+(*(piup));
   _prefetch_spinor_dble(sp);
   _avx_vector_xch_dble();
   _avx_vector_addsub_dble();
   u+=1;
   _avx_su3_multiply_pair_dble(*u);

   _avx_pair_load_dble((*sm).c1,(*sm).c2);
   _avx_vector_add_dble();
   _avx_pair_store_dble((*sm).c1,(*sm).c2);

   _avx_pair_load_dble((*sm).c3,(*sm).c4);
   _avx_vector_xch_dble();
   _avx_vector_subadd_dble();
   _avx_pair_store_dble((*sm).c3,(*sm).c4);

/******************************* direction +3 *********************************/

   _avx_weyl_load_dble((*rs).w[0]);
   _avx_weyl_load_up_dble((*rs).w[1]);

   sm=pl+(*(pidn));
   _prefetch_spinor_dble(sm);
   _avx_vector_i_subadd_dble();
   u+=1;
   _avx_su3_inverse_multiply_pair_dble(*u);

   _avx_pair_load_dble((*sp).c1,(*sp).c2);
   _avx_vector_add_dble();
   _avx_pair_store_dble((*sp).c1,(*sp).c2);

   _avx_pair_load_dble((*sp).c3,(*sp).c4);
   _avx_vector_i_addsub_dble();
   _avx_pair_store_dble((*sp).c3,(*sp).c4);

/******************************* direction -3 *********************************/

   _avx_weyl_load_dble((*rs).w[0]);
   _avx_weyl_load_up_dble((*rs).w[1]);

   _avx_vector_i_addsub_dble();
   u+=1;
   _avx_su3_multiply_pair_dble(*u);

   _avx_pair_load_dble((*sm).c1,(*sm).c2);
   _avx_vector_add_dble();
   _avx_pair_store_dble((*sm).c1,(*sm).c2);

   _avx_pair_load_dble((*sm).c3,(*sm).c4);
   _avx_vector_i_subadd_dble();
   _avx_pair_store_dble((*sm).c3,(*sm).c4);

   _avx_zeroupper();
}

#elif (defined x64)
#include "sse2.h"

#define _load_cst(c) \
__asm__ __volatile__ ("movddup %0, %%xmm15" \
                      : \
                      : \
                      "m" (c) \
                      : \
                      "xmm15")

#define _mul_cst() \
__asm__ __volatile__ ("mulpd %%xmm15, %%xmm0 \n\t" \
                      "mulpd %%xmm15, %%xmm1 \n\t" \
                      "mulpd %%xmm15, %%xmm2" \
                      : \
                      : \
                      : \
                      "xmm0", "xmm1", "xmm2")

#define _mul_cst_up() \
__asm__ __volatile__ ("mulpd %%xmm15, %%xmm3 \n\t" \
                      "mulpd %%xmm15, %%xmm4 \n\t" \
                      "mulpd %%xmm15, %%xmm5" \
                      : \
                      : \
                      : \
                      "xmm3", "xmm4", "xmm5")


static void doe(double coe,int *piup,int *pidn,
                su3_dble *u,spinor_dble *pk,spin_t *rs)
{
   spinor_dble *sp,*sm;

/******************************* direction +0 *********************************/

   sp=pk+(*(piup++));

   _sse_load_dble((*sp).c1);
   _sse_load_up_dble((*sp).c3);

   sm=pk+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _sse_vector_add_dble();
   _sse_su3_multiply_dble(*u);
   _sse_store_up_dble((*rs).s.c1);
   _sse_store_up_dble((*rs).s.c3);

   _sse_load_dble((*sp).c2);
   _sse_load_up_dble((*sp).c4);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_add_dble();
   _sse_su3_multiply_dble(*u);

   _sse_store_up_dble((*rs).s.c2);
   _sse_store_up_dble((*rs).s.c4);

/******************************* direction -0 *********************************/

   _sse_load_dble((*sm).c1);
   _sse_load_up_dble((*sm).c3);

   sp=pk+(*(piup++));
   _prefetch_spinor_dble(sp);
   _sse_vector_sub_dble();
   u+=1;
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*rs).s.c1);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c1);

   _sse_load_dble((*rs).s.c3);
   _sse_vector_sub_dble();
   _sse_store_dble((*rs).s.c3);

   _sse_load_dble((*sm).c2);
   _sse_load_up_dble((*sm).c4);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_sub_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*rs).s.c2);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c2);

   _sse_load_dble((*rs).s.c4);
   _sse_vector_sub_dble();
   _sse_store_dble((*rs).s.c4);

/******************************* direction +1 *********************************/

   _sse_load_dble((*sp).c1);
   _sse_load_up_dble((*sp).c4);

   sm=pk+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _sse_vector_i_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*rs).s.c1);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c1);

   _sse_load_dble((*rs).s.c4);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble((*rs).s.c4);

   _sse_load_dble((*sp).c2);
   _sse_load_up_dble((*sp).c3);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_i_add_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*rs).s.c2);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c2);

   _sse_load_dble((*rs).s.c3);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble((*rs).s.c3);

/******************************* direction -1 *********************************/

   _sse_load_dble((*sm).c1);
   _sse_load_up_dble((*sm).c4);

   sp=pk+(*(piup++));
   _prefetch_spinor_dble(sp);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   u+=1;
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*rs).s.c1);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c1);

   _sse_load_dble((*rs).s.c4);
   _sse_vector_i_add_dble();
   _sse_store_dble((*rs).s.c4);

   _sse_load_dble((*sm).c2);
   _sse_load_up_dble((*sm).c3);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*rs).s.c2);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c2);

   _sse_load_dble((*rs).s.c3);
   _sse_vector_i_add_dble();
   _sse_store_dble((*rs).s.c3);

/******************************* direction +2 *********************************/

   _sse_load_dble((*sp).c1);
   _sse_load_up_dble((*sp).c4);

   sm=pk+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _sse_vector_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*rs).s.c1);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c1);

   _sse_load_dble((*rs).s.c4);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c4);

   _sse_load_dble((*sp).c2);
   _sse_load_up_dble((*sp).c3);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_sub_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*rs).s.c2);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c2);

   _sse_load_dble((*rs).s.c3);
   _sse_vector_sub_dble();
   _sse_store_dble((*rs).s.c3);

/******************************* direction -2 *********************************/

   _sse_load_dble((*sm).c1);
   _sse_load_up_dble((*sm).c4);

   sp=pk+(*(piup));
   _prefetch_spinor_dble(sp);
   _sse_vector_sub_dble();
   u+=1;
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*rs).s.c1);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c1);

   _sse_load_dble((*rs).s.c4);
   _sse_vector_sub_dble();
   _sse_store_dble((*rs).s.c4);

   _sse_load_dble((*sm).c2);
   _sse_load_up_dble((*sm).c3);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_add_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*rs).s.c2);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c2);

   _sse_load_dble((*rs).s.c3);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c3);

/******************************* direction +3 *********************************/

   _sse_load_dble((*sp).c1);
   _sse_load_up_dble((*sp).c3);

   sm=pk+(*(pidn));
   _prefetch_spinor_dble(sm);
   _sse_vector_i_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*rs).s.c1);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c1);

   _sse_load_dble((*rs).s.c3);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble((*rs).s.c3);

   _sse_load_dble((*sp).c2);
   _sse_load_up_dble((*sp).c4);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*rs).s.c2);
   _sse_vector_add_dble();
   _sse_store_dble((*rs).s.c2);

   _sse_load_dble((*rs).s.c4);
   _sse_vector_i_add_dble();
   _sse_store_dble((*rs).s.c4);

/******************************* direction -3 *********************************/

   _sse_load_dble((*sm).c1);
   _sse_load_up_dble((*sm).c3);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   u+=1;
   _sse_su3_inverse_multiply_dble(*u);

   _load_cst(coe);
   _sse_load_dble((*rs).s.c1);
   _sse_vector_add_dble();
   _mul_cst();
   _sse_store_dble((*rs).s.c1);

   _sse_load_dble((*rs).s.c3);
   _sse_vector_i_add_dble();
   _mul_cst();
   _sse_store_dble((*rs).s.c3);

   _sse_load_dble((*sm).c2);
   _sse_load_up_dble((*sm).c4);

   u+=1;
   _prefetch_su3_dble(u);
   u-=1;

   _sse_vector_i_add_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _load_cst(coe);
   _sse_load_dble((*rs).s.c2);
   _sse_vector_add_dble();
   _mul_cst();
   _sse_store_dble((*rs).s.c2);

   _sse_load_dble((*rs).s.c4);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _mul_cst();
   _sse_store_dble((*rs).s.c4);
}


static void deo(double ceo,int *piup,int *pidn,
                su3_dble *u,spin_t *rs,spinor_dble *pl)
{
   spinor_dble *sp,*sm;

/******************************* direction +0 *********************************/

   sp=pl+(*(piup++));
   _prefetch_spinor_dble(sp);

   _load_cst(ceo);
   _sse_load_dble((*rs).s.c1);
   _sse_load_up_dble((*rs).s.c3);
   _mul_cst();
   _mul_cst_up();
   _sse_store_dble((*rs).s.c1);
   _sse_store_up_dble((*rs).s.c3);

   sm=pl+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _sse_vector_sub_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c1);

   _sse_load_dble((*sp).c3);
   _sse_vector_sub_dble();
   _sse_store_dble((*sp).c3);

   _load_cst(ceo);
   _sse_load_dble((*rs).s.c2);
   _sse_load_up_dble((*rs).s.c4);
   _mul_cst();
   _mul_cst_up();
   _sse_store_dble((*rs).s.c2);
   _sse_store_up_dble((*rs).s.c4);

   _sse_vector_sub_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c2);

   _sse_load_dble((*sp).c4);
   _sse_vector_sub_dble();
   _sse_store_dble((*sp).c4);

/******************************* direction -0 *********************************/

   _sse_load_dble((*rs).s.c1);
   _sse_load_up_dble((*rs).s.c3);

   sp=pl+(*(piup++));
   _prefetch_spinor_dble(sp);
   _sse_vector_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c1);

   _sse_load_dble((*sm).c3);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c3);

   _sse_load_dble((*rs).s.c2);
   _sse_load_up_dble((*rs).s.c4);

   _sse_vector_add_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c2);

   _sse_load_dble((*sm).c4);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c4);

/******************************* direction +1 *********************************/

   _sse_load_dble((*rs).s.c1);
   _sse_load_up_dble((*rs).s.c4);

   sm=pl+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   u+=1;
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c1);

   _sse_load_dble((*sp).c4);
   _sse_vector_i_add_dble();
   _sse_store_dble((*sp).c4);

   _sse_load_dble((*rs).s.c2);
   _sse_load_up_dble((*rs).s.c3);

   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c2);

   _sse_load_dble((*sp).c3);
   _sse_vector_i_add_dble();
   _sse_store_dble((*sp).c3);

/******************************* direction -1 *********************************/

   _sse_load_dble((*rs).s.c1);
   _sse_load_up_dble((*rs).s.c4);

   sp=pl+(*(piup++));
   _prefetch_spinor_dble(sp);
   _sse_vector_i_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c1);

   _sse_load_dble((*sm).c4);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble((*sm).c4);

   _sse_load_dble((*rs).s.c2);
   _sse_load_up_dble((*rs).s.c3);

   _sse_vector_i_add_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c2);

   _sse_load_dble((*sm).c3);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble((*sm).c3);

/******************************* direction +2 *********************************/

   _sse_load_dble((*rs).s.c1);
   _sse_load_up_dble((*rs).s.c4);

   sm=pl+(*(pidn++));
   _prefetch_spinor_dble(sm);
   _sse_vector_sub_dble();
   u+=1;
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c1);

   _sse_load_dble((*sp).c4);
   _sse_vector_sub_dble();
   _sse_store_dble((*sp).c4);

   _sse_load_dble((*rs).s.c2);
   _sse_load_up_dble((*rs).s.c3);

   _sse_vector_add_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c2);

   _sse_load_dble((*sp).c3);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c3);

/******************************* direction -2 *********************************/

   _sse_load_dble((*rs).s.c1);
   _sse_load_up_dble((*rs).s.c4);

   sp=pl+(*(piup));
   _prefetch_spinor_dble(sp);
   _sse_vector_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c1);

   _sse_load_dble((*sm).c4);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c4);

   _sse_load_dble((*rs).s.c2);
   _sse_load_up_dble((*rs).s.c3);

   _sse_vector_sub_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c2);

   _sse_load_dble((*sm).c3);
   _sse_vector_sub_dble();
   _sse_store_dble((*sm).c3);

/******************************* direction +3 *********************************/

   _sse_load_dble((*rs).s.c1);
   _sse_load_up_dble((*rs).s.c3);

   sm=pl+(*(pidn));
   _prefetch_spinor_dble(sm);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   u+=1;
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c1);

   _sse_load_dble((*sp).c3);
   _sse_vector_i_add_dble();
   _sse_store_dble((*sp).c3);

   _sse_load_dble((*rs).s.c2);
   _sse_load_up_dble((*rs).s.c4);

   _sse_vector_i_add_dble();
   _sse_su3_inverse_multiply_dble(*u);

   _sse_load_dble((*sp).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sp).c2);

   _sse_load_dble((*sp).c4);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble((*sp).c4);

/******************************* direction -3 *********************************/

   _sse_load_dble((*rs).s.c1);
   _sse_load_up_dble((*rs).s.c3);

   _sse_vector_i_add_dble();
   u+=1;
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c1);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c1);

   _sse_load_dble((*sm).c3);
   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_store_dble((*sm).c3);

   _sse_load_dble((*rs).s.c2);
   _sse_load_up_dble((*rs).s.c4);

   _sse_vector_i_mul_dble();
   _sse_vector_sub_dble();
   _sse_su3_multiply_dble(*u);

   _sse_load_dble((*sm).c2);
   _sse_vector_add_dble();
   _sse_store_dble((*sm).c2);

   _sse_load_dble((*sm).c4);
   _sse_vector_i_add_dble();
   _sse_store_dble((*sm).c4);
}

#else

#define _vector_mul_assign(r,c) \
   (r).c1.re*=(c); \
   (r).c1.im*=(c); \
   (r).c2.re*=(c); \
   (r).c2.im*=(c); \
   (r).c3.re*=(c); \
   (r).c3.im*=(c)


static void doe(double coe,int *piup,int *pidn,
                su3_dble *u,spinor_dble *pk,spin_t *rs)
{
   spinor_dble *sp,*sm;
   su3_vector_dble psi,chi;

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


static void deo(double ceo,int *piup,int *pidn,
                su3_dble *u,spin_t *rs,spinor_dble *pl)
{
   spinor_dble *sp,*sm;
   su3_vector_dble psi,chi;

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

void Dw_dble(double mu,spinor_dble *s,spinor_dble *r)
{
   int bc;
   int k,ix,t,ofs,vol,isb;
   int *piup,*pidn;
   double muo;
   su3_dble *udb,*ud;
   pauli_dble *swb,*swd;
   spin_t *rs,*so,*ro;
   tm_parms_t tm;

   cpsd_int_bnd(0x1,s);

   bc=bc_type();
   tm=tm_parms();
   if (tm.eoflg==1)
      muo=0.0;
   else
      muo=mu;

   udb=udfld();
   swb=swdfld();

#pragma omp parallel private(k,ix,t,vol,ofs,isb,piup,pidn,ud,swd,rs,so,ro)
   {
      k=omp_get_thread_num();

      vol=(BNDRY/2)/NTHREAD;
      ofs=k*vol;

      if (k==(NTHREAD-1))
         vol=(BNDRY/2)-ofs;

      set_sd2zero(vol,0,r+VOLUME+ofs);

      vol=VOLUME_TRD/2;
      ofs=k*vol;
      apply_sw_dble(vol,mu,swb+2*ofs,s+ofs,r+ofs);

      ud=udb+8*ofs;
      ofs+=VOLUME/2;
      piup=iup[ofs];
      pidn=idn[ofs];
      swd=swb+2*ofs;
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
                  doe(-0.5,piup,pidn,ud,s,rs);

                  mul_pauli_dble(muo,swd,(*so).w,(*ro).w);
                  mul_pauli_dble(-muo,swd+1,(*so).w+1,(*ro).w+1);

                  _vector_add_assign((*ro).s.c1,(*rs).s.c1);
                  _vector_add_assign((*ro).s.c2,(*rs).s.c2);
                  _vector_add_assign((*ro).s.c3,(*rs).s.c3);
                  _vector_add_assign((*ro).s.c4,(*rs).s.c4);
                  (*rs)=(*so);

                  deo(-0.5,piup,pidn,ud,rs,r);
               }
               else
               {
                  (*so).s=sd0;
                  (*ro).s=sd0;
               }

               piup+=4;
               pidn+=4;
               ud+=8;
               swd+=2;
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
               doe(-0.5,piup,pidn,ud,s,rs);

               mul_pauli_dble(muo,swd,(*so).w,(*ro).w);
               mul_pauli_dble(-muo,swd+1,(*so).w+1,(*ro).w+1);

               _vector_add_assign((*ro).s.c1,(*rs).s.c1);
               _vector_add_assign((*ro).s.c2,(*rs).s.c2);
               _vector_add_assign((*ro).s.c3,(*rs).s.c3);
               _vector_add_assign((*ro).s.c4,(*rs).s.c4);
               (*rs)=(*so);

               deo(-0.5,piup,pidn,ud,rs,r);

               piup+=4;
               pidn+=4;
               ud+=8;
               swd+=2;
               so+=1;
               ro+=1;
            }
         }
      }

      afree(rs);
   }

   cpsd_ext_bnd(0x1,r);
}


void Dwee_dble(double mu,spinor_dble *s,spinor_dble *r)
{
   int bc;
   int k,ix,t,ofs,vol;
   pauli_dble *swb,*swd;
   spin_t *se,*re;

   bc=bc_type();
   swb=swdfld();

#pragma omp parallel private(k,ix,t,vol,ofs,swd,se,re)
   {
      k=omp_get_thread_num();

      vol=VOLUME_TRD/2;
      ofs=k*vol;
      swd=swb+2*ofs;
      se=(spin_t*)(s+ofs);
      re=(spin_t*)(r+ofs);

      if (((cpr[0]==0)&&(bc!=3))||((cpr[0]==(NPROC0-1))&&(bc==0)))
      {
         for (ix=ofs;ix<(ofs+vol);ix++)
         {
            t=global_time(ix);

            if ((t>0)&&((t<(N0-1))||(bc!=0)))
            {
               mul_pauli_dble(mu,swd,(*se).w,(*re).w);
               mul_pauli_dble(-mu,swd+1,(*se).w+1,(*re).w+1);
            }
            else
            {
               (*se).s=sd0;
               (*re).s=sd0;
            }

            swd+=2;
            se+=1;
            re+=1;
         }
      }
      else
      {
         for (ix=ofs;ix<(ofs+vol);ix++)
         {
            mul_pauli_dble(mu,swd,(*se).w,(*re).w);
            mul_pauli_dble(-mu,swd+1,(*se).w+1,(*re).w+1);

            swd+=2;
            se+=1;
            re+=1;
         }
      }
   }
}


void Dwoo_dble(double mu,spinor_dble *s,spinor_dble *r)
{
   int bc;
   int k,ix,t,ofs,vol;
   pauli_dble *swb,*swd;
   spin_t *so,*ro;
   tm_parms_t tm;

   bc=bc_type();
   tm=tm_parms();
   if (tm.eoflg==1)
      mu=0.0;
   swb=swdfld();

#pragma omp parallel private(k,ix,t,vol,ofs,swd,so,ro)
   {
      k=omp_get_thread_num();

      vol=VOLUME_TRD/2;
      ofs=VOLUME/2+k*vol;
      swd=swb+2*ofs;
      so=(spin_t*)(s+ofs);
      ro=(spin_t*)(r+ofs);

      if (((cpr[0]==0)&&(bc!=3))||((cpr[0]==(NPROC0-1))&&(bc==0)))
      {
         for (ix=ofs;ix<(ofs+vol);ix++)
         {
            t=global_time(ix);

            if ((t>0)&&((t<(N0-1))||(bc!=0)))
            {
               mul_pauli_dble(mu,swd,(*so).w,(*ro).w);
               mul_pauli_dble(-mu,swd+1,(*so).w+1,(*ro).w+1);
            }
            else
            {
               (*so).s=sd0;
               (*ro).s=sd0;
            }

            swd+=2;
            so+=1;
            ro+=1;
         }
      }
      else
      {
         for (ix=ofs;ix<(ofs+vol);ix++)
         {
            mul_pauli_dble(mu,swd,(*so).w,(*ro).w);
            mul_pauli_dble(-mu,swd+1,(*so).w+1,(*ro).w+1);

            swd+=2;
            so+=1;
            ro+=1;
         }
      }
   }
}


void Dwoe_dble(spinor_dble *s,spinor_dble *r)
{
   int bc;
   int k,ix,t,ofs,vol,isb;
   int *piup,*pidn;
   su3_dble *udb,*ud;
   spin_t *rs,*ro;

   cpsd_int_bnd(0x1,s);

   bc=bc_type();
   udb=udfld();

#pragma omp parallel private(k,ix,t,vol,ofs,isb,piup,pidn,ud,rs,ro)
   {
      k=omp_get_thread_num();

      ofs=k*VOLUME_TRD/2;
      ud=udb+8*ofs;
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
                  doe(-0.5,piup,pidn,ud,s,rs);
                  (*ro)=(*rs);
               }
               else
                  (*ro).s=sd0;

               piup+=4;
               pidn+=4;
               ud+=8;
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
               doe(-0.5,piup,pidn,ud,s,rs);
               (*ro)=(*rs);

               piup+=4;
               pidn+=4;
               ud+=8;
               ro+=1;
            }
         }
      }

      afree(rs);
   }
}


void Dweo_dble(spinor_dble *s,spinor_dble *r)
{
   int bc;
   int k,ix,t,ofs,vol,isb;
   int *piup,*pidn;
   su3_dble *udb,*ud;
   spin_t *rs,*so;

   bc=bc_type();
   udb=udfld();

#pragma omp parallel private(k,ix,t,vol,ofs,isb,piup,pidn,ud,rs,so)
   {
      k=omp_get_thread_num();

      vol=(BNDRY/2)/NTHREAD;
      ofs=k*vol;

      if (k==(NTHREAD-1))
         vol=(BNDRY/2)-ofs;

      set_sd2zero(vol,0,r+VOLUME+ofs);

      vol=VOLUME_TRD/2;
      ofs=k*vol;
      ud=udb+8*ofs;
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
                  deo(0.5,piup,pidn,ud,rs,r);
               }
               else
                  (*so).s=sd0;

               piup+=4;
               pidn+=4;
               ud+=8;
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
               deo(0.5,piup,pidn,ud,rs,r);

               piup+=4;
               pidn+=4;
               ud+=8;
               so+=1;
            }
         }
      }

      afree(rs);
   }

   cpsd_ext_bnd(0x1,r);
}


void Dwhat_dble(double mu,spinor_dble *s,spinor_dble *r)
{
   int bc;
   int k,ix,t,ofs,vol,isb;
   int *piup,*pidn;
   su3_dble *udb,*ud;
   pauli_dble *swb,*swd;
   spin_t *rs;

   cpsd_int_bnd(0x1,s);

   bc=bc_type();
   udb=udfld();
   swb=swdfld();

#pragma omp parallel private(k,ix,t,vol,ofs,isb,piup,pidn,ud,swd,rs)
   {
      k=omp_get_thread_num();

      vol=(BNDRY/2)/NTHREAD;
      ofs=k*vol;

      if (k==(NTHREAD-1))
         vol=(BNDRY/2)-ofs;

      set_sd2zero(vol,0,r+VOLUME+ofs);

      vol=VOLUME_TRD/2;
      ofs=k*vol;
      apply_sw_dble(vol,mu,swb+2*ofs,s+ofs,r+ofs);

      ud=udb+8*ofs;
      ofs+=VOLUME/2;
      piup=iup[ofs];
      pidn=idn[ofs];
      swd=swb+2*ofs;
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
                  doe(-0.5,piup,pidn,ud,s,rs);

                  mul_pauli_dble(0.0,swd,(*rs).w,(*rs).w);
                  mul_pauli_dble(0.0,swd+1,(*rs).w+1,(*rs).w+1);

                  deo(0.5,piup,pidn,ud,rs,r);
               }

               piup+=4;
               pidn+=4;
               ud+=8;
               swd+=2;
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
               doe(-0.5,piup,pidn,ud,s,rs);

               mul_pauli_dble(0.0,swd,(*rs).w,(*rs).w);
               mul_pauli_dble(0.0,swd+1,(*rs).w+1,(*rs).w+1);

               deo(0.5,piup,pidn,ud,rs,r);

               piup+=4;
               pidn+=4;
               ud+=8;
               swd+=2;
            }
         }
      }

      afree(rs);
   }

   cpsd_ext_bnd(0x1,r);
}


void Dw_blk_dble(blk_grid_t grid,int n,double mu,int k,int l)
{
   int nb,isw,vol,volh,ibu,ibd;
   int *piup,*pidn,*ibp,*ibm;
   su3_dble *ud,*um;
   pauli_dble *swd;
   spinor_dble *s,*r;
   spin_t *rs,*so,*ro;
   block_t *b;
   tm_parms_t tm;

   b=blk_list(grid,&nb,&isw)+n;

   vol=(*b).vol;
   volh=vol/2;
   s=(*b).sd[k];
   r=(*b).sd[l];
   so=(spin_t*)(s+volh);
   ro=(spin_t*)(r+volh);

   s[vol]=sd0;
   r[vol]=sd0;
   swd=(*b).swd;
   apply_sw_dble(volh,mu,swd,s,r);
   tm=tm_parms();
   if (tm.eoflg==1)
      mu=0.0;

   piup=(*b).iup[volh];
   pidn=(*b).idn[volh];
   swd+=vol;
   ud=(*b).ud;
   um=ud+4*vol;
   rs=amalloc(sizeof(*rs),5);

   if ((*b).nbp)
   {
      ibp=(*b).ibp;
      ibm=ibp+(*b).nbp/2;

      for (;ibp<ibm;ibp++)
         s[*ibp]=sd0;

      ibu=((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0)&&(bc_type()==0));
      ibd=((cpr[0]==0)&&((*b).bo[0]==0));

      for (;ud<um;ud+=8)
      {
         if (((pidn[0]<vol)||(!ibd))&&((piup[0]<vol)||(!ibu)))
         {
            doe(-0.5,piup,pidn,ud,s,rs);

            mul_pauli_dble(mu,swd,(*so).w,(*ro).w);
            mul_pauli_dble(-mu,swd+1,(*so).w+1,(*ro).w+1);

            _vector_add_assign((*ro).s.c1,(*rs).s.c1);
            _vector_add_assign((*ro).s.c2,(*rs).s.c2);
            _vector_add_assign((*ro).s.c3,(*rs).s.c3);
            _vector_add_assign((*ro).s.c4,(*rs).s.c4);
            (*rs)=(*so);

            deo(-0.5,piup,pidn,ud,rs,r);
         }
         else
         {
            (*so).s=sd0;
            (*ro).s=sd0;
         }

         piup+=4;
         pidn+=4;
         swd+=2;
         ro+=1;
         so+=1;
      }

      ibp=(*b).ibp;
      ibm=ibp+(*b).nbp/2;

      for (;ibp<ibm;ibp++)
         r[*ibp]=sd0;
   }
   else
   {
      for (;ud<um;ud+=8)
      {
         doe(-0.5,piup,pidn,ud,s,rs);

         mul_pauli_dble(mu,swd,(*so).w,(*ro).w);
         mul_pauli_dble(-mu,swd+1,(*so).w+1,(*ro).w+1);

         _vector_add_assign((*ro).s.c1,(*rs).s.c1);
         _vector_add_assign((*ro).s.c2,(*rs).s.c2);
         _vector_add_assign((*ro).s.c3,(*rs).s.c3);
         _vector_add_assign((*ro).s.c4,(*rs).s.c4);
         (*rs)=(*so);

         deo(-0.5,piup,pidn,ud,rs,r);

         piup+=4;
         pidn+=4;
         swd+=2;
         ro+=1;
         so+=1;
      }
   }

   afree(rs);
}


void Dwee_blk_dble(blk_grid_t grid,int n,double mu,int k,int l)
{
   int nb,isw,vol,ibu,ibd;
   int *piup,*pidn;
   pauli_dble *swd,*swm;
   spin_t *se,*re;
   block_t *b;

   b=blk_list(grid,&nb,&isw)+n;

   vol=(*b).vol;
   se=(spin_t*)((*b).sd[k]);
   re=(spin_t*)((*b).sd[l]);
   swd=(*b).swd;
   swm=swd+vol;

   if ((*b).nbp)
   {
      piup=(*b).iup[0];
      pidn=(*b).idn[0];

      ibu=((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0)&&(bc_type()==0));
      ibd=((cpr[0]==0)&&((*b).bo[0]==0));

      for (;swd<swm;swd+=2)
      {
         if (((pidn[0]<vol)||(!ibd))&&((piup[0]<vol)||(!ibu)))
         {
            mul_pauli_dble(mu,swd,(*se).w,(*re).w);
            mul_pauli_dble(-mu,swd+1,(*se).w+1,(*re).w+1);
         }
         else
         {
            (*se).s=sd0;
            (*re).s=sd0;
         }

         piup+=4;
         pidn+=4;
         se+=1;
         re+=1;
      }
   }
   else
   {
      for (;swd<swm;swd+=2)
      {
         mul_pauli_dble(mu,swd,(*se).w,(*re).w);
         mul_pauli_dble(-mu,swd+1,(*se).w+1,(*re).w+1);

         se+=1;
         re+=1;
      }
   }
}


void Dwoo_blk_dble(blk_grid_t grid,int n,double mu,int k,int l)
{
   int nb,isw,vol,volh,ibu,ibd;
   int *piup,*pidn;
   pauli_dble *swd,*swm;
   spin_t *so,*ro;
   block_t *b;
   tm_parms_t tm;

   b=blk_list(grid,&nb,&isw)+n;

   vol=(*b).vol;
   volh=vol/2;
   so=(spin_t*)((*b).sd[k]+volh);
   ro=(spin_t*)((*b).sd[l]+volh);
   tm=tm_parms();
   if (tm.eoflg==1)
      mu=0.0;

   swd=(*b).swd+vol;
   swm=swd+vol;

   if ((*b).nbp)
   {
      piup=(*b).iup[volh];
      pidn=(*b).idn[volh];

      ibu=((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0)&&(bc_type()==0));
      ibd=((cpr[0]==0)&&((*b).bo[0]==0));

      for (;swd<swm;swd+=2)
      {
         if (((pidn[0]<vol)||(!ibd))&&((piup[0]<vol)||(!ibu)))
         {
            mul_pauli_dble(mu,swd,(*so).w,(*ro).w);
            mul_pauli_dble(-mu,swd+1,(*so).w+1,(*ro).w+1);
         }
         else
         {
            (*so).s=sd0;
            (*ro).s=sd0;
         }

         piup+=4;
         pidn+=4;
         so+=1;
         ro+=1;
      }
   }
   else
   {
      for (;swd<swm;swd+=2)
      {
         mul_pauli_dble(mu,swd,(*so).w,(*ro).w);
         mul_pauli_dble(-mu,swd+1,(*so).w+1,(*ro).w+1);

         so+=1;
         ro+=1;
      }
   }
}


void Dwoe_blk_dble(blk_grid_t grid,int n,int k,int l)
{
   int nb,isw,vol,volh,ibu,ibd;
   int *piup,*pidn,*ibp,*ibm;
   su3_dble *ud,*um;
   spinor_dble *s;
   spin_t *rs,*ro;
   block_t *b;

   b=blk_list(grid,&nb,&isw)+n;

   vol=(*b).vol;
   volh=vol/2;
   s=(*b).sd[k];
   ro=(spin_t*)((*b).sd[l]+volh);
   s[vol]=sd0;

   piup=(*b).iup[volh];
   pidn=(*b).idn[volh];
   ud=(*b).ud;
   um=ud+4*vol;
   rs=amalloc(sizeof(*rs),5);

   if ((*b).nbp)
   {
      ibp=(*b).ibp;
      ibm=ibp+(*b).nbp/2;

      for (;ibp<ibm;ibp++)
         s[*ibp]=sd0;

      ibu=((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0)&&(bc_type()==0));
      ibd=((cpr[0]==0)&&((*b).bo[0]==0));

      for (;ud<um;ud+=8)
      {
         if (((pidn[0]<vol)||(!ibd))&&((piup[0]<vol)||(!ibu)))
         {
            doe(-0.5,piup,pidn,ud,s,rs);
            (*ro)=(*rs);
         }
         else
            (*ro).s=sd0;

         piup+=4;
         pidn+=4;
         ro+=1;
      }
   }
   else
   {
      for (;ud<um;ud+=8)
      {
         doe(-0.5,piup,pidn,ud,s,rs);
         (*ro)=(*rs);

         piup+=4;
         pidn+=4;
         ro+=1;
      }
   }

   afree(rs);
}


void Dweo_blk_dble(blk_grid_t grid,int n,int k,int l)
{
   int nb,isw,vol,volh,ibu,ibd;
   int *piup,*pidn,*ibp,*ibm;
   su3_dble *ud,*um;
   spinor_dble *r;
   spin_t *rs,*so;
   block_t *b;

   b=blk_list(grid,&nb,&isw)+n;

   vol=(*b).vol;
   volh=vol/2;
   so=(spin_t*)((*b).sd[k]+volh);
   r=(*b).sd[l];
   r[vol]=sd0;

   piup=(*b).iup[volh];
   pidn=(*b).idn[volh];
   ud=(*b).ud;
   um=ud+4*vol;
   rs=amalloc(sizeof(*rs),5);

   if ((*b).nbp)
   {
      ibu=((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0)&&(bc_type()==0));
      ibd=((cpr[0]==0)&&((*b).bo[0]==0));

      for (;ud<um;ud+=8)
      {
         if (((pidn[0]<vol)||(!ibd))&&((piup[0]<vol)||(!ibu)))
         {
            (*rs)=(*so);
            deo(0.5,piup,pidn,ud,rs,r);
         }
         else
            (*so).s=sd0;

         piup+=4;
         pidn+=4;
         so+=1;
      }

      ibp=(*b).ibp;
      ibm=ibp+(*b).nbp/2;

      for (;ibp<ibm;ibp++)
         r[*ibp]=sd0;
   }
   else
   {
      for (;ud<um;ud+=8)
      {
         (*rs)=(*so);
         deo(0.5,piup,pidn,ud,rs,r);

         piup+=4;
         pidn+=4;
         so+=1;
      }
   }

   afree(rs);
}


void Dwhat_blk_dble(blk_grid_t grid,int n,double mu,int k,int l)
{
   int nb,isw,vol,volh,ibu,ibd;
   int *piup,*pidn,*ibp,*ibm;
   su3_dble *ud,*um;
   pauli_dble *swd;
   spinor_dble *s,*r;
   spin_t *rs;
   block_t *b;

   b=blk_list(grid,&nb,&isw)+n;

   vol=(*b).vol;
   volh=vol/2;
   s=(*b).sd[k];
   r=(*b).sd[l];

   s[vol]=sd0;
   r[vol]=sd0;
   swd=(*b).swd;
   apply_sw_dble(volh,mu,swd,s,r);

   piup=(*b).iup[volh];
   pidn=(*b).idn[volh];
   swd+=vol;
   ud=(*b).ud;
   um=ud+4*vol;
   rs=amalloc(sizeof(*rs),5);

   if ((*b).nbp)
   {
      ibp=(*b).ibp;
      ibm=ibp+(*b).nbp/2;

      for (;ibp<ibm;ibp++)
         s[*ibp]=sd0;

      ibu=((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0)&&(bc_type()==0));
      ibd=((cpr[0]==0)&&((*b).bo[0]==0));

      for (;ud<um;ud+=8)
      {
         if (((pidn[0]<vol)||(!ibd))&&((piup[0]<vol)||(!ibu)))
         {
            doe(-0.5,piup,pidn,ud,s,rs);

            mul_pauli_dble(0.0f,swd,(*rs).w,(*rs).w);
            mul_pauli_dble(0.0f,swd+1,(*rs).w+1,(*rs).w+1);

            deo(0.5,piup,pidn,ud,rs,r);
         }

         piup+=4;
         pidn+=4;
         swd+=2;
      }

      ibp=(*b).ibp;
      ibm=ibp+(*b).nbp/2;

      for (;ibp<ibm;ibp++)
         r[*ibp]=sd0;
   }
   else
   {
      for (;ud<um;ud+=8)
      {
         doe(-0.5,piup,pidn,ud,s,rs);

         mul_pauli_dble(0.0f,swd,(*rs).w,(*rs).w);
         mul_pauli_dble(0.0f,swd+1,(*rs).w+1,(*rs).w+1);

         deo(0.5,piup,pidn,ud,rs,r);

         piup+=4;
         pidn+=4;
         swd+=2;
      }
   }

   afree(rs);
}
