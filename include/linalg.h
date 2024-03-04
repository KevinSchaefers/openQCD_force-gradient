
/*******************************************************************************
*
* File linalg.h
*
* Copyright (C) 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef LINALG_H
#define LINALG_H

#ifndef SU3_H
#include "su3.h"
#endif

#ifndef UTILS_H
#include "utils.h"
#endif

typedef struct
{
   int nmx;
   double *r;
   complex_dble *d;
} cmat_wsp_t;

/* CMATRIX_C */
extern void cmat_vec(int n,complex *a,complex *v,complex *w);
extern void cmat_vec_assign(int n,complex *a,complex *v,complex *w);
extern void cmat_add(int n,complex *a,complex *b,complex *c);
extern void cmat_sub(int n,complex *a,complex *b,complex *c);
extern void cmat_mul(int n,complex *a,complex *b,complex *c);
extern void cmat_dag(int n,complex *a,complex *b);

/* CMATRIX_DBLE_C */
extern void cmat_vec_dble(int n,complex_dble *a,complex_dble *v,
                          complex_dble *w);
extern void cmat_vec_assign_dble(int n,complex_dble *a,complex_dble *v,
                                 complex_dble *w);
extern void cmat_add_dble(int n,complex_dble *a,complex_dble *b,
                          complex_dble *c);
extern void cmat_sub_dble(int n,complex_dble *a,complex_dble *b,
                          complex_dble *c);
extern void cmat_mul_dble(int n,complex_dble *a,complex_dble *b,
                          complex_dble *c);
extern void cmat_dag_dble(int n,complex_dble *a,complex_dble *b);
extern cmat_wsp_t *alloc_cmat_wsp(int nmx);
extern void free_cmat_wsp(cmat_wsp_t *cwsp);
extern int cmat_inv_dble(int n,complex_dble *a,cmat_wsp_t *cswp,
                         complex_dble *b,double *k);

/* LIEALG_C */
extern void random_alg(int vol,int icom,su3_alg_dble *X);
extern qflt norm_square_alg(int vol,int icom,su3_alg_dble *X);
extern qflt scalar_prod_alg(int vol,int icom,su3_alg_dble *X,su3_alg_dble *Y);
extern double unorm_alg(int vol,int icom,su3_alg_dble *X);
extern void set_alg2zero(int vol,int icom,su3_alg_dble *X);
extern void set_ualg2zero(int vol,int icom,u3_alg_dble *X);
extern void assign_alg2alg(int vol,int icom,su3_alg_dble *X,su3_alg_dble *Y);
extern void flip_assign_alg2alg(int vol,int icom,su3_alg_dble *X,
                                su3_alg_dble *Y);
extern void muladd_assign_alg(int vol,int icom,double r,su3_alg_dble *X,
                              su3_alg_dble *Y);

/* SALG_C */
extern complex spinor_prod(int vol,int icom,spinor *s,spinor *r);
extern float spinor_prod_re(int vol,int icom,spinor *s,spinor *r);
extern float norm_square(int vol,int icom,spinor *s);
extern void mulc_spinor_add(int vol,int icom,spinor *s,spinor *r,complex z);
extern void mulr_spinor_add(int vol,int icom,spinor *s,spinor *r,float c);
extern void scale(int vol,int icom,float c,spinor *s);
extern void project(int vol,int icom,spinor *s,spinor *r);
extern float normalize(int vol,int icom,spinor *s);
extern void mulg5(int vol,int icom,spinor *s);
extern void mulmg5(int vol,int icom,spinor *s);

/* SALG_DBLE_C */
extern complex_qflt spinor_prod_dble(int vol,int icom,spinor_dble *s,
                                     spinor_dble *r);
extern qflt spinor_prod_re_dble(int vol,int icom,spinor_dble *s,
                                spinor_dble *r);
extern complex_qflt spinor_prod5_dble(int vol,int icom,spinor_dble *s,
                                      spinor_dble *r);
extern qflt norm_square_dble(int vol,int icom,spinor_dble *s);
extern void mulc_spinor_add_dble(int vol,int icom,spinor_dble *s,
                                 spinor_dble *r,complex_dble z);
extern void mulr_spinor_add_dble(int vol,int icom,spinor_dble *s,
                                 spinor_dble *r,double c);
extern void combine_spinor_dble(int vol,int icom,spinor_dble *s,
                                spinor_dble *r,double cs,double cr);
extern void scale_dble(int vol,int icom,double c,spinor_dble *s);
extern void project_dble(int vol,int icom,spinor_dble *s,spinor_dble *r);
extern double normalize_dble(int vol,int icom,spinor_dble *s);
extern void mulg5_dble(int vol,int icom,spinor_dble *s);
extern void mulmg5_dble(int vol,int icom,spinor_dble *s);

/* VALG_C */
extern complex_dble vprod(int n,int icom,complex *v,complex *w);
extern double vnorm_square(int n,int icom,complex *v);
extern void mulc_vadd(int n,int icom,complex *v,complex *w,complex z);
extern void vscale(int n,int icom,float r,complex *v);
extern void vproject(int n,int icom,complex *v,complex *w);
extern double vnormalize(int n,int icom,complex *v);

/* VALG_DBLE_C */
extern complex_qflt vprod_dble(int n,int icom,complex_dble *v,complex_dble *w);
extern qflt vnorm_square_dble(int n,int icom,complex_dble *v);
extern void mulc_vadd_dble(int n,int icom,complex_dble *v,complex_dble *w,
                           complex_dble z);
extern void vscale_dble(int n,int icom,double r,complex_dble *v);
extern void vproject_dble(int n,int icom,complex_dble *v,complex_dble *w);
extern double vnormalize_dble(int n,int icom,complex_dble *v);

#endif
