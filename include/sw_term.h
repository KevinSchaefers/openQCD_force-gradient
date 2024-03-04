
/*******************************************************************************
*
* File sw_term.h
*
* Copyright (C) 2005, 2009, 2011, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef SW_TERM_H
#define SW_TERM_H

#ifndef SU3_H
#include "su3.h"
#endif

#ifndef UTILS_H
#include "utils.h"
#endif

typedef struct
{
   double *rr;
   complex_dble *aa,*cc,*dd;
} pauli_wsp_t;

typedef struct
{
   int N;
   double **csv,**ksv,**qsv;
   double *psv;
   weyl_dble *wsv;
   pauli_dble *Asv;
} swexp_wsp_t;

/* PAULI_C */
extern void mul_pauli(float mu,pauli *m,weyl *s,weyl *r);
extern void mul_pauli2(float mu,pauli *m,spinor *s,spinor *r);
extern void assign_pauli(int vol,pauli_dble *md,pauli *m);
extern void apply_sw(int vol,float mu,pauli *m,spinor *s,spinor *r);

/* PAULI_DBLE_C */
extern void mul_pauli_dble(double mu,pauli_dble *m,weyl_dble *s,weyl_dble *r);
extern pauli_wsp_t *alloc_pauli_wsp(void);
extern void free_pauli_wsp(pauli_wsp_t *pwsp);
extern int inv_pauli_dble(double mu,pauli_dble *m,pauli_wsp_t *pwsp,
                          pauli_dble *im);
extern complex_dble det_pauli_dble(double mu,pauli_dble *m,pauli_wsp_t *pwsp);
extern void apply_sw_dble(int vol,double mu,pauli_dble *m,
                          spinor_dble *s,spinor_dble *r);
extern int apply_swinv_dble(int vol,double mu,pauli_dble *m,spinor_dble *s,
                            pauli_wsp_t *pwsp,spinor_dble *r);

/* SWALG_C */
extern void pauli2weyl(pauli_dble *A,weyl_dble *w);
extern void weyl2pauli(weyl_dble *w,pauli_dble *A);
extern void prod_pauli_mat(pauli_dble *A,weyl_dble *w,weyl_dble *v);
extern void add_pauli_mat(pauli_dble *A,pauli_dble *B,pauli_dble *C);
extern void lc3_pauli_mat(double *c,pauli_dble *A1,pauli_dble *A2,
                          pauli_dble *B);
extern double tr0_pauli_mat(pauli_dble *A);
extern double tr1_pauli_mat(pauli_dble *A,pauli_dble *B);

/* SWEXP_C */
extern swexp_wsp_t *alloc_swexp_wsp(int N);
extern void free_swexp_wsp(swexp_wsp_t *swsp);
extern void sw_exp(int s,pauli_dble *A,double r,swexp_wsp_t *swsp,
                   pauli_dble *B);
extern void sw_dexp(pauli_dble *A,double r,swexp_wsp_t *swsp,double *q);

/* SWFLDS_C */
extern pauli *swfld(void);
extern pauli_dble *swdfld(void);
extern void assign_swd2sw(void);

/* SW_TERM_C */
extern int sw_order(void);
extern void pauli_term(double c,u3_alg_dble **ft,pauli_dble *m);
extern int sw_term(ptset_t set);

#endif
