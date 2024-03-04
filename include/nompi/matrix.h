
/*******************************************************************************
*
* File nompi/matrix.h
*
* Copyright (C) 2009, 2010, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef MATRIX_H
#define MATRIX_H

/* HOUSE_C */
extern double house(int n,double *amat);

/* JACOBI_C */
extern void jacobi(int n,double *a,double *d,double *v);

/* MATRIX_C */
extern void mat_vec(int n,double *a,double *v,double *w);
extern void mat_add(int n,double *a,double *b,double *c);
extern void mat_sub(int n,double *a,double *b,double *c);
extern void mat_mulr(int n,double r,double a[]);
extern void mat_mulr_add(int n,double r,double a[],double b[]);
extern void mat_mul(int n,double *a,double *b,double *c);
extern void mat_mul_tr(int n,double *a,double *b,double *c);
extern void mat_sym(int n,double *a);
extern void mat_inv(int n,double *a,double *ainv,double *k);
extern double mat_trace(int n,double *a);

#endif
