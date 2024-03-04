
/*******************************************************************************
*
* File linsolv.h
*
* Copyright (C) 2011, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef LINSOLV_H
#define LINSOLV_H

#ifndef SU3_H
#include "su3.h"
#endif

/* CGNE_C */
extern double cgne(int vol,void (*Dop)(spinor *s,spinor *r),
                   void (*Dop_dble)(spinor_dble *s,spinor_dble *r),
                   spinor **ws,spinor_dble **wsd,int nmx,int istop,double res,
                   spinor_dble *eta,spinor_dble *psi,int *status);

/* FGCR4VD_C */
extern double fgcr4vd(int vol,void (*Dop)(complex_dble *v,complex_dble *w),
                      void (*Mop)(complex_dble *v,complex_dble *w,
                                  complex_dble *z),
                      complex_dble **wvd,int nkv,int nmx,double res,
                      complex_dble *eta,complex_dble *psi,int *status);

/* FGCR_C */
extern double fgcr(int vol,void (*Dop)(spinor_dble *s,spinor_dble *r),
                   void (*Mop)(int k,spinor *rho,spinor *phi,spinor *chi),
                   spinor **ws,spinor_dble **wsd,int nkv,int nmx,int istop,
                   double res,spinor_dble *eta,spinor_dble *psi,int *status);

/* GCR4V_C */
extern float gcr4v(int vol,void (*Dop)(complex *v,complex *w),complex **wv,
                   int nmx,float res,complex *eta,complex *psi,int *status);

/* MSCG_C */
extern void mscg(int vol,int nmu,double *mu,
                 void (*Dop_dble)(double mu,spinor_dble *s,spinor_dble *r),
                 spinor_dble **wsd,int nmx,int istop,double *res,
                 spinor_dble *eta,spinor_dble **psi,int *status);

#endif
