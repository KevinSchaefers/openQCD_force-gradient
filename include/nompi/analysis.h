
/*******************************************************************************
*
* File nompi/analysis.h
*
* Copyright (C) 2009, 2010, 2011, 2015 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef ANALYSIS_H
#define ANALYSIS_H

typedef struct
{
   int n;
   double *x,*y,*c;
} fit_t;

typedef struct jdat_t
{
  int n;
  double fbar,sigf;
  double *f;
  struct jdat_t *next;
} jdat_t;

/* BOOTSTRAP_C */
extern int sample_gauss(int n,double *avg,double *cov,int nms,int ndc,
                        double *ev,jdat_t **jdat);

/* JACK_C */
extern void check_jdat(int nobs,jdat_t **jdat,int flag);
extern void free_jdat(jdat_t *jdat);
extern void alloc_jdat(int n,int ndc,jdat_t *jdat);
extern void jackbin(int n,int bs,int ndc,double *a,jdat_t *jdat);
extern void newobs(int nobs,jdat_t **jdat,double (*f)(int nobs,double *obs),
                   jdat_t *jnew);
extern void jackcov(int nobs,jdat_t **jdat,double *cov);
extern double jackbias(jdat_t *jdat);
extern void fwrite_jdat(jdat_t *jdat,FILE *out);
extern void fread_jdat(jdat_t *jdat,FILE *in);
extern void copy_jdat(jdat_t *jdin,jdat_t *jdout);
extern void embed_jdat(int n,int offset,jdat_t *jdin,jdat_t *jdout);

/* JFIT_C */
extern void jfit_cov(int ndat,jdat_t **jdat,jdat_t *jcov);
extern double jfit_icov(int ndat,jdat_t **jdat,jdat_t *jcov);
extern double jfit_const(int type,int ndat,jdat_t **jdat,double *k,
                         jdat_t *jpar);
extern double jfit(int ndat,jdat_t **jdat,int npar,jdat_t **jpar,
                   double (*lhf)(int ndat,double *dat,int npar,double *par),
                   double *x0,double *x1,double *x2,
                   int imx,double omega1,double omega2,int *status);

/* LSQFIT_C */
extern double least_squares(fit_t dat,int m,int r,double (*f)(int i,double x),
                            double *w,double *s);
extern void fit_parms(fit_t dat,int m,double *s,double *val,double *cov);

/* POLYFIT_C */
extern double polyfit(fit_t dat,int pmin,int pmax,double *a,double *cov);

#endif
