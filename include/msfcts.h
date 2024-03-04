
/*******************************************************************************
*
* File msfcts.h
*
* Copyright (C) 2017, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef MSFCTS_H
#define MSFCTS_H

#ifndef SU3_H
#include "su3.h"
#endif

#ifndef ARCHIVE_H
#include "archive.h"
#endif

/* FARCHIVE */
extern void write_msfld(char *out,int n,double **f);
extern void read_msfld(char *in,int n,double **f);
extern void export_msfld(char *out,int n,double **f);
extern void import_msfld(char *in,int n,double **f);
extern void blk_export_msfld(int *bs,char *out,int n,double **f);
extern void blk_import_msfld(char *in,int n,double **f);

/* FIODAT_C */
extern void read_msflds(iodat_t *iodat,char *cnfg,int n,double **f);
extern void save_msflds(int icnfg,char *nbase,iodat_t *iodat,int n,double **f);

/* FLDOPS_C */
extern void gather_msfld(double *f,complex_dble *rf);
extern void scatter_msfld(complex_dble *rf,double *f);
extern void apply_fft(int type,complex_dble *rf,complex_dble *rft);
extern void convolute_msfld(int *s,double *f,double *g,complex_dble *rf,
                            complex_dble *rg,double *fg);
extern void shift_msfld(int *s,double *f,complex_dble *rf,double *g);
extern void copy_msfld(double *f,double *g);
extern void add_msfld(double *f,double *g);
extern void mul_msfld(double *f,double *g);
extern void mulr_msfld(double r,double *f);

/* LATAVG_C */
extern void sphere_msfld(int r,double *f);
extern void sphere3d_msfld(int r,double *f);
extern void sphere_sum(int dmax,double *f,double *sm);
extern void sphere3d_sum(int dmax,double *f,double *sm);
extern double avg_msfld(double *f);
extern double center_msfld(double *f);
extern void cov_msfld(int dmax,double *f,double *g,complex_dble *rf,
                      complex_dble *rg,double *w,double *cov);
extern void cov3d_msfld(int dmax,double *f,double *g,complex_dble *rf,
                        complex_dble *rg,double *w,double *cov);

/* PT2PT_C */
extern void pt2pt(int i3d,int r,double *f,complex_dble *rf,
                  double *w1,double *w2,double *obs);

#endif
