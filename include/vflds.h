
/*******************************************************************************
*
* File vflds.h
*
* Copyright (C) 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef VFLDS_H
#define VFLDS_H

#ifndef SU3_H
#include "su3.h"
#endif

/* VCOM_C */
extern void cpv_int_bnd(int ieo,complex *v);

/* VDCOM_C */
extern void cpvd_int_bnd(int ieo,complex_dble *vd);

/* VFLDS_C */
extern complex **vflds(void);
extern complex_dble **vdflds(void);

/* VINIT_C */
extern void set_v2zero(int n,int icom,complex *v);
extern void set_vd2zero(int n,int icom,complex_dble *vd);
extern void random_v(int n,int icom,complex *v,float sigma);
extern void random_vd(int n,int icom,complex_dble *vd,double sigma);
extern void assign_v2v(int n,int icom,complex *v,complex *w);
extern void assign_v2vd(int n,int icom,complex *v,complex_dble *wd);
extern void assign_vd2v(int n,int icom,complex_dble *vd,complex *w);
extern void assign_vd2vd(int n,int icom,complex_dble *vd,complex_dble *wd);
extern void add_vd2vd(int n,int icom,complex_dble *vd,complex_dble *wd);
extern void add_v2vd(int n,int icom,complex *v,complex_dble *wd);
extern void diff_vd2vd(int n,int icom,complex_dble *vd,complex_dble *wd);
extern void diff_vd2v(int n,int icom,complex_dble *vd,complex_dble *wd,
                      complex *w);
extern void diff_v2v(int n,int icom,complex *v,complex *w);

#endif
