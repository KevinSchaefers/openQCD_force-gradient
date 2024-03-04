
/*******************************************************************************
*
* File dft.h
*
* Copyright (C) 2015 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef DFT_H
#define DFT_H

#ifndef FLAGS_H
#include "flags.h"
#endif

typedef struct
{
   int nwf,nekx,nfs,nbuf;
   complex_dble *wf,*ekx;
   complex_dble **fs,**fts,*buf;
} dft_wsp_t;

/* DFT4D_C */
extern void dft4d(int id,complex_dble *f,complex_dble *ft);
extern void inv_dft4d(int id,complex_dble *ft,complex_dble *f);

/* DFT_COM_C */
extern void dft_gather(int mu,int *nx,int *mf,complex_dble **lf,
                       complex_dble **f);
extern void dft_scatter(int mu,int *nx,int *mf,complex_dble **f,
                        complex_dble **lf);

/* DFT_SHUF_C */
extern void dft_shuf(int nx,int ny,int csize,complex_dble *f,complex_dble *rf);

/* DFT_WSPACE_C */
extern dft_wsp_t *alloc_dft_wsp(void);
extern void free_dft_wsp(dft_wsp_t *dwsp);
extern complex_dble *set_dft_wsp0(int n,dft_wsp_t *dwsp);
extern complex_dble **set_dft_wsp1(int n,dft_wsp_t *dwsp);
extern complex_dble *set_dft_wsp2(int n,dft_wsp_t *dwsp);

/* FFT_C */
extern int fft(dft_parms_t *dp,int nf,complex_dble **f,complex_dble **ft);
extern int inv_fft(dft_parms_t *dp,int nf,complex_dble **f,complex_dble **ft);

/* SMALL_DFT_C */
extern int small_dft(int s,int n,int nf,complex_dble *w,complex_dble **f,
                     dft_wsp_t *dwsp,complex_dble **ft);

#endif
