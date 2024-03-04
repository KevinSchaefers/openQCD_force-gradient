
/*******************************************************************************
*
* File little.h
*
* Copyright (C) 2011, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef LITTLE_H
#define LITTLE_H

#ifndef SU3_H
#include "su3.h"
#endif

#ifndef BLOCK_H
#include "block.h"
#endif

typedef struct
{
   int Ns,nb;
   complex **Ablk,**Ahop;
} Aw_t;

typedef struct
{
   int Ns,nb;
   complex_dble **Ablk,**Ahop;
} Aw_dble_t;

/* AW_COM1_C */
extern void cpse_int_bnd(int ieo,int ifc,spinor *flds);

/* AW_COM2_C */
extern void cpAhop_ext_bnd(int ieo,int ifc,complex_dble **Amats);

/* AW_C */
extern void Aw(complex *v,complex *w);
extern void Aweeinv(complex *v,complex *w);
extern void Awooinv(complex *v,complex *w);
extern void Awoe(complex *v,complex *w);
extern void Aweo(complex *v,complex *w);
extern void Awhat(complex *v,complex *w);

/* AW_DBLE_C */
extern void Aw_dble(complex_dble *v,complex_dble *w);
extern void Aweeinv_dble(complex_dble *v,complex_dble *w);
extern void Awooinv_dble(complex_dble *v,complex_dble *w);
extern void Awoe_dble(complex_dble *v,complex_dble *w);
extern void Aweo_dble(complex_dble *v,complex_dble *w);
extern void Awhat_dble(complex_dble *v,complex_dble *w);

/* AW_GEN_C */
extern void gather_ud(int ifc,su3_dble *udb,block_t *b);
extern void gather_se(int n,int ifc,block_t *b,spinor *sbuf);
extern void gather_so(int n,int ifc,block_t *b,spinor_dble *sbuf);
extern void (*spinor_prod_gamma[])(int vol,spinor_dble *sd,spinor_dble *rd,
                                   complex_dble *sp);

/* AW_BLK_C */
#if ((defined AW_BLK_C)||(defined AW_OPS_C))
extern void set_Awblk(double mu);
extern int update_Awblk(double mu);
#endif

/* AW_HOP_C */
#if ((defined AW_HOP_C)||(defined AW_OPS_C))
extern void set_Awhop(void);
#endif

/* AW_OPS_C */
extern Aw_t Awop(void);
extern Aw_t Awophat(void);
extern Aw_dble_t Awop_dble(void);
extern Aw_dble_t Awophat_dble(void);
extern void set_Aw(double mu);
extern int set_Awhat(double mu);

/* LTL_MODES_C */
extern int set_ltl_modes(void);
extern complex_dble *ltl_matrix(void);
extern void dfl_Lvd(complex_dble *vd);
extern void dfl_LRvd(complex_dble *vd,complex_dble *wd);
extern void dfl_RLvd(complex_dble *vd,complex_dble *wd);
extern void dfl_Lv(complex *v);

#endif
