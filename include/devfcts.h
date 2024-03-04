
/*******************************************************************************
*
* File devfcts.h
*
* Copyright (C) 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef DEVFCTS_H
#define DEVFCTS_H

#ifndef SU3_H
#include "su3.h"
#endif

/* GTRANS_C */
extern su3_dble *gtrans(void);
extern void random_gtrans(void);
extern void apply_gtrans2ud(void);

/* ROTUD_C */
extern int check_active(su3_alg_dble *fld);
extern void rot_ud(double eps);

/* DUTILS_C */
extern void random_shift(int *svec);

#endif
