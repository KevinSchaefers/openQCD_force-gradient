
/*******************************************************************************
*
* File global.h
*
* Copyright (C) 2009, 2011, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Global parameters and arrays
*
*******************************************************************************/

#ifndef GLOBAL_H
#define GLOBAL_H

#define NPROC0 1
#define NPROC1 1
#define NPROC2 1
#define NPROC3 1

#define NPROC0_BLK 1
#define NPROC1_BLK 1
#define NPROC2_BLK 1
#define NPROC3_BLK 1

#define L0 16
#define L1 8
#define L2 8
#define L3 8

#define L0_TRD 8
#define L1_TRD 8
#define L2_TRD 8
#define L3_TRD 8

#define NAME_SIZE 128

/****************************** do not change *********************************/

#if ((NPROC0<1)||(NPROC1<1)||(NPROC2<1)||(NPROC3<1)|| \
    ((NPROC0>1)&&((NPROC0%2)!=0))||((NPROC1>1)&&((NPROC1%2)!=0))|| \
    ((NPROC2>1)&&((NPROC2%2)!=0))||((NPROC3>1)&&((NPROC3%2)!=0)))
#error : The number of processes in each direction must be 1 or a multiple of 2
#endif

#if ((NPROC0_BLK<1)||(NPROC0_BLK>NPROC0)||((NPROC0%NPROC0_BLK)!=0)|| \
     (NPROC1_BLK<1)||(NPROC1_BLK>NPROC1)||((NPROC1%NPROC1_BLK)!=0)|| \
     (NPROC2_BLK<1)||(NPROC2_BLK>NPROC2)||((NPROC2%NPROC2_BLK)!=0)|| \
     (NPROC3_BLK<1)||(NPROC3_BLK>NPROC3)||((NPROC3%NPROC3_BLK)!=0))
#error : Improper processor block sizes NPROC0_BLK,..,NPROC3_BLK
#endif

#if ((L0_TRD<4)||((L0_TRD%2)!=0)||(L1_TRD<4)||((L1_TRD%2)!=0)|| \
     (L2_TRD<4)||((L2_TRD%2)!=0)||(L3_TRD<4)||((L3_TRD%2)!=0))
#error : The thread-local lattice sizes must be even and not smaller than 4
#endif

#if ((L0<1)||((L0%L0_TRD)!=0)||(L1<1)||((L1%L1_TRD)!=0)|| \
     (L2<1)||((L2%L2_TRD)!=0)||(L3<1)||((L3%L3_TRD)!=0))
#error : The thread-local lattice sizes must divide the local lattice sizes
#endif

#if (NAME_SIZE<128)
#error : NAME_SIZE must be greater or equal to 128
#endif

#define NPROC (NPROC0*NPROC1*NPROC2*NPROC3)

#define VOLUME (L0*L1*L2*L3)
#define FACE0 ((1-(NPROC0%2))*L1*L2*L3)
#define FACE1 ((1-(NPROC1%2))*L2*L3*L0)
#define FACE2 ((1-(NPROC2%2))*L3*L0*L1)
#define FACE3 ((1-(NPROC3%2))*L0*L1*L2)
#define BNDRY (2*(FACE0+FACE1+FACE2+FACE3))
#define NSPIN (VOLUME+(BNDRY/2))

#define VOLUME_TRD (L0_TRD*L1_TRD*L2_TRD*L3_TRD)
#define NTHREAD (VOLUME/VOLUME_TRD)

#define ALIGN 6

#if defined MAIN_PROGRAM
int cpr[4];
int npr[8];
int sbofs[16];
int sbvol[16];

int *ipt=NULL;
int (*iup)[4]=NULL;
int (*idn)[4]=NULL;
int *map=NULL;
#else
extern int cpr[4];
extern int npr[8];
extern int sbofs[16];
extern int sbvol[16];

extern int *ipt;
extern int (*iup)[4];
extern int (*idn)[4];
extern int *map;
#endif

#endif
