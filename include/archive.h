
/*******************************************************************************
*
* File archive.h
*
* Copyright (C) 2011, 2017, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef ARCHIVE_H
#define ARCHIVE_H

#ifndef SU3_H
#include "su3.h"
#endif

typedef struct
{
   int types;
   int nb,ib,bs[4];
   int nio_nodes,nio_streams;
   char *cnfg_dir,*block_dir,*local_dir;
} iodat_t;


/* ARCHIVE_C */
extern void set_nio_streams(int nio);
extern int nio_streams(void);
extern void write_cnfg(char *out);
extern void read_cnfg(char *in);
extern void lat_sizes(char *in,int *nl);
extern void export_cnfg(char *out);
extern void import_cnfg(char *in,int mask);
extern void blk_sizes(char *in,int *nl,int *bs);
extern int blk_index(int *nl,int *bs,int *nb);
extern int blk_root_process(int *nl,int *bs,int *bo,int *nb,int *ib);
extern void blk_export_cnfg(int *bs,char *out);
extern void blk_import_cnfg(char *in,int mask);

/* IODAT_C */
extern void read_iodat(char *section,char *io,iodat_t *iodat);
extern void print_iodat(char *io,iodat_t *iodat);
extern void read_flds(iodat_t *iodat,char *cnfg,int mask,int ioflg);
extern void save_flds(int icnfg,char *nbase,iodat_t *iodat,int ioflg);
extern void remove_flds(int icnfg,char *nbase,iodat_t *iodat,int ioflg);
extern void check_iodat(iodat_t *iodat,char *io,int ioflg,char *cnfg);
extern void write_iodat_parms(FILE *fdat,iodat_t *iodat);
extern void check_iodat_parms(FILE *fdat,iodat_t *iodat);

/* MARCHIVE_C */
extern void write_mfld(char *out);
extern void read_mfld(char *in);
extern void export_mfld(char *out);
extern void import_mfld(char *in);
extern void blk_export_mfld(int *bs,char *out);
extern void blk_import_mfld(char *in);

/* SARCHIVE_C */
extern void write_sfld(char *out,int eo,spinor_dble *sd);
extern void read_sfld(char *in,int eo,spinor_dble *sd);
extern void export_sfld(char *out,int eo,spinor_dble *sd);
extern void import_sfld(char *in,int eo,spinor_dble *sd);
extern void blk_export_sfld(int *bs,char *out,int eo,spinor_dble *sd);
extern void blk_import_sfld(char *in,int eo,spinor_dble *sd);

#endif
