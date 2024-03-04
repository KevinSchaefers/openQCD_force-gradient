
/*******************************************************************************
*
* File nompi/utils.h
*
* Copyright (C) 2009-2011, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef UTILS_H
#define UTILS_H

#include <limits.h>
#include <float.h>

#define NAME_SIZE 128

#if (SHRT_MAX==0x7fffffff)
typedef short int stdint_t;
typedef unsigned short int stduint_t;
#elif (INT_MAX==0x7fffffff)
typedef int stdint_t;
typedef unsigned int stduint_t;
#elif (LONG_MAX==0x7fffffff)
typedef long int stdint_t;
typedef unsigned long int stduint_t;
#else
#error : There is no four-byte integer type on this machine
#endif

#if (SHRT_MAX==0x7fffffffffffffff)
typedef short int stdlong_t;
typedef unsigned short int stdulong_t;
#elif (INT_MAX==0x7fffffffffffffff)
typedef int stdlong_t;
typedef unsigned int stdulong_t;
#elif (LONG_MAX==0x7fffffffffffffff)
typedef long int stdlong_t;
typedef unsigned long int stdulong_t;
#else
#error : There is no eight-byte integer type on this machine
#endif

#undef UNKNOWN_ENDIAN
#undef LITTLE_ENDIAN
#undef BIG_ENDIAN

#define UNKNOWN_ENDIAN 0
#define LITTLE_ENDIAN 1
#define BIG_ENDIAN 2

#undef IMAX
#define IMAX(n,m) ((n)+((m)-(n))*((m)>(n)))

typedef enum
{
   ALL_PTS,EVEN_PTS,ODD_PTS,NO_PTS,PT_SETS
} ptset_t;

typedef struct
{
   unsigned int d,p;
   size_t size,*n;
   void *a;
} array_t;

/* ARRAY */
extern array_t *alloc_array(unsigned int d,size_t *n,size_t size,
                            unsigned int p);
extern void free_array(array_t *a);
extern void write_array(FILE *fdat,array_t *a);
extern void read_array(FILE *fdat,array_t *a);

/* ENDIAN_C */
extern int endianness(void);
extern void bswap_int(int n,void *a);
extern void bswap_double(int n,void *a);

/* ERROR_C */
extern void set_error_file(char *path,int loc_flag);
extern void error(int test,int no,char *name,char *format,...);
extern void error_root(int test,int no,char *name,char *format,...);
extern void error_loc(int test,int no,char *name,char *format,...);

/* MUTILS_C */
extern void check_machine(void);
extern int find_opt(int argc,char *argv[],char *opt);
extern int digits(double x,double dx,char *fmt);
extern int fdigits(double x);
extern int name_size(char *format,...);
extern long find_section(FILE *stream,char *title);
extern long read_line(FILE *stream,char *tag,char *format,...);
extern int count_tokens(FILE *stream,char *tag);
extern void read_iprms(FILE *stream,char *tag,int n,int *iprms);
extern void read_dprms(FILE *stream,char *tag,int n,double *dprms);

/* PARMS_C */
extern int write_parms(FILE *fdat,int n,int *i,int m,double *r);
extern int read_parms(FILE *fdat,int *n,int **i,int *m,double **r);
extern int check_parms(FILE *fdat,int n,int *i,int m,double *r);

/* QSUM_C */
extern void acc_qflt(double u,double *qr);
extern void add_qflt(double *qu,double *qv,double *qr);
extern void scl_qflt(double u,double *qr);
extern void mul_qflt(double *qu,double *qv,double *qr);

/* UTILS_C */
extern int safe_mod(int x,int y);
extern void *amalloc(size_t size,int p);
extern void afree(void *addr);
extern void message(char *format,...);

#endif
