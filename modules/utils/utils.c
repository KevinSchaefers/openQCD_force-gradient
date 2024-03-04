
/*******************************************************************************
*
* File utils.c
*
* Copyright (C) 2005, 2008, 2011, 2016, 2019, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Collection of basic utility programs.
*
*   int safe_mod(int x,int y)
*     Returns x mod y, where y is assumed positive and x can have any
*     sign. The returned value is in the interval [0,y).
*
*   void divide_range(int n,int m,int *a,int *b)
*     Divides the range 0,..,n-1 into m consecutive ranges a[k],..,b[k]-1,
*     k=0,..,m-1, with sizes b[k]-a[k] equal to n/m+1 if k<n%m and n/m
*     if k>=n%m. No operations are performed if n<0 or m<1.
*
*   void *amalloc(size_t size,int p)
*     Allocates an aligned memory area of "size" bytes, with a starting
*     address (the return value) that is an integer multiple of 2^p. A
*     NULL pointer is returned if the allocation was not successful.
*
*   void afree(void *addr)
*     Frees the aligned memory area at address "addr" that was previously
*     allocated using amalloc. If the memory space at this address was
*     already freed using afree, or if the address does not match an
*     address previously returned by amalloc, the program does not do
*     anything.
*
*   int mpi_permanent_tag(void)
*     Returns a new send tag that is guaranteed to be unique and which
*     is therefore suitable for use in permanent communication requests.
*     The available number of tags of this kind is 16384.
*
*   int mpi_tag(void)
*     Returns a new send tag for use in non-permanent communications.
*     Note that the counter for these tags wraps around after 16384
*     tags have been delivered.
*
*   void message(char *format,...)
*     Prints a message from process 0 to stdout. The usage and argument
*     list is the same as in the case of the printf function.
*
*   void mpi_init(int argc,char **argv)
*     In MPI and hybrid MPI/OpenMP main programs, this function may be
*     used as a replacement of MPI_Init(&argc,&argv). In the hybrid case,
*     MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,..) is called and
*     the availability of the thread-safety level MPI_THREAD_FUNNELED is
*     checked (an error occurs if not).
*
* The programs safe_mod(), amalloc() and afree() are thread-safe. All
* other programs are assumed to be called by the OpenMP master thread.
*
*******************************************************************************/

#define UTILS_C

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#if (defined _OPENMP)
#include <omp.h>
#endif
#include "mpi.h"
#include "utils.h"
#include "global.h"

#define MAX_TAG 32767
#define MAX_PERMANENT_TAG MAX_TAG/2

static int pcmn_cnt=-1,cmn_cnt=MAX_TAG;


int safe_mod(int x,int y)
{
   if (x>=0)
      return x%y;
   else
      return (y-(abs(x)%y))%y;
}


void divide_range(int n,int m,int *a,int *b)
{
   int k,l,d;

   if ((n>=0)&&(m>=1))
   {
      l=n%m;
      d=n/m;
      a[0]=0;
      b[0]=d+(l>0);

      for (k=1;k<m;k++)
      {
         a[k]=b[k-1];
         b[k]=a[k]+d+(l>k);
      }
   }
}


void *amalloc(size_t size,int p)
{
   char *true_addr,*addr;
   unsigned int shift,isize,*iaddr;
   stdulong_t mask;

   if ((size<=0)||(p<0))
      return(NULL);

   shift=(unsigned int)((1<<p)-1);
   mask=(stdulong_t)(shift);
   isize=(unsigned int)(sizeof(unsigned int));
   true_addr=malloc(size+shift+isize);

   if (true_addr==NULL)
      return(NULL);

   addr=(char*)(((stdulong_t)(true_addr+shift+isize))&(~mask));
   iaddr=(unsigned int*)(addr-isize);
   (*iaddr)=(unsigned int)(addr-true_addr);

   return (void*)(addr);
}


void afree(void *addr)
{
   char *true_addr;
   unsigned int isize,*iaddr;

   isize=(unsigned int)(sizeof(unsigned int));
   iaddr=(unsigned int*)((char*)(addr)-isize);
   true_addr=(char*)(addr)-(*iaddr);

   free(true_addr);
}


int mpi_permanent_tag(void)
{
   if (pcmn_cnt<MAX_PERMANENT_TAG)
      pcmn_cnt+=1;
   else
      error_loc(1,1,"mpi_permanent_tag [utils.c]",
                "Requested more than 16384 tags");

   return pcmn_cnt;
}


int mpi_tag(void)
{
   if (cmn_cnt==MAX_TAG)
      cmn_cnt=MAX_PERMANENT_TAG;

   cmn_cnt+=1;

   return cmn_cnt;
}

void message(char *format,...)
{
   int my_rank;
   va_list args;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      va_start(args,format);
      vprintf(format,args);
      va_end(args);
   }
}

#if (defined _OPENMP)

void mpi_init(int argc,char **argv)
{
   int provided;

   MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);

   error_root(provided<MPI_THREAD_FUNNELED,1,"mpi_init_thread [utils.c]",
              "Thread-safety level MPI_THREAD_FUNNELED is not supported");

   omp_set_num_threads(NTHREAD);
   omp_set_dynamic(0);
   omp_set_nested(0);
}

#else

void mpi_init(int argc,char **argv)
{
   MPI_Init(&argc,&argv);
}

#endif
