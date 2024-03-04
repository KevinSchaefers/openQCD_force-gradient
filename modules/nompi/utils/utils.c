
/*******************************************************************************
*
* File utils.c
*
* Copyright (C) 2005, 2008, 2011, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)

* Collection of basic utility programs.
*
*   int safe_mod(int x,int y)
*     Returns x mod y, where y is assumed positive and x can have any
*     sign. The return value is in the interval [0,y).
*
*   void *amalloc(size_t size,int p)
*     Allocates an aligned memory area of "size" bytes, with starting
*     address (the return value) that is an integer multiple of 2^p.
*
*   void afree(void *addr)
*     Frees the aligned memory area at address "addr" that was
*     previously allocated using amalloc.
*
*   void error(int test,int no,char *name,char *format,...)
*     Checks whether "test"=0 and if not aborts the program gracefully
*     with error number "no" after printing the "name" of the calling
*     program and an error message to stdout. The message is formed using
*     the "format" string and any additional arguments, exactly as in a
*     printf statement.
*
*   void error_root(int test,int no,char *name,char *format,...)
*     Same as error(), provided for compatibility.
*
*   void error_loc(int test,int no,char *name,char *format,...)
*     Same as error(), provided for compatibility.
*
*   void message(char *format,...)
*     Same as printf(), provided for compatibility.
*
*******************************************************************************/

#define UTILS_C

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include "utils.h"


int safe_mod(int x,int y)
{
   if (x>=0)
      return(x%y);
   else
      return((y-(abs(x)%y))%y);
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


void error(int test,int no,char *name,char *format,...)
{
   va_list args;

   if (test!=0)
   {
      printf("\nError in %s:\n",name);
      va_start(args,format);
      vprintf(format,args);
      va_end(args);
      printf("\nProgram aborted\n\n");
      exit(no);
   }
}


void error_root(int test,int no,char *name,char *format,...)
{
   va_list args;

   if (test!=0)
   {
      printf("\nError in %s:\n",name);
      va_start(args,format);
      vprintf(format,args);
      va_end(args);
      printf("\nProgram aborted\n\n");
      exit(no);
   }
}


void error_loc(int test,int no,char *name,char *format,...)
{
   va_list args;

   if (test!=0)
   {
      printf("\nError in %s:\n",name);
      va_start(args,format);
      vprintf(format,args);
      va_end(args);
      printf("\nProgram aborted\n\n");
      exit(no);
   }
}


void message(char *format,...)
{
   va_list args;

   va_start(args,format);
   vprintf(format,args);
   va_end(args);
}
