
/*******************************************************************************
*
* File error.c
*
* Copyright (C) 2015, 2019 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Error handling functions.
*
*   void set_error_file(char *path,int loc_flag)
*     Sets the path of the file to which error messages are written. If
*     loc_flag!=0, the number of the local MPI process is appended to the
*     file name when the error_loc() function writes a message. Otherwise
*     the same file is used for all error messages.
*
*   void error(int test,int no,char *name,char *format,...)
*     Checks whether "test"=0 on all processes and, if not, aborts the
*     program gracefully with error number "no" after printing the "name"
*     of the calling program and an error message to the error file from
*     process 0. The message is formed using the "format" string and any
*     additional arguments, exactly as in a printf() statement.
*
*   void error_root(int test,int no,char *name,char *format,...)
*     Same as the error() function except that "test" is examined on
*     process 0 only.
*
*   void error_loc(int test,int no,char *name,char *format,...)
*     Same as the error() function except that "test" is examined only on
*     the calling MPI processes. Each process, where "test"!=0, prints an
*     error message to stdout and aborts the program.
*
* When the error file is not or not yet set, all error messages are written
* to stdout. The error file name need not be the same on all MPI processes
* and it may be reset at any given time. The error file used by the error()
* function is the one specified on the root process.
*
* It is permissible to direct the error messages to a file that is used for
* other purposes as well, as long as the file is not open for reading when
* an error occurs (it may, however, be open in "w" or "a" mode).
*
* If an error is detected and the macro IGNORE_ERRORS is not defined, the
* programs in this module abort the MPI program by calling MPI_Abort().
*
* The programs in this module are assumed to be executed by the OpenMP master
* thread. Except for the program error(), which must be called on all MPI
* processes simultaneously, the programs do not perform any communications
* and can be locally called.
*
*******************************************************************************/

#define ERROR_C

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include "mpi.h"
#include "utils.h"
#include "global.h"

static int iset=0,iloc=0;
static char fname[NAME_SIZE+1],fname_loc[NAME_SIZE];
static char inum[3*sizeof(int)];


static void wait(int s)
{
   time_t t0,t1;
   double dt;

   t0=time(NULL);

   if (t0==(time_t)(-1))
      return;

   dt=0.0;

   while (s>(int)(dt))
   {
      t1=time(NULL);
      dt=difftime(t1,t0);
   }
}


void set_error_file(char *path,int loc_flag)
{
   int my_rank,nlen;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   strncpy(fname,path,NAME_SIZE);
   fname[NAME_SIZE]='\0';
   nlen=strlen(fname);

   if (loc_flag)
   {
      sprintf(inum,"%d",my_rank);
      nlen+=strlen(inum)+1;
   }

   error_loc(nlen>=NAME_SIZE,1,"set_error_file [error.c]",
             "File name is too long");

   if (loc_flag)
   {
      sprintf(fname_loc,"%s_%d",path,my_rank);
      iloc=1;
   }

   iset=1;
}


void error(int test,int no,char *name,char *format,...)
{
   int i,all,my_rank;
   va_list args;
   FILE *ferr;

   i=(test!=0);
   MPI_Allreduce(&i,&all,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);

   if (all==0)
      return;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      if (iset)
      {
         ferr=fopen(fname,"a");
         fflush(NULL);
         fprintf(ferr,"\nError in %s (error no=%d):\n",name,no);
         va_start(args,format);
         vfprintf(ferr,format,args);
         va_end(args);
#ifndef IGNORE_ERRORS
         fprintf(ferr,"\nProgram aborted\n");
#else
         fprintf(ferr,"\nProgram execution continues (IGNORE_ERRORS is set)\n");
#endif
         fclose(ferr);
      }
      else
      {
         fflush(NULL);
         printf("\nError in %s (error no=%d):\n",name,no);
         va_start(args,format);
         vprintf(format,args);
         va_end(args);
#ifndef IGNORE_ERRORS
         printf("\nProgram aborted\n");
#else
         printf("\nProgram execution continues (IGNORE_ERRORS is set)\n");
#endif
      }

      fflush(NULL);
      wait(1);
#ifndef IGNORE_ERRORS
      MPI_Abort(MPI_COMM_WORLD,no);
#endif
   }

   MPI_Barrier(MPI_COMM_WORLD);
}


void error_root(int test,int no,char *name,char *format,...)
{
   int my_rank;
   va_list args;
   FILE *ferr;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if ((my_rank==0)&&(test!=0))
   {
      if (iset)
      {
         ferr=fopen(fname,"a");
         fflush(NULL);
         fprintf(ferr,"\nError in %s (error no=%d):\n",name,no);
         va_start(args,format);
         vfprintf(ferr,format,args);
         va_end(args);
#ifndef IGNORE_ERRORS
         fprintf(ferr,"\nProgram aborted\n");
#else
         fprintf(ferr,"\nProgram execution continues (IGNORE_ERRORS is set)\n");
#endif
         fclose(ferr);
      }
      else
      {
         fflush(NULL);
         printf("\nError in %s (error no=%d):\n",name,no);
         va_start(args,format);
         vprintf(format,args);
         va_end(args);
#ifndef IGNORE_ERRORS
         printf("\nProgram aborted\n");
#else
         printf("\nProgram execution continues (IGNORE_ERRORS is set)\n");
#endif
      }

      fflush(NULL);
      wait(1);
#ifndef IGNORE_ERRORS
      MPI_Abort(MPI_COMM_WORLD,no);
#endif
   }
}


void error_loc(int test,int no,char *name,char *format,...)
{
   int my_rank;
   va_list args;
   FILE *ferr;

   if (test!=0)
   {
      MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

      if (iset)
      {
         if (iloc)
         {
            ferr=fopen(fname_loc,"a");
            fflush(NULL);
            fprintf(ferr,"\nError in %s (error no=%d):\n",name,no);
         }
         else
         {
            ferr=fopen(fname,"a");
            fflush(NULL);
            fprintf(ferr,"\nError in %s (error no=%d, process no=%d):\n",
                    name,no,my_rank);
         }
         va_start(args,format);
         vfprintf(ferr,format,args);
         va_end(args);
#ifndef IGNORE_ERRORS
         fprintf(ferr,"\nProgram aborted\n");
#else
         fprintf(ferr,"\nProgram execution continues (IGNORE_ERRORS is set)\n");
#endif
         fclose(ferr);
      }
      else
      {
         fflush(NULL);
         printf("\nError in %s (error no=%d, process no=%d):\n",
                name,no,my_rank);
         va_start(args,format);
         vprintf(format,args);
         va_end(args);
#ifndef IGNORE_ERRORS
         printf("\nProgram aborted\n");
#else
         printf("\nProgram execution continues (IGNORE_ERRORS is set)\n");
#endif
      }

      fflush(NULL);
      wait(1);
#ifndef IGNORE_ERRORS
      MPI_Abort(MPI_COMM_WORLD,no);
#endif
   }
}
