
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2015 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the round-robin code in the communication routines.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "utils.h"
#include "random.h"
#include "dft.h"

static int *is=NULL;


static int round_robin(int np,int r,int cp0)
{
   int cp1;

   if (cp0==0)
      cp1=np-1-r;
   else
   {
      cp1=np-1-cp0-r;
      if (cp1<0)
         cp1+=(np-1);

      if (cp1>0)
      {
         cp1-=r;
         if (cp1<=0)
            cp1+=(np-1);
      }
   }

   return cp1;
}


static void check_pairs(int np)
{
   int r,k,j,l,ie;

   ie=0;

   for (r=0;r<(np-1);r++)
   {
      for (k=0;k<np;k++)
      {
         j=round_robin(np,r,k);
         l=round_robin(np,r,j);

         ie|=((j<0)||(j>=np)||(l!=k));
      }
   }

   error(ie!=0,1,"check_pairs [check2.c]",
         "Processes are not properly paired");
}


static void check_duplicates(int np)
{
   int r,k,j,ie;

   if (is!=NULL)
      free(is);

   is=malloc(np*sizeof(*is));
   error(is==NULL,1,"check_duplicates [check2.c]",
         "Unable to allocate auxiliary array");
   ie=0;

   for (k=0;k<np;k++)
   {
      for (j=0;j<np;j++)
         is[j]=0;

      for (r=0;r<(np-1);r++)
      {
         j=round_robin(np,r,k);
         is[j]+=1;
      }

      for (j=0;j<np;j++)
      {
         if (j!=k)
            ie|=(is[j]!=1);
         else
            ie|=(is[j]!=0);
      }
   }

   error(ie!=0,1,"check_duplicates [check2.c]",
         "Duplicate process pairs");

   for (r=0;r<(np-1);r++)
   {
      for (j=0;j<np;j++)
         is[j]=0;

      for (k=0;k<np;k++)
      {
         j=round_robin(np,r,k);
         is[j]+=1;
      }

      for (j=0;j<np;j++)
         ie|=(is[j]!=1);
   }

   error(ie!=0,1,"check_duplicates [check2.c]",
         "Duplicate target processes");
}


int main(void)
{
   int np;

   printf("\n");
   printf("Round-robin code used in dft_gather()\n");
   printf("-------------------------------------\n\n");

   while(1)
   {
      printf("Select number of processes (must be even): ");
      (void)scanf("%d",&np);

      error((np<2)||((np%2)!=0),1,"main [check2.c]",
            "Improper choice of np");

      check_pairs(np);
      check_duplicates(np);

      printf("No errors discovered\n\n");
   }

   exit(0);
}
