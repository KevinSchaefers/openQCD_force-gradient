
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Test of the programs in the module array.c.
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "random.h"
#include "utils.h"


static void check_array(array_t *a)
{
   unsigned int d,p,i,ie;
   int *ri;
   size_t m,k,*n;
   long addr,base;
   void **p0,**p1;
   double *rx;

   d=(*a).d;
   n=(*a).n;
   p=(*a).p;

   p0=(void**)((*a).a);
   m=n[0];
   ie=0;

   if (d>1)
   {
      for (i=1;i<(d-1);i++)
      {
         m*=n[i];
         p1=(void**)(p0[0]);

         for (k=0;k<n[i-1];k++)
            ie|=((void**)(p0[k])!=p1+k*n[i]);

         p0=p1;
      }

      m*=n[d-1];
      p0=(void**)(p0[0]);
   }

   error(ie!=0,1,"check_array [check3.c]",
         "Multi-dimensional array is not correctly allocated");

   base=1L<<p;
   addr=(long)(p0);

   error((addr%base)!=0L,1,"check_array [check3.c]",
         "Multi-dimensional array is not correctly aligned");

   if ((*a).size==sizeof(double))
   {
      rx=(double*)(p0);

      for (k=0;k<m;k++)
         ie|=(rx[k]!=0.0);
   }
   else if ((*a).size==sizeof(int))
   {
      ri=(int*)(p0);

      for (k=0;k<m;k++)
         ie|=(ri[k]!=0);
   }
   else
      error(1,1,"check_array [check3.c]",
            "Unexpected size parameter");

   error(ie!=0,1,"check_array [check3.c]",
         "Multi-dimensional array is not correctly initialized");
}


static void random_data(array_t *a)
{
   int ***ai;
   size_t i0,i1,i2,*n;
   double m,r,***ad;

   n=(*a).n;
   m=1.0e8;

   if ((*a).size==4)
   {
      ai=(int***)((*a).a);

      for (i0=0;i0<n[0];i0++)
      {
         for (i1=0;i1<n[1];i1++)
         {
            for (i2=0;i2<n[2];i2++)
            {
               gauss_dble(&r,1);
               ai[i0][i1][i2]=(int)(m*r);
            }
         }
      }
   }
   else
   {
      ad=(double***)((*a).a);

      for (i0=0;i0<n[0];i0++)
      {
         for (i1=0;i1<n[1];i1++)
         {
            for (i2=0;i2<n[2];i2++)
            {
               gauss_dble(&r,1);
               ad[i0][i1][i2]=m*r;
            }
         }
      }
   }
}


static void cmp_arrays(array_t *a,array_t *b)
{
   int ie;
   int ***ai,***bi;
   size_t i0,i1,i2,*n;
   double ***ad,***bd;

   n=(*a).n;
   ie=0;

   if ((*a).size==4)
   {
      ai=(int***)((*a).a);
      bi=(int***)((*b).a);

      for (i0=0;i0<n[0];i0++)
      {
         for (i1=0;i1<n[1];i1++)
         {
            for (i2=0;i2<n[2];i2++)
               ie|=(ai[i0][i1][i2]!=bi[i0][i1][i2]);
         }
      }
   }
   else
   {
      ad=(double***)((*a).a);
      bd=(double***)((*b).a);

      for (i0=0;i0<n[0];i0++)
      {
         for (i1=0;i1<n[1];i1++)
         {
            for (i2=0;i2<n[2];i2++)
               ie|=(ad[i0][i1][i2]!=bd[i0][i1][i2]);
         }
      }
   }

   error(ie!=0,1,"cmp_arrays [check3.c]","Array elements differ");
}


static void check_array_io(array_t *a)
{
   FILE *fdat;
   array_t *b;

   random_data(a);

   fdat=fopen("test.dat","wb");
   error(fdat==NULL,1,"check_array_io [check3.c]","Unable to open data file");
   write_array(fdat,a);
   fclose(fdat);

   b=alloc_array((*a).d,(*a).n,(*a).size,(*a).p);
   fdat=fopen("test.dat","rb");
   error(fdat==NULL,2,"check_array_io [check3.c]","Unable to open data file");
   read_array(fdat,b);
   fclose(fdat);
   remove("test.dat");
   cmp_arrays(a,b);
   free_array(b);
}


int main(void)
{
   unsigned int d;
   size_t n[4];
   array_t *a;

   printf("\n");
   printf("Test of the programs in the module array.c\n");
   printf("------------------------------------------\n\n");

   rlxd_init(1,1,1234,1);

   d=1;
   n[0]=7;
   a=alloc_array(d,n,sizeof(double),4);
   check_array(a);
   free_array(a);

   d=2;
   n[0]=4;
   n[1]=16;
   a=alloc_array(d,n,sizeof(double),0);
   check_array(a);
   free_array(a);

   d=3;
   n[0]=5;
   n[1]=2;
   n[2]=1;
   a=alloc_array(d,n,sizeof(double),5);
   check_array(a);
   check_array_io(a);
   free_array(a);

   d=4;
   n[0]=5;
   n[1]=6;
   n[2]=7;
   n[3]=8;
   a=alloc_array(d,n,sizeof(double),9);
   check_array(a);
   free_array(a);

   d=1;
   n[0]=7;
   a=alloc_array(d,n,sizeof(int),4);
   check_array(a);
   free_array(a);

   d=2;
   n[0]=4;
   n[1]=16;
   a=alloc_array(d,n,sizeof(int),0);
   check_array(a);
   free_array(a);

   d=3;
   n[0]=5;
   n[1]=2;
   n[2]=1;
   a=alloc_array(d,n,sizeof(int),5);
   check_array(a);
   check_array_io(a);
   free_array(a);

   d=4;
   n[0]=5;
   n[1]=6;
   n[2]=7;
   n[3]=8;
   a=alloc_array(d,n,sizeof(int),9);
   check_array(a);
   free_array(a);

   printf("No errors discovered --- the programs work correctly\n\n");

   exit(0);
}
