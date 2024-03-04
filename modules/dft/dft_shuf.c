
/*******************************************************************************
*
* File dft_shuf.c
*
* Copyright (C) 2015 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Array reordering program for the parallel discrete Fourier transform.
*
*   void dft_shuf(int nx,int ny,int csize,complex_dble *f,complex_dble *rf)
*     Rearranges the csize*nx*ny elements of the array f into an array rf
*     such that rf[s+csize*(i+j*nx)]=f[s+csize*(j+i*ny)], where the index
*     ranges are s=0,..,csize-1, i=0,..,nx-1 and j=0,..,ny-1.
*
* This program effectively transposes a nx x ny matrix M[i][j] of structures
* of csize complex numbers, where the index s in f[s+csize*(j+i*ny)] labels
* the csize elements of the structure.
*
* In the context of a multi-dimensional discrete Fourier transform, dft_shuf()
* is used to cycle through the space directions. A complex field in four
* dimensions, for example, may be locally represented by an array
*
*  phi[ix],  ix=x3+L3*x2+L2*L3*x1+L1*L2*L3*x0,
*
* where x0,..,x3 are the space coordinates (x*=0,1,..,L*-1). With this data
* layout, the Fourier transform in direction mu=0 may easily be performed
* by introducing an array lf[x0][j]=phi[j+L1*L2*L3*x0] with two indices,
* where j=x3+L3*x2+L2*L3*x1. Now when
*
*  dft_shuf(L0,L1*L2*L3,1,phi,rphi)
*
* is applied, the value of phi at the point (x0,x1,x2,x3) is given by
* rf[x1][k]=rphi[k+L2*L3*L0*x1], k=x0+L0*x3+L3*L0*x2. The program thus
* effectively performs a cyclic reshuffling of the coordinates in the
* lexicographic indexing of the space points.
*
* The program dft_shuf() is assumed to be called by the OpenMP master
* thread on any MPI process.
*
*******************************************************************************/

#define DFT_SHUF_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"
#include "dft.h"
#include "global.h"


void dft_shuf(int nx,int ny,int csize,complex_dble *f,complex_dble *rf)
{
   int k,i,j,s;
   int a[NTHREAD],b[NTHREAD];
   complex_dble *w1,*w2;

   error_loc((nx<1)||(ny<1)||(csize<1),1,"dft_shuf [dft_shuf.c]",
             "Parameters are out of range");

   if (nx>ny)
   {
      divide_range(nx,NTHREAD,a,b);

#pragma omp parallel private(k,i,j,s,w1,w2)
      {
         k=omp_get_thread_num();
         w2=f+a[k]*csize*ny;

         for (i=a[k];i<b[k];i++)
         {
            w1=rf+csize*i;

            for (j=0;j<ny;j++)
            {
               for (s=0;s<csize;s++)
                  w1[s]=w2[s];

               w1+=csize*nx;
               w2+=csize;
            }
         }
      }
   }
   else
   {
      divide_range(ny,NTHREAD,a,b);

#pragma omp parallel private(k,i,j,s,w1,w2)
      {
         k=omp_get_thread_num();
         w1=rf+a[k]*nx*csize;

         for (j=a[k];j<b[k];j++)
         {
            w2=f+csize*j;

            for (i=0;i<nx;i++)
            {
               for (s=0;s<csize;s++)
                  w1[s]=w2[s];

               w1+=csize;
               w2+=ny*csize;
            }
         }
      }
   }
}
