
/*******************************************************************************
*
* File dft_wspace.c
*
* Copyright (C) 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* DFT workspace utility programs.
*
*   dft_wsp_t *alloc_dft_wsp(void)
*     Allocates a dft_wsp_t structure, sets its elements to zero and returns
*     its address. If the allocation fails, the program returns NULL.
*
*   void free_dft_wsp(dft_wsp_t *dwsp)
*     Frees the arrays in the structure dwsp and subsequently the structure
*     itself. Any NULL arrays encountered, including dwsp itself, are skipped
*     over.
*
*   complex_dble *set_dft_wsp0(int n,dft_wsp_t *dwsp)
*     Ensures that the arrays wf and ekx in the workspace dwsp have at
*     least n elements and sets ekx[k]=exp(i*2*pi*k/n) for all k=0,..,n-1.
*     The program returns wf or NULL if the allocation failed.
*
*   complex_dble *set_dft_wsp1(int n,dft_wsp_t *dwsp)
*     Ensures that the arrays fs and fts in the workspace dwsp have at
*     least n elements. The program returns fs or NULL if the allocation
*     failed.
*
*   complex_dble *set_dft_wsp2(int n,dft_wsp_t *dwsp)
*     Ensures that the array buf in the workspace dwsp has at least n
*     elements. The program returns buf or NULL if the allocation failed.
*
* The type dft_wsp_t is defined in include/dft.h. Its elements are
*
*   nwf        Current length of the arrays wf and exk.
*
*   nexk       Current value of n where ekx[k]=exp(i*2*pi*k/n) for all
*              k=0,..,n-1.
*
*   nfs        Current length of the arrays fs and fts.
*
*   nbuf       Current length of the array buf.
*
*   wf         Arrays of complex_dble.
*   exk
*
*   fs         Arrays of *complex_dble.
*   fts
*
*   buf        Array of complex_dble aligned to a 16 byte boundary.
*
* The arrays wf and exk are used in the module small_dft.c, all other
* arrays in the module fft.c.
*
* The programs in this module are thread-safe.
*
*******************************************************************************/

#define DFT_WSPACE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "dft.h"


dft_wsp_t *alloc_dft_wsp(void)
{
   dft_wsp_t *dwsp;

   dwsp=malloc(1*sizeof(*dwsp));

   if (dwsp!=NULL)
   {
      (*dwsp).nwf=0;
      (*dwsp).nekx=0;
      (*dwsp).nfs=0;
      (*dwsp).nbuf=0;

      (*dwsp).wf=NULL;
      (*dwsp).ekx=NULL;
      (*dwsp).fs=NULL;
      (*dwsp).fts=NULL;
      (*dwsp).buf=NULL;
   }

   return dwsp;
}


void free_dft_wsp(dft_wsp_t *dwsp)
{
   if (dwsp!=NULL)
   {
      if ((*dwsp).wf!=NULL)
         free((*dwsp).wf);
      if ((*dwsp).fs!=NULL)
         free((*dwsp).fs);
      if ((*dwsp).buf!=NULL)
         afree((*dwsp).buf);

      free(dwsp);
   }
}


static void set_ekx(int n,complex_dble *ekx)
{
   int k;
   double r;

   r=8.0*atan(1.0)/(double)(n);

   for (k=0;k<n;k++)
   {
      ekx[k].re=cos((double)(k)*r);
      ekx[k].im=sin((double)(k)*r);
   }
}


complex_dble *set_dft_wsp0(int n,dft_wsp_t *dwsp)
{
   int nwf;
   complex_dble *wf;

   if (n<1)
      return NULL;
   else if (n!=(*dwsp).nekx)
   {
      nwf=(*dwsp).nwf;

      if (n>nwf)
      {
         wf=malloc(2*n*sizeof(*wf));

         if (wf==NULL)
            return NULL;
         else if (nwf>0)
            free((*dwsp).wf);

         (*dwsp).nwf=n;
         (*dwsp).wf=wf;
      }

      (*dwsp).nekx=n;
      (*dwsp).ekx=(*dwsp).wf+n;
      set_ekx(n,(*dwsp).ekx);
   }

   return (*dwsp).wf;
}


complex_dble **set_dft_wsp1(int n,dft_wsp_t *dwsp)
{
   int nfs;
   complex_dble **fs;

   if (n<1)
      return NULL;

   nfs=(*dwsp).nfs;

   if (n>nfs)
   {
      fs=malloc(2*n*sizeof(*fs));

      if (fs==NULL)
         return NULL;
      else if (nfs>0)
         free((*dwsp).fs);

      (*dwsp).nfs=n;
      (*dwsp).fs=fs;
   }

   (*dwsp).fts=(*dwsp).fs+n;

   return (*dwsp).fs;
}


complex_dble *set_dft_wsp2(int n,dft_wsp_t *dwsp)
{
   int nbuf;
   complex_dble *buf;

   if (n<1)
      return NULL;

   nbuf=(*dwsp).nbuf;

   if (n>nbuf)
   {
      buf=amalloc(n*sizeof(*buf),4);

      if (buf==NULL)
         return NULL;
      else if (nbuf>0)
         afree((*dwsp).buf);

      (*dwsp).nbuf=n;
      (*dwsp).buf=buf;
   }

   return (*dwsp).buf;
}
