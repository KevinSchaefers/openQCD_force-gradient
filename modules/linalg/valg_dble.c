
/*******************************************************************************
*
* File valg_dble.c
*
* Copyright (C) 2007-2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic linear algebra routines for double-precision complex fields.
*
*   complex_qflt vprod_dble(int n,int icom,complex_dble *v,complex_dble *w)
*     Returns the scalar product of the n-vectors v and w.
*
*   qflt vnorm_square_dble(int n,int icom,complex_dble *v)
*     Returns the square of the norm of the n-vector v.
*
*   void mulc_vadd_dble(int n,int icom,complex_dble *v,complex_dble *w,
*                       complex_dble z)
*     Replaces the n-vector v by v+z*w.
*
*   void vscale_dble(int n,int icom,double r,complex_dble *v)
*     Replaces the n-vector v by r*v.
*
*   void vproject_dble(int n,int icom,complex_dble *v,complex_dble *w)
*     Replaces the n-vector v by v-(w,v)*w.
*
*   double vnormalize_dble(int n,int icom,complex_dble *v)
*     Replaces the n-vector v by v/||v|| and returns the 2-norm ||v||.
*     The division is omitted if the norm vanishes and an error occurs
*     if icom!=0 (if icom=0 it is up to the calling program to check
*     that the norm was non-zero).
*
* The quadruple-precision types qflt and complex_qflt are defined in su3.h.
* See doc/qsum.pdf for further explanations.
*
* All these programs operate on complex n-vectors whose base addresses are
* passed through the arguments. The length n of the arrays is specified by
* the parameter n and the meaning of the parameter icom is explained in
* the file sflds/README.icom.
*
*******************************************************************************/

#define VALG_DBLE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "utils.h"
#include "linalg.h"
#include "vflds.h"
#include "global.h"


static complex_qflt loc_vprod_dble(int n,complex_dble *v,complex_dble *w)
{
   double *qsm[2];
   complex_qflt cqsm;
   complex_dble smz,*vm,*vb;

   qsm[0]=cqsm.re.q;
   qsm[1]=cqsm.im.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   qsm[1][0]=0.0;
   qsm[1][1]=0.0;
   vm=v+n;

   for (vb=v;vb<vm;)
   {
      vb+=32;
      if (vb>vm)
         vb=vm;
      smz.re=0.0;
      smz.im=0.0;

      for (;v<vb;v++)
      {
         smz.re+=((*v).re*(*w).re+(*v).im*(*w).im);
         smz.im+=((*v).re*(*w).im-(*v).im*(*w).re);
         w+=1;
      }

      acc_qflt(smz.re,qsm[0]);
      acc_qflt(smz.im,qsm[1]);
   }

   return cqsm;
}


static qflt loc_vnorm_square_dble(int n,complex_dble *v)
{
   double smx,*qsm[1];
   qflt rqsm;
   complex_dble *vm,*vb;

   qsm[0]=rqsm.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   vm=v+n;

   for (vb=v;vb<vm;)
   {
      vb+=32;
      if (vb>vm)
         vb=vm;
      smx=0.0;

      for (;v<vb;v++)
         smx+=((*v).re*(*v).re+(*v).im*(*v).im);

      acc_qflt(smx,qsm[0]);
   }

   return rqsm;
}


static void loc_mulc_vadd_dble(int n,complex_dble *v,complex_dble *w,
                               complex_dble z)
{
   complex_dble *vm;

   vm=v+n;

   for (;v<vm;v++)
   {
      (*v).re+=(z.re*(*w).re-z.im*(*w).im);
      (*v).im+=(z.re*(*w).im+z.im*(*w).re);
      w+=1;
   }
}


static void loc_vscale_dble(int n,double r,complex_dble *v)
{
   complex_dble *vm;

   vm=v+n;

   for (;v<vm;v++)
   {
      (*v).re*=r;
      (*v).im*=r;
   }
}


complex_qflt vprod_dble(int n,int icom,complex_dble *v,complex_dble *w)
{
   int k;
   double *qsm[2];
   qflt reqsm,imqsm;
   complex_qflt cqsm;

   if ((icom&0x2)==0)
      cqsm=loc_vprod_dble(n,v,w);
   else
   {
      reqsm.q[0]=0.0;
      reqsm.q[1]=0.0;
      imqsm.q[0]=0.0;
      imqsm.q[1]=0.0;

#pragma omp parallel private(k,cqsm) reduction(sum_qflt : reqsm,imqsm)
      {
         k=omp_get_thread_num();
         cqsm=loc_vprod_dble(n,v+k*n,w+k*n);

         reqsm.q[0]=cqsm.re.q[0];
         reqsm.q[1]=cqsm.re.q[1];
         imqsm.q[0]=cqsm.im.q[0];
         imqsm.q[1]=cqsm.im.q[1];
      }

      cqsm.re.q[0]=reqsm.q[0];
      cqsm.re.q[1]=reqsm.q[1];
      cqsm.im.q[0]=imqsm.q[0];
      cqsm.im.q[1]=imqsm.q[1];
   }

   if ((NPROC>1)&&(icom&0x1))
   {
      qsm[0]=cqsm.re.q;
      qsm[1]=cqsm.im.q;
      global_qsum(2,qsm,qsm);
   }

   return cqsm;
}


qflt vnorm_square_dble(int n,int icom,complex_dble *v)
{
   int k;
   double *qsm[1];
   qflt rqsm;

   if ((icom&0x2)==0)
      rqsm=loc_vnorm_square_dble(n,v);
   else
   {
      rqsm.q[0]=0.0;
      rqsm.q[1]=0.0;

#pragma omp parallel private(k) reduction(sum_qflt : rqsm)
      {
         k=omp_get_thread_num();
         rqsm=loc_vnorm_square_dble(n,v+k*n);
      }
   }

   if ((NPROC>1)&&(icom&0x1))
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   return rqsm;
}


void mulc_vadd_dble(int n,int icom,complex_dble *v,complex_dble *w,
                    complex_dble z)
{
   int k;

   if ((icom&0x2)==0)
      loc_mulc_vadd_dble(n,v,w,z);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_mulc_vadd_dble(n,v+k*n,w+k*n,z);
      }
   }
}


void vscale_dble(int n,int icom,double r,complex_dble *v)
{
   int k;

   if ((icom&0x2)==0)
      loc_vscale_dble(n,r,v);
   else
   {
#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();
         loc_vscale_dble(n,r,v+k*n);
      }
   }
}


void vproject_dble(int n,int icom,complex_dble *v,complex_dble *w)
{
   complex_dble z;
   complex_qflt qz;

   qz=vprod_dble(n,icom,w,v);
   z.re=-qz.re.q[0];
   z.im=-qz.im.q[0];
   mulc_vadd_dble(n,icom,v,w,z);
}


double vnormalize_dble(int n,int icom,complex_dble *v)
{
   double r;
   qflt qr;

   qr=vnorm_square_dble(n,icom,v);
   r=sqrt(qr.q[0]);

   if (r!=0.0)
      vscale_dble(n,icom,1.0/r,v);
   else if (icom!=0)
      error_loc(1,1,"vnormalize_dble [valg_dble.c]",
                "Vector field has vanishing norm");

   return r;
}
