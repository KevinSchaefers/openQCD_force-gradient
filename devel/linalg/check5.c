
/*******************************************************************************
*
* File check5.c
*
* Copyright (C) 2007-2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Checks on the programs in the module valg_dble.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "vflds.h"
#include "linalg.h"
#include "global.h"


static complex_dble sp(int vol,complex_dble *pk,complex_dble *pl)
{
   int ix;
   double x,y;
   complex_dble z;

   x=0.0;
   y=0.0;

   for (ix=0;ix<vol;ix++)
   {
      x+=(double)((*pk).re*(*pl).re+(*pk).im*(*pl).im);
      y+=(double)((*pk).re*(*pl).im-(*pk).im*(*pl).re);
      pk+=1;
      pl+=1;
   }

   z.re=x;
   z.im=y;

   return z;
}


int main(int argc,char *argv[])
{
   int my_rank,vol,icom,i;
   int bs[4],Ns,nb,nv;
   double r,zsq;
   double d,dmax,dall;
   qflt qr;
   complex_dble w,z;
   complex_dble **wvd,*pk,*pl;
   complex_qflt qz;
   FILE *flog=NULL,*fin=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check5.log","w",stdout);
      fin=freopen("check4.in","r",stdin);

      printf("\n");
      printf("Checks on the programs in the module valg_dble\n");
      printf("----------------------------------------------\n\n");

      print_lattice_sizes();

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      fclose(fin);

      printf("bs = %d %d %d %d\n\n",bs[0],bs[1],bs[2],bs[3]);
      fflush(flog);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);

   check_machine();
   start_ranlux(0,12345);
   geometry();

   Ns=4;
   set_dfl_parms(bs,Ns);
   nb=VOLUME/(bs[0]*bs[1]*bs[2]*bs[3]);
   nv=Ns*nb;

   alloc_wvd(10);
   wvd=reserve_wvd(10);
   dall=0.0;

   for (icom=0;icom<4;icom++)
   {
      if (((icom&0x1)==0)||(NPROC>1))
      {
         if (icom&0x2)
            vol=nv/NTHREAD;
         else
            vol=nv;

         if (my_rank==0)
         {
            if (icom==0)
            {
               printf("Local checks w/o threading\n");
               printf("==========================\n\n");
            }
            else if (icom==1)
            {
               printf("Global checks w/o threading\n");
               printf("===========================\n\n");
            }
            else if (icom==2)
            {
               printf("Local checks with threading\n");
               printf("===========================\n\n");
            }
            else
            {
               printf("Global checks with threading\n");
               printf("============================\n\n");
            }
         }

         for (i=0;i<10;i++)
            random_vd(vol,icom,wvd[i],1.0f);

         dmax=0.0;

         for (i=0;i<10;i++)
         {
            pk=wvd[i];
            pl=wvd[9-i];

            if (icom&0x1)
            {
               z=sp(nv,pk,pl);
               MPI_Reduce(&z.re,&w.re,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
               MPI_Bcast(&w.re,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
            }
            else
               w=sp(nv,pk,pl);

            qz=vprod_dble(vol,icom,pk,pl);
            z.re=qz.re.q[0];
            z.im=qz.im.q[0];
            qr=vnorm_square_dble(vol,icom,pk);
            r=qr.q[0];
            qr=vnorm_square_dble(vol,icom,pl);
            r*=qr.q[0];
            d=(z.re-w.re)*(z.re-w.re)+(z.im-w.im)*(z.im-w.im);
            d=sqrt(d/r);
            if (d>dmax)
               dmax=d;

            qz=vprod_dble(vol,icom,pk,pk);
            z.re=qz.re.q[0];
            z.im=qz.im.q[0];
            qr=vnorm_square_dble(vol,icom,pk);
            r=qr.q[0];

            d=fabs(z.im/r);
            if (d>dmax)
               dmax=d;

            d=fabs(z.re/r-1.0);
            if (d>dmax)
               dmax=d;
         }

         if (my_rank==0)
         {
            if (dmax>dall)
               dall=dmax;
            printf("Check of vprod_dble and vnorm_square_dble: %.2e\n\n",
                   dmax);
         }

         dmax=0.0;
         z.re= 0.345;
         z.im=-0.876;
         zsq=z.re*z.re+z.im*z.im;

         for (i=0;i<9;i++)
         {
            pk=wvd[i];
            pl=wvd[i+1];

            qz=vprod_dble(vol,icom,pk,pl);
            w.re=qz.re.q[0];
            w.im=qz.im.q[0];
            qr=vnorm_square_dble(vol,icom,pk);
            r=qr.q[0];
            qr=vnorm_square_dble(vol,icom,pl);
            r+=zsq*qr.q[0]+2.0*(z.re*w.re-z.im*w.im);
            mulc_vadd_dble(vol,icom,pk,pl,z);

            qr=vnorm_square_dble(vol,icom,pk);
            d=fabs(r/qr.q[0]-1.0);
            if (d>dmax)
               dmax=d;
         }

         if (my_rank==0)
         {
            if (dmax>dall)
               dall=dmax;
            printf("Consistency of vprod_dble, vnorm_square_dble\n");
            printf("and mulc_vadd_dble: %.2e\n\n",dmax);
         }

         for (i=0;i<10;i++)
            random_vd(vol,icom,wvd[i],1.0f);

         dmax=0.0;

         for (i=0;i<10;i++)
         {
            pk=wvd[i];

            if (i>0)
            {
               pl=wvd[i-1];
               vproject_dble(vol,icom,pk,pl);
               qz=vprod_dble(vol,icom,pk,pl);
               z.re=qz.re.q[0];
               z.im=qz.im.q[0];
               qr=vnorm_square_dble(vol,icom,pk);

               d=(fabs(z.re)+fabs(z.im))/sqrt(qr.q[0]);
               if (d>dmax)
                  dmax=d;
            }

            (void)(vnormalize_dble(vol,icom,pk));
            qr=vnorm_square_dble(vol,icom,pk);

            d=fabs(qr.q[0]-1.0);
            if (d>dmax)
               dmax=d;
         }

         if (my_rank==0)
         {
            if (dmax>dall)
               dall=dmax;
            printf("Consistency of vprod_dble, vnorm_square_dble,\n");
            printf("vnormalize_dble and vproject_dble: %.2e\n\n",dmax);
         }
      }
   }

   if (my_rank==0)
   {
      printf("Maximal deviation in all tests: %.2e\n\n",dall);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
