
/*******************************************************************************
*
* File check4.c
*
* Copyright (C) 2007, 2011, 2016, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Consistency checks on the programs in the module valg.
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


static complex_dble sp(int vol,complex *pk,complex *pl)
{
   int ix;
   complex_dble z;

   z.re=0.0;
   z.im=0.0;

   for (ix=0;ix<vol;ix++)
   {
      z.re+=(double)((*pk).re*(*pl).re+(*pk).im*(*pl).im);
      z.im+=(double)((*pk).re*(*pl).im-(*pk).im*(*pl).re);
      pk+=1;
      pl+=1;
   }

   return z;
}


int main(int argc,char *argv[])
{
   int my_rank,vol,icom,i;
   int bs[4],Ns,nb,nv;
   double rd,zsq;
   double d,dmax,dall;
   complex **wv,*pk,*pl,z;
   complex_dble wd,zd;
   FILE *flog=NULL,*fin=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check4.log","w",stdout);
      fin=freopen("check4.in","r",stdin);

      printf("\n");
      printf("Checks on the programs in the module valg\n");
      printf("-----------------------------------------\n\n");

      print_lattice_sizes();

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      fclose(fin);

      printf("bs = %d %d %d %d\n\n",bs[0],bs[1],bs[2],bs[3]);
      fflush(flog);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);

   start_ranlux(0,12345);
   geometry();

   Ns=4;
   set_dfl_parms(bs,Ns);
   nb=VOLUME/(bs[0]*bs[1]*bs[2]*bs[3]);
   nv=Ns*nb;

   alloc_wv(10);
   wv=reserve_wv(10);
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
            random_v(vol,icom,wv[i],1.0f);

         dmax=0.0;

         for (i=0;i<10;i++)
         {
            pk=wv[i];
            pl=wv[9-i];

            if (icom&0x1)
            {
               zd=sp(nv,pk,pl);
               MPI_Reduce(&zd.re,&wd.re,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
               MPI_Bcast(&wd.re,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
            }
            else
               wd=sp(nv,pk,pl);

            zd=vprod(vol,icom,pk,pl);
            rd=vnorm_square(vol,icom,pk)*vnorm_square(vol,icom,pl);
            d=(zd.re-wd.re)*(zd.re-wd.re)+(zd.im-wd.im)*(zd.im-wd.im);
            d=sqrt(d/rd);
            if (d>dmax)
               dmax=d;

            zd=vprod(vol,icom,pk,pk);
            rd=vnorm_square(vol,icom,pk);

            d=fabs((double)(zd.im/rd));
            if (d>dmax)
               dmax=d;

            d=fabs((double)(zd.re/rd-1.0f));
            if (d>dmax)
               dmax=d;
         }

         if (my_rank==0)
         {
            if (dmax>dall)
               dall=dmax;
            printf("Check of vprod and vnorm_square: %.2e\n\n",dmax);
         }

         dmax=0.0;
         zd.re= 0.345f;
         zd.im=-0.876f;
         zsq=zd.re*zd.re+zd.im*zd.im;

         for (i=0;i<9;i++)
         {
            pk=wv[i];
            pl=wv[i+1];

            wd=vprod(vol,icom,pk,pl);
            rd=vnorm_square(vol,icom,pk)+zsq*vnorm_square(vol,icom,pl)
               +2.0*(zd.re*wd.re-zd.im*wd.im);
            z.re=(float)(zd.re);
            z.im=(float)(zd.im);
            mulc_vadd(vol,icom,pk,pl,z);

            d=fabs(rd/vnorm_square(vol,icom,pk)-1.0);
            if (d>dmax)
               dmax=d;
         }

         if (my_rank==0)
         {
            if (dmax>dall)
               dall=dmax;
            printf("Consistency of vprod, vnorm_square\n");
            printf("and mulc_vadd: %.2e\n\n",dmax);
         }

         for (i=0;i<10;i++)
            random_v(vol,icom,wv[i],1.0f);

         dmax=0.0;

         for (i=0;i<10;i++)
         {
            pk=wv[i];

            if (i>0)
            {
               pl=wv[i-1];
               vproject(vol,icom,pk,pl);
               zd=vprod(vol,icom,pk,pl);

               d=(fabs(zd.re)+fabs(zd.im))/
                  sqrt(vnorm_square(vol,icom,pk));

               if (d>dmax)
                  dmax=d;
            }

            vnormalize(vol,icom,pk);
            rd=vnorm_square(vol,icom,pk);

            d=fabs(rd-1.0);
            if (d>dmax)
               dmax=d;
         }

         if (my_rank==0)
         {
            if (dmax>dall)
               dall=dmax;
            printf("Consistency of vprod, vnorm_square,\n");
            printf("vnormalize and vproject: %.2e\n\n",dmax);
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
