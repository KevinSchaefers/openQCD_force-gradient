
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2005-2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Checks on the programs in the module salg.c.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "sflds.h"
#include "linalg.h"
#include "global.h"

#define _acc_sp(z,x,y) \
   (z).re+=(double)((x).re*(y).re+(x).im*(y).im); \
   (z).im+=(double)((x).re*(y).im-(x).im*(y).re)


static complex sp(int vol,spinor *pk,spinor *pl)
{
   complex w;
   complex_dble z;
   spinor *pm;

   z.re=0.0;
   z.im=0.0;
   pm=pk+vol;

   for (;pk<pm;pk++)
   {
      _acc_sp(z,(*pk).c1.c1,(*pl).c1.c1);
      _acc_sp(z,(*pk).c1.c2,(*pl).c1.c2);
      _acc_sp(z,(*pk).c1.c3,(*pl).c1.c3);

      _acc_sp(z,(*pk).c2.c1,(*pl).c2.c1);
      _acc_sp(z,(*pk).c2.c2,(*pl).c2.c2);
      _acc_sp(z,(*pk).c2.c3,(*pl).c2.c3);

      _acc_sp(z,(*pk).c3.c1,(*pl).c3.c1);
      _acc_sp(z,(*pk).c3.c2,(*pl).c3.c2);
      _acc_sp(z,(*pk).c3.c3,(*pl).c3.c3);

      _acc_sp(z,(*pk).c4.c1,(*pl).c4.c1);
      _acc_sp(z,(*pk).c4.c2,(*pl).c4.c2);
      _acc_sp(z,(*pk).c4.c3,(*pl).c4.c3);

      pl+=1;
   }

   w.re=(float)(z.re);
   w.im=(float)(z.im);

   return w;
}


int main(int argc,char *argv[])
{
   int my_rank,vol,icom,i;
   float r,zsq;
   double d,dmax,dall;
   complex z,w;
   spinor **ps,*pk,*pl,*pj;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);

      printf("\n");
      printf("Consistency of the programs in the module salg\n");
      printf("----------------------------------------------\n\n");

      print_lattice_sizes();
   }

   start_ranlux(0,12345);
   geometry();
   alloc_ws(10);
   ps=reserve_ws(10);
   dall=0.0;

   for (icom=0;icom<4;icom++)
   {
      if (((icom&0x1)==0)||(NPROC>1))
      {
         if (icom&0x2)
            vol=VOLUME_TRD;
         else
            vol=VOLUME;

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
            random_s(vol,icom,ps[i],1.0f);

         dmax=0.0;

         for (i=0;i<10;i++)
         {
            pk=ps[i];
            pl=ps[9-i];

            if (icom&0x1)
            {
               z=sp(VOLUME,pk,pl);
               MPI_Reduce(&z.re,&w.re,2,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
               MPI_Bcast(&w.re,2,MPI_FLOAT,0,MPI_COMM_WORLD);
            }
            else
               w=sp(VOLUME,pk,pl);

            z=spinor_prod(vol,icom,pk,pl);
            r=norm_square(vol,icom,pk)*norm_square(vol,icom,pl);
            d=(double)((z.re-w.re)*(z.re-w.re)+(z.im-w.im)*(z.im-w.im));
            d=sqrt(d/(double)(r));
            if (d>dmax)
               dmax=d;

            w.re=spinor_prod_re(vol,icom,pk,pl);
            d=(double)((z.re-w.re)*(z.re-w.re));
            d=fabs(d/(double)(r));
            if (d>dmax)
               dmax=d;

            z=spinor_prod(vol,icom,pk,pk);
            r=norm_square(vol,icom,pk);

            d=fabs((double)(z.im/r));
            if (d>dmax)
               dmax=d;

            d=fabs((double)(z.re/r-1.0f));
            if (d>dmax)
               dmax=d;
         }

         if (my_rank==0)
         {
            if (dmax>dall)
               dall=dmax;
            printf("Check of spinor_prod, spinor_prod_re\n");
            printf("and norm_square: %.2e\n\n",dmax);
         }

         dmax=0.0;
         z.re= 0.345f;
         z.im=-0.876f;
         zsq=z.re*z.re+z.im*z.im;

         for (i=0;i<9;i++)
         {
            pk=ps[i];
            pl=ps[i+1];

            w=spinor_prod(vol,icom,pk,pl);
            r=norm_square(vol,icom,pk)+zsq*norm_square(vol,icom,pl)
               +2.0f*(z.re*w.re-z.im*w.im);
            mulc_spinor_add(vol,icom,pk,pl,z);

            d=fabs((double)(r/norm_square(vol,icom,pk)-1.0f));
            if (d>dmax)
               dmax=d;
         }

         if (my_rank==0)
         {
            if (dmax>dall)
               dall=dmax;
            printf("Consistency of spinor_prod, norm_square\n");
            printf("and mulc_spinor_add: %.2e\n\n",dmax);
         }

         for (i=0;i<10;i++)
            random_s(vol,icom,ps[i],1.0f);

         dmax=0.0;
         r=-1.234f;
         z.re=-r;
         z.im=0.0f;

         for (i=0;i<8;i+=3)
         {
            pk=ps[i];
            pl=ps[i+1];
            pj=ps[i+2];

            assign_s2s(vol,icom,pk,pj);
            mulr_spinor_add(vol,icom,pk,pl,r);
            mulc_spinor_add(vol,icom,pk,pl,z);
            mulr_spinor_add(vol,icom,pk,pj,-1.0);

            d=(double)(norm_square(vol,icom,pk)/norm_square(vol,icom,pj));
            d=sqrt(d);
            if (d>dmax)
               dmax=d;

            assign_s2s(vol,icom,pl,pk);
            scale(vol,icom,r,pk);
            mulc_spinor_add(vol,icom,pk,pl,z);

            d=(double)(norm_square(vol,icom,pk)/norm_square(vol,icom,pl));
            d=sqrt(d);
            if (d>dmax)
               dmax=d;
         }

         if (my_rank==0)
         {
            if (dmax>dall)
               dall=dmax;
            printf("Consistency of mulr_spinor_add, scale\n");
            printf("and mulc_spinor_add: %.2e\n\n",dmax);
         }

         for (i=0;i<10;i++)
            random_s(vol,icom,ps[i],1.0f);

         dmax=0.0;

         for (i=0;i<10;i++)
         {
            pk=ps[i];

            if (i>0)
            {
               pl=ps[i-1];
               project(vol,icom,pk,pl);
               z=spinor_prod(vol,icom,pk,pl);

               d=(fabs((double)(z.re))+
                  fabs((double)(z.im)))/
                  sqrt((double)(norm_square(vol,icom,pk)));

               if (d>dmax)
                  dmax=d;
            }

            normalize(vol,icom,pk);
            r=norm_square(vol,icom,pk);

            d=fabs((double)(r-1.0f));
            if (d>dmax)
               dmax=d;
         }

         if (my_rank==0)
         {
            if (dmax>dall)
               dall=dmax;
            printf("Consistency of spinor_prod, norm_square,\n");
            printf("normalize and project: %.2e\n\n",dmax);
         }

         dmax=0.0;

         for (i=0;i<5;i++)
         {
            pk=ps[i];
            pl=ps[9-i];
            random_s(vol,icom,pk,1.0f);
            assign_s2s(vol,icom,pk,pl);
            mulg5(vol,icom,pk);
            mulg5(vol,icom,pk);

            z.re=-1.0f;
            z.im=0.0f;

            mulc_spinor_add(vol,icom,pl,pk,z);
            r=norm_square(vol,icom,pl)/norm_square(vol,icom,pk);
            d=sqrt((double)(r));
            if (d>dmax)
               dmax=d;

            random_s(vol,icom,pl,1.0f);
            z=spinor_prod(vol,icom,pk,pl);
            mulg5(vol,icom,pk);
            mulg5(vol,icom,pl);
            w=spinor_prod(vol,icom,pk,pl);

            d=(fabs((double)(z.re-w.re))+fabs((double)(z.im-w.im)))/
               (fabs((double)(z.re))+fabs((double)(z.im)));
            if (d>dmax)
               dmax=d;

            random_s(vol,icom,pk,1.0f);
            assign_s2s(vol,icom,pk,pl);
            mulg5(vol,icom,pk);
            mulmg5(vol,icom,pk);

            z.re=1.0f;
            z.im=0.0f;

            mulc_spinor_add(vol,icom,pl,pk,z);
            r=norm_square(vol,icom,pl)/norm_square(vol,icom,pk);
            d=sqrt((double)(r));
            if (d>dmax)
               dmax=d;
         }

         if (my_rank==0)
         {
            if (dmax>dall)
               dall=dmax;
            printf("Check of mulg5 and mulmg5: %.2e\n\n",dmax);
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
