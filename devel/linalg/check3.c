
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2005-2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Checks on the programs in the module salg_dble.c.
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


static complex_dble sp(int vol,spinor_dble *pk,spinor_dble *pl)
{
   complex_dble z;
   spinor_dble *pm;

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

   return z;
}


int main(int argc,char *argv[])
{
   int my_rank,vol,icom,i;
   double r,cs,cr,zsq,d,dmax,dall;
   qflt qr;
   complex_dble z,w;
   complex_qflt qz;
   spinor_dble **psd,*pk,*pl,*pj;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);

      printf("\n");
      printf("Consistency of the programs in the module salg_dble\n");
      printf("---------------------------------------------------\n\n");

      print_lattice_sizes();
   }

   check_machine();
   start_ranlux(0,12345);
   geometry();
   alloc_wsd(10);
   psd=reserve_wsd(10);
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
            random_sd(vol,icom,psd[i],1.0);

         dmax=0.0;

         for (i=0;i<10;i++)
         {
            pk=psd[i];
            pl=psd[9-i];

            if (icom&0x1)
            {
               z=sp(VOLUME,pk,pl);
               MPI_Reduce(&z.re,&w.re,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
               MPI_Bcast(&w.re,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
            }
            else
               w=sp(VOLUME,pk,pl);

            qz=spinor_prod_dble(vol,icom,pk,pl);
            z.re=qz.re.q[0];
            z.im=qz.im.q[0];

            qr=norm_square_dble(vol,icom,pk);
            r=qr.q[0];
            qr=norm_square_dble(vol,icom,pl);
            r*=qr.q[0];
            d=(z.re-w.re)*(z.re-w.re)+(z.im-w.im)*(z.im-w.im);
            d=sqrt(d/r);
            if (d>dmax)
               dmax=d;

            qr=spinor_prod_re_dble(vol,icom,pk,pl);
            w.re=qr.q[0];
            d=(z.re-w.re)*(z.re-w.re);
            d=sqrt(d/r);
            if (d>dmax)
               dmax=d;

            qz=spinor_prod_dble(vol,icom,pk,pk);
            z.re=qz.re.q[0];
            z.im=qz.im.q[0];
            qr=norm_square_dble(vol,icom,pk);
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
            printf("Check of spinor_prod, spinor_prod_re\n");
            printf("and norm_square: %.2e\n\n",dmax);
         }

         dmax=0.0;

         for (i=0;i<10;i++)
         {
            pk=psd[i];
            pl=psd[9-i];

            qz=spinor_prod5_dble(vol,icom,pk,pl);
            z.re=qz.re.q[0];
            z.im=qz.im.q[0];
            mulg5_dble(vol,icom,pl);
            qz=spinor_prod_dble(vol,icom,pk,pl);
            w.re=qz.re.q[0];
            w.im=qz.im.q[0];
            qr=norm_square_dble(vol,icom,pk);
            r=qr.q[0];
            qr=norm_square_dble(vol,icom,pl);
            r*=qr.q[0];
            d=(z.re-w.re)*(z.re-w.re)+(z.im-w.im)*(z.im-w.im);
            d=sqrt(d/r);
            if (d>dmax)
               dmax=d;
         }

         if (my_rank==0)
         {
            if (dmax>dall)
               dall=dmax;
            printf("Consistency check of spinor_prod5, mulg5\n");
            printf("and spinor_prod: %.2e\n\n",dmax);
         }

         dmax=0.0;
         z.re= 0.345;
         z.im=-0.876;
         zsq=z.re*z.re+z.im*z.im;

         for (i=0;i<9;i++)
         {
            pk=psd[i];
            pl=psd[i+1];

            qz=spinor_prod_dble(vol,icom,pk,pl);
            w.re=qz.re.q[0];
            w.im=qz.im.q[0];
            qr=norm_square_dble(vol,icom,pk);
            r=qr.q[0];
            qr=norm_square_dble(vol,icom,pl);
            r+=zsq*qr.q[0]+2.0*(z.re*w.re-z.im*w.im);
            mulc_spinor_add_dble(vol,icom,pk,pl,z);
            qr=norm_square_dble(vol,icom,pk);

            d=fabs(r/qr.q[0]-1.0);
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
            random_sd(vol,icom,psd[i],1.0);

         dmax=0.0;
         r=-1.234;
         z.re=-r;
         z.im=0.0;

         for (i=0;i<8;i+=3)
         {
            pk=psd[i];
            pl=psd[i+1];
            pj=psd[i+2];

            assign_sd2sd(vol,icom,pk,pj);
            mulr_spinor_add_dble(vol,icom,pk,pl,r);
            mulc_spinor_add_dble(vol,icom,pk,pl,z);
            mulr_spinor_add_dble(vol,icom,pk,pj,-1.0);

            qr=norm_square_dble(vol,icom,pk);
            d=qr.q[0];
            qr=norm_square_dble(vol,icom,pj);
            d=sqrt(d/qr.q[0]);
            if (d>dmax)
               dmax=d;

            assign_sd2sd(vol,icom,pl,pk);
            scale_dble(vol,icom,r,pk);
            mulc_spinor_add_dble(vol,icom,pk,pl,z);

            qr=norm_square_dble(vol,icom,pk);
            d=qr.q[0];
            qr=norm_square_dble(vol,icom,pl);
            d=sqrt(d/qr.q[0]);
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
            random_sd(vol,icom,psd[i],1.0);

         dmax=0.0;
         cs=0.785;
         cr=-1.567;

         for (i=0;i<8;i+=3)
         {
            pk=psd[i];
            pl=psd[i+1];
            pj=psd[i+2];

            assign_sd2sd(vol,icom,pk,pj);
            combine_spinor_dble(vol,icom,pk,pl,cs,cr);
            scale_dble(vol,icom,cs,pj);
            mulr_spinor_add_dble(vol,icom,pj,pl,cr);
            mulr_spinor_add_dble(vol,icom,pk,pj,-1.0);

            qr=norm_square_dble(vol,icom,pk);
            d=qr.q[0];
            qr=norm_square_dble(vol,icom,pj);
            d=sqrt(d/qr.q[0]);
            if (d>dmax)
               dmax=d;
         }

         if (my_rank==0)
         {
            if (dmax>dall)
               dall=dmax;
            printf("Consistency of mulr_spinor_add, scale\n");
            printf("and combine_spinor: %.2e\n\n",dmax);
         }

         for (i=0;i<10;i++)
            random_sd(vol,icom,psd[i],1.0);

         dmax=0.0;

         for (i=0;i<10;i++)
         {
            pk=psd[i];

            if (i>0)
            {
               pl=psd[i-1];
               project_dble(vol,icom,pk,pl);
               qz=spinor_prod_dble(vol,icom,pk,pl);
               z.re=qz.re.q[0];
               z.im=qz.im.q[0];
               qr=norm_square_dble(vol,icom,pk);
               d=(fabs(z.re)+fabs(z.im))/sqrt(qr.q[0]);

               if (d>dmax)
                  dmax=d;
            }

            (void)(normalize_dble(vol,icom,pk));
            qr=norm_square_dble(vol,icom,pk);
            r=qr.q[0];

            d=fabs(r-1.0);
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
            pk=psd[i];
            pl=psd[9-i];
            random_sd(vol,icom,pk,1.0);
            assign_sd2sd(vol,icom,pk,pl);
            mulg5_dble(vol,icom,pk);
            mulg5_dble(vol,icom,pk);

            z.re=-1.0;
            z.im=0.0;

            mulc_spinor_add_dble(vol,icom,pl,pk,z);
            qr=norm_square_dble(vol,icom,pl);
            r=qr.q[0];
            qr=norm_square_dble(vol,icom,pk);
            d=sqrt(r/qr.q[0]);
            if (d>dmax)
               dmax=d;

            random_sd(vol,icom,pl,1.0);
            qz=spinor_prod_dble(vol,icom,pk,pl);
            z.re=qz.re.q[0];
            z.im=qz.im.q[0];
            mulg5_dble(vol,icom,pk);
            mulg5_dble(vol,icom,pl);
            qz=spinor_prod_dble(vol,icom,pk,pl);
            w.re=qz.re.q[0];
            w.im=qz.im.q[0];

            d=(fabs(z.re-w.re)+fabs(z.im-w.im))/
               (fabs(z.re)+fabs(z.im));
            if (d>dmax)
               dmax=d;

            random_sd(vol,icom,pk,1.0);
            assign_sd2sd(vol,icom,pk,pl);
            mulg5_dble(vol,icom,pk);
            mulmg5_dble(vol,icom,pk);

            z.re=1.0;
            z.im=0.0;

            mulc_spinor_add_dble(vol,icom,pl,pk,z);
            qr=norm_square_dble(vol,icom,pl);
            r=qr.q[0];
            qr=norm_square_dble(vol,icom,pk);
            d=sqrt(r/qr.q[0]);
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
