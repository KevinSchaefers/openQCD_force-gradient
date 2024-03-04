
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2005-2011, 2018, 2021 Martin Luescher, Filippo Palombi
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Checks of the programs in the module liealg.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "utils.h"
#include "lattice.h"
#include "linalg.h"
#include "global.h"

static int my_rank;
static double var[64],var_all[64];


static void alg2mat(su3_alg_dble *X,su3_dble *u)
{
   (*u).c11.re=0.0;
   (*u).c11.im=(*X).c1+(*X).c2;

   (*u).c22.re=0.0;
   (*u).c22.im=(*X).c2-2.0*(*X).c1;

   (*u).c33.re=0.0;
   (*u).c33.im=(*X).c1-2.0*(*X).c2;

   (*u).c12.re=(*X).c3;
   (*u).c12.im=(*X).c4;
   (*u).c21.re=-(*X).c3;
   (*u).c21.im=(*X).c4;

   (*u).c13.re=(*X).c5;
   (*u).c13.im=(*X).c6;
   (*u).c31.re=-(*X).c5;
   (*u).c31.im=(*X).c6;

   (*u).c23.re=(*X).c7;
   (*u).c23.im=(*X).c8;
   (*u).c32.re=-(*X).c7;
   (*u).c32.im=(*X).c8;
}


static double frob_norm_square(su3_alg_dble *X)
{
   su3_dble wud[2] ALIGNED16;

   alg2mat(X,wud);
   su3xsu3(wud,wud,wud+1);

   return -2.0*(wud[1].c11.re+wud[1].c22.re+wud[1].c33.re);
}


static void chk_norm_square(su3_alg_dble *X)
{
   int n,i,ie;
   int vol,icom;
   double d,dmax,*r;
   qflt qr,qrsm;

   for (icom=0;icom<4;icom+=2)
   {
      if (icom&0x2)
         vol=VOLUME_TRD;
      else
         vol=VOLUME;

      set_alg2zero(vol,icom,X);
      ie=0;

      for (n=0;n<VOLUME;n++)
      {
         r=(double*)(X+n);

         for (i=0;i<8;i++)
            ie|=(r[i]!=0.0);

         X[n].c3=1.0;
      }

      qrsm=norm_square_alg(vol,icom,X);

      error(ie!=0,1,"chk_norm_square [check1.c]",
            "Field is incorrectly initialized");
      error(qrsm.q[0]!=(4.0*(double)(VOLUME)),1,
            "chk_norm_square [check1.c]","Incorrect element count");
   }

   qrsm.q[0]=0.0;
   qrsm.q[1]=0.0;
   dmax=0.0;
   random_alg(VOLUME,0,X);

   for (n=0;n<VOLUME;n++)
   {
      qr=norm_square_alg(1,0,X+n);
      d=fabs(1.0-(frob_norm_square(X+n)/qr.q[0]));

      if (d>dmax)
         dmax=d;

      add_qflt(qr.q,qrsm.q,qrsm.q);
   }

   if (NPROC>1)
   {
      d=dmax;
      MPI_Reduce(&d,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   }

   if (my_rank==0)
   {
      printf("Program norm_square_alg():\n");
      printf("Relative deviation = %.1e (single elements)\n",dmax);
   }

   qr=norm_square_alg(VOLUME,0,X);
   qr.q[0]=-qr.q[0];
   qr.q[1]=-qr.q[1];
   add_qflt(qr.q,qrsm.q,qrsm.q);
   dmax=fabs(qrsm.q[0]/qr.q[0]);

   if (NPROC>1)
   {
      d=dmax;
      MPI_Reduce(&d,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
   }

   if (my_rank==0)
      printf("Relative deviation = %.1e (local sum)\n",dmax);

   qr=norm_square_alg(VOLUME,0,X);
   qrsm=norm_square_alg(VOLUME,1,X);

   if (NPROC>1)
   {
      d=qr.q[0];
      MPI_Reduce(&d,qr.q,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   }

   dmax=fabs(1.0-qr.q[0]/qrsm.q[0]);

   if (my_rank==0)
      printf("Relative deviation = %.1e (global sum)\n",dmax);

   qr=norm_square_alg(VOLUME_TRD,3,X);
   qr.q[0]=-qr.q[0];
   qr.q[1]=-qr.q[1];

   add_qflt(qr.q,qrsm.q,qr.q);
   dmax=fabs(qr.q[0]/qrsm.q[0]);

   if (my_rank==0)
      printf("Relative deviation = %.1e (global threaded sum)\n\n",dmax);
}


static void chk_scalar_prod(su3_alg_dble *X,su3_alg_dble *Y)
{
   int n,vol,icom;
   double dmax;
   qflt qrx,qry,qrxy,qrxpy;

   for (icom=0;icom<4;icom+=2)
   {
      if (icom&0x2)
         vol=VOLUME_TRD;
      else
         vol=VOLUME;

      random_alg(vol,icom,X);
      random_alg(vol,icom,Y);

      qrx=norm_square_alg(vol,icom+1,X);
      qry=norm_square_alg(vol,icom+1,Y);
      qrxy=scalar_prod_alg(vol,icom+1,X,Y);

      for (n=0;n<VOLUME;n++)
      {
         X[n].c1+=Y[n].c1;
         X[n].c2+=Y[n].c2;
         X[n].c3+=Y[n].c3;
         X[n].c4+=Y[n].c4;
         X[n].c5+=Y[n].c5;
         X[n].c6+=Y[n].c6;
         X[n].c7+=Y[n].c7;
         X[n].c8+=Y[n].c8;
      }

      qrxpy=norm_square_alg(vol,icom+1,X);
      qrx.q[0]=-qrx.q[0];
      qrx.q[1]=-qrx.q[1];
      qry.q[0]=-qry.q[0];
      qry.q[1]=-qry.q[1];
      add_qflt(qrx.q,qrxpy.q,qrxpy.q);
      add_qflt(qry.q,qrxpy.q,qrxpy.q);
      dmax=fabs(0.5*qrxpy.q[0]-qrxy.q[0]);
      dmax/=sqrt(qrx.q[0]*qry.q[0]);

      if (my_rank==0)
      {
         printf("Program scalar_prod_alg() (icom=%d):\n",icom);
         printf("Relative deviation = %.1e\n",dmax);
         if (icom)
            printf("\n");
      }
   }
}


static void chk_random_alg(su3_alg_dble *X)
{
   int n,i,j;
   int vol,icom;
   double d,dmax,*r;
   double rn,cij,eij;

   for (icom=0;icom<4;icom+=2)
   {
      if (icom&0x2)
         vol=VOLUME_TRD;
      else
         vol=VOLUME;

      random_alg(vol,icom,X);

      for (i=0;i<8;i++)
      {
         for (j=0;j<8;j++)
            var[8*i+j]=0.0;
      }

      for (n=0;n<VOLUME;n++)
      {
         r=(double*)(X+n);

         for (i=0;i<8;i++)
         {
            for (j=i;j<8;j++)
               var[8*i+j]+=r[i]*r[j];
         }
      }

      if (NPROC>1)
         MPI_Reduce(var,var_all,64,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      else
      {
         for (i=0;i<64;i++)
            var_all[i]=var[i];
      }

      if (my_rank==0)
      {
         printf("Program random_alg() (icom=%d):\n",icom);
         dmax=0.0;
         rn=1.0/((double)(VOLUME)*(double)(NPROC));

         for (i=0;i<8;i++)
         {
            for (j=i;j<8;j++)
            {
               if ((i==j)&&(i>1))
               {
                  cij=1.0/4.0;
                  eij=sqrt(2.0*rn)/4.0;
               }
               else if (i==j)
               {
                  cij=1.0/9.0;
                  eij=sqrt(2.0*rn)/9.0;
               }
               else if ((i==0)&&(j==1))
               {
                  cij=1.0/18.0;
                  eij=sqrt(5.0*rn)/18.0;
               }
               else if ((i<2)&&(j>1))
               {
                  cij=0.0;
                  eij=sqrt(rn)/6.0;
               }
               else
               {
                  cij=0.0;
                  eij=sqrt(rn)/4.0;
               }

               var_all[8*i+j]*=rn;

               if (cij!=0.0)
               {
                  printf("<b[%d]*b[%d]> = % .4e, deviation = %.1e+-%.1e\n",
                         i,j,var_all[8*i+j],fabs(var_all[8*i+j]-cij),eij);
               }
               else
               {
                  d=fabs(var_all[8*i+j])/eij;

                  if (d>dmax)
                     dmax=d;
               }
            }
         }

         eij=sqrt(rn)/4.0;
         printf("\n");
         printf("For all other i,j, ");
         printf("max|<b[i]*b[j]>| = %.1e (should be %.1e or so)\n\n",
                dmax*eij,2.0*eij);
      }
   }
}


static void chk_mul_add_assign(su3_alg_dble *X,su3_alg_dble *Y)
{
   int vol,icom;
   double rn,dmax;
   qflt qrx,qry,qrxy,qrsm;

   rn=-1.2345;

   for (icom=0;icom<4;icom+=2)
   {
      if (icom&0x2)
         vol=VOLUME_TRD;
      else
         vol=VOLUME;

      random_alg(vol,icom,X);
      random_alg(vol,icom,Y);

      qrx=norm_square_alg(vol,icom+1,X);
      qry=norm_square_alg(vol,icom+1,Y);
      qrxy=scalar_prod_alg(vol,icom+1,X,Y);

      muladd_assign_alg(vol,icom,rn,X,Y);
      qrsm=norm_square_alg(vol,icom+1,Y);

      dmax=qrsm.q[0]-qry.q[0]-rn*rn*qrx.q[0]-2.0*rn*qrxy.q[0];
      dmax=fabs(dmax)/qrsm.q[0];

      if (my_rank==0)
      {
         printf("Program muladd_assign_alg() (icom=%d):\n",icom);
         printf("Relative deviation = %.1e\n",dmax);
         if (icom&0x2)
            printf("\n");
      }
   }
}


static void chk_unorm_alg(su3_alg_dble *X)
{
   int n,vol,icom;
   double nrm,ns,dist,dmax,tol;

   for (icom=0;icom<4;icom+=2)
   {
      if (icom&0x2)
         vol=VOLUME_TRD;
      else
         vol=VOLUME;

      random_alg(vol,icom,X);
      nrm=unorm_alg(vol,icom+1,X);
      dist=nrm;
      dmax=0.0;

      for (n=0;n<VOLUME;n++)
      {
         ns=sqrt(frob_norm_square(X+n));

         if (ns<=nrm)
         {
            ns=nrm-ns;
            if (ns<dist)
               dist=ns;
         }
         else
         {
            ns=ns-nrm;
            if (ns<dist)
               dist=ns;
            if (ns>dmax)
               dmax=ns;
         }
      }

      tol=16.0*nrm*DBL_EPSILON;

      if (NPROC>1)
      {
         ns=dist;
         MPI_Reduce(&ns,&dist,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
         ns=dmax;
         MPI_Reduce(&ns,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      }

      if (my_rank==0)
      {
         printf("Program unorm_alg() (icom=%d):\n",icom);
         if ((dist<=tol)&&(dmax<=tol))
            printf("No deviation discovered\n\n");
         else
         {
            error(dist>tol,1,"chk_unorm_alg [check1.c]",
                  "Uniform norm is not correctly calculated");
         }
      }
   }
}


int main(int argc,char *argv[])
{
   su3_alg_dble *X,*Y;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);

      printf("\n");
      printf("Checks of the programs in the module liealg\n");
      printf("-------------------------------------------\n\n");

      print_lattice_sizes();
   }

   check_machine();
   start_ranlux(0,1234);
   geometry();

   X=amalloc(2*VOLUME*sizeof(*X),4);
   error(X==NULL,1,"main [check1.c]","Unable to allocate field array");
   Y=X+VOLUME;

   chk_norm_square(X);
   chk_scalar_prod(X,Y);
   chk_random_alg(X);
   chk_mul_add_assign(X,Y);
   chk_unorm_alg(X);

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
