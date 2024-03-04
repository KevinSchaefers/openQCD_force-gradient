
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Checks of the quadruple-precision summation programs.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <gmp.h>
#include "mpi.h"
#include "random.h"
#include "utils.h"
#include "global.h"


static void new_qflt(double *q)
{
   double m,r[4];

   m=ldexp(1.0,-52);
   ranlxd(r,4);
   r[0]=r[0]+0.03125*r[2];
   r[1]=r[1]+0.03125*r[3];

   q[0]=r[0];
   q[1]=0.0;

   acc_qflt(m*r[1],q);
}


static double qrel_diff(double *q,mpf_t *X)
{
   double d;
   mpf_t Y,Z,W;

   mpf_inits(Y,Z,W,NULL);

   mpf_set_d(Y,q[0]);
   mpf_set_d(Z,q[1]);
   mpf_add(W,Y,Z);
   mpf_sub(Y,*X,W);
   mpf_div(Z,Y,*X);
   d=fabs(mpf_get_d(Z));

   mpf_clears(Y,Z,W,NULL);

   return d;
}


static void global_mpf_sum(double *q,mpf_t *S)
{
   int my_rank,i,dmy;
   double r[2];
   mpf_t X,Y,Z;
   MPI_Status stat;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   dmy=1;
   mpf_inits(X,Y,Z,NULL);

   if (my_rank==0)
   {
      mpf_set_d(X,q[0]);
      mpf_set_d(Y,q[1]);
      mpf_add(*S,X,Y);
   }

   for (i=1;i<NPROC;i++)
   {
      if (my_rank==0)
      {
         MPI_Send(&dmy,1,MPI_INT,i,1,MPI_COMM_WORLD);
         MPI_Recv(r,2,MPI_DOUBLE,i,2,MPI_COMM_WORLD,&stat);

         mpf_set_d(X,r[0]);
         mpf_set_d(Y,r[1]);
         mpf_add(Z,*S,X);
         mpf_add(*S,Z,Y);
      }
      else if (my_rank==i)
      {
         MPI_Recv(&dmy,1,MPI_INT,0,1,MPI_COMM_WORLD,&stat);
         MPI_Send(q,2,MPI_DOUBLE,0,2,MPI_COMM_WORLD);
      }
   }

   mpf_clears(X,Y,Z,NULL);
}


int main(int argc,char *argv[])
{
   int my_rank;
   int i,j,n,is,ie;
   double xi,xn,d,dmax;
   double q1[2],q2[2],q3[2],qr[2];
   double s1[2],s2[2],s3[2];
   double *qu[3],*qv[3];
   mpf_t X,X3,X4,X5,Y,Z,Xsm,W;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);

      printf("\n");
      printf("Check of the quadruple-precision summation programs\n");
      printf("---------------------------------------------------\n\n");

      print_lattice_sizes();
   }

   check_machine();
   start_ranlux(0,1234);

   mpf_set_default_prec(128);
   mpf_inits(X,Y,Z,X3,X4,X5,Xsm,W,NULL);

   n=9700;
   qr[0]=0.0;
   qr[1]=0.0;
   ie=0;

   for (i=0;i<n;i++)
   {
      xi=(double)(i);
      xi=xi*xi;
      xi=xi*xi;
      acc_qflt(xi,qr);

      ie|=((qr[0]+qr[1])!=qr[0]);
   }

   error(ie!=0,1,"main [check2.c]",
         "Accumulated sum is not a proper quadruple-precision number");

   xn=(double)(n);

   mpf_set_d(X,xn);
   mpf_mul(Y,X,X);
   mpf_mul(X3,Y,X);
   mpf_mul(X4,X3,X);
   mpf_mul(X5,X4,X);

   mpf_set_d(Z,6.0);
   mpf_mul(Xsm,X5,Z);

   mpf_set_d(Z,-15.0);
   mpf_mul(Y,X4,Z);
   mpf_add(W,Xsm,Y);

   mpf_set_d(Z,10.0);
   mpf_mul(Y,X3,Z);
   mpf_add(Xsm,W,Y);

   mpf_sub(W,Xsm,X);

   mpf_set_d(Z,30.0);
   mpf_div(Xsm,W,Z);

   mpf_set_d(X,qr[0]);
   mpf_set_d(Y,qr[1]);
   mpf_add(Z,X,Y);
   mpf_sub(W,Xsm,Z);
   d=fabs(mpf_get_d(W));
   MPI_Reduce(&d,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      printf("Sum of n^4 for n from 0 to 9700-1:\n");
      gmp_printf("Sum       = %.20Fe\n",Z);
      gmp_printf("Check     = %.20Fe (formula using gmp)\n",Xsm);
      printf("Deviation = %.1e\n\n",dmax);
   }

   n=10000000;

   for (is=0;is<4;is++)
   {
      dmax=0.0;
      qr[0]=0.0;
      qr[1]=0.0;
      mpf_set_d(Xsm,0.0);

      for (i=0;i<n;i++)
      {
         new_qflt(q1);

         if ((is&0x1)&&(i&0x1))
         {
            q1[0]=-q1[0];
            q1[1]=-q1[1];
         }

         if (is>1)
            add_qflt(q1,qr,qr);
         else
         {
            acc_qflt(q1[0],qr);
            q1[1]=0.0;
         }

         ie|=((qr[0]+qr[1])!=qr[0]);

         mpf_set_d(X,q1[0]);
         mpf_add(Y,Xsm,X);
         mpf_set_d(X,q1[1]);
         mpf_add(Xsm,Y,X);
      }

      error(ie!=0,1,"main [check2.c]",
            "Sum is not a proper quadruple-precision number");

      d=qrel_diff(qr,&Xsm);
      MPI_Reduce(&d,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

      if (my_rank==0)
      {
         if (is>1)
            printf("Check of add_qflt() by summing 10^7 fully populated\n"
                   "quadruple-precision random numbers of O(1)");
         else
            printf("Check of acc_qflt() by summing 10^7 fully populated\n"
                   "double-precision random numbers of O(1)");
         if (is&0x1)
            printf(" with alternating sign:\n");
         else
            printf(":\n");

         printf("Local sum = %.16e\n",qr[0]);
         printf("Maximal relative deviation = %.1e\n",dmax);
      }

      global_mpf_sum(qr,&Xsm);
      qu[0]=qr;
      qv[0]=q1;
      global_qsum(1,qu,qv);
      d=qrel_diff(q1,&Xsm);

      if (my_rank==0)
      {
         printf("Global sum = %.16e\n",q1[0]);
         printf("Relative deviation = %.1e (global sum only)\n\n",d);
      }
   }

   dmax=0.0;
   qu[0]=q1;
   qu[1]=q2;
   qu[2]=q3;
   qv[0]=s1;
   qv[1]=s2;
   qv[2]=s3;

   for (i=0;i<10;i++)
   {
      new_qflt(q1);
      new_qflt(q2);
      new_qflt(q3);

      global_qsum(3,qu,qv);

      for (j=0;j<3;j++)
      {
         global_mpf_sum(qu[j],&Xsm);
         d=qrel_diff(qv[j],&Xsm);
         if (d>dmax)
            dmax=d;
      }

      q1[0]=s1[0];
      q1[1]=s1[1];
      MPI_Bcast(q1,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
      ie|=((q1[0]!=s1[0])||(q1[1]!=s1[1]));

      q1[0]=s2[0];
      q1[1]=s2[1];
      MPI_Bcast(q1,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
      ie|=((q1[0]!=s2[0])||(q1[1]!=s2[1]));

      q1[0]=s3[0];
      q1[1]=s3[1];
      MPI_Bcast(q1,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
      ie|=((q1[0]!=s3[0])||(q1[1]!=s3[1]));
   }

   if (my_rank==0)
   {
      printf("Global array sum:\n");
      printf("Relative deviation = %.1e\n",dmax);
   }

   dmax=0.0;

   for (i=0;i<10;i++)
   {
      new_qflt(q1);
      new_qflt(q2);
      new_qflt(q3);

      s1[0]=q1[0];
      s1[1]=q1[1];
      s2[0]=q2[0];
      s2[1]=q2[1];
      s3[0]=q3[0];
      s3[1]=q3[1];

      global_qsum(3,qu,qu);

      for (j=0;j<3;j++)
      {
         global_mpf_sum(qv[j],&Xsm);
         d=qrel_diff(qu[j],&Xsm);
         if (d>dmax)
            dmax=d;
      }

      s1[0]=q1[0];
      s1[1]=q1[1];
      MPI_Bcast(s1,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
      ie|=((q1[0]!=s1[0])||(q1[1]!=s1[1]));

      s1[0]=q2[0];
      s1[1]=q2[1];
      MPI_Bcast(s1,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
      ie|=((q2[0]!=s1[0])||(q2[1]!=s1[1]));

      s1[0]=q3[0];
      s1[1]=q3[1];
      MPI_Bcast(s1,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
      ie|=((q3[0]!=s1[0])||(q3[1]!=s1[1]));
   }

   if (my_rank==0)
   {
      printf("Relative deviation = %.1e (in place summation)\n\n",dmax);
   }

   error(ie!=0,1,"main [check2.c]","Global sum is not global");
   dmax=0.0;

   for (i=0;i<n;i++)
   {
      new_qflt(q1);
      new_qflt(qr);

      mpf_set_d(X,q1[0]);
      mpf_set_d(Y,q1[1]);
      mpf_add(Z,X,Y);
      mpf_set_d(X,qr[0]);
      mpf_set_d(Y,qr[1]);
      mpf_add(W,X,Y);
      mpf_mul(Xsm,W,Z);

      mul_qflt(q1,qr,qr);
      ie|=((qr[0]+qr[1])!=qr[0]);

      d=qrel_diff(qr,&Xsm);
      if (d>dmax)
         dmax=d;

      new_qflt(q1);
      new_qflt(qr);

      mpf_set_d(X,q1[0]);
      mpf_set_d(Y,q1[1]);
      mpf_add(Z,X,Y);
      mpf_set_d(X,qr[0]);
      mpf_mul(W,Z,X);

      scl_qflt(qr[0],q1);
      ie|=((q1[0]+q1[1])!=q1[0]);

      d=qrel_diff(q1,&W);
      if (d>dmax)
         dmax=d;
   }

   error(ie!=0,1,"main [check2.c]",
         "Multiplication results do not comply with qadruple precision format");
   d=dmax;
   MPI_Reduce(&d,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      mpf_set_d(X,q1[0]);
      mpf_set_d(Y,q1[1]);
      mpf_add(Z,X,Y);

      printf("Multiplication of random numbers:\n");
      gmp_printf("Last product   = %.20Fe\n",Z);
      gmp_printf("Check          = %.20Fe (formula using gmp)\n",W);
      printf("Maximal relative deviation = %.1e\n\n",dmax);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
