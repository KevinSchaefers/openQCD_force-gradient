
/*******************************************************************************
*
* File sw_term.c
*
* Copyright (C) 2011-2018, 2021 Martin Luescher, Antonio Rago
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the SW term.
*
*   int sw_order(void)
*     Returns the order N required for the computation of the exponential
*     of the Pauli term [scaled by 1/(4+m0)] to machine precision.
*
*   void pauli_term(double c,u3_alg_dble **ft,pauli_dble *m)
*     Computes the Pauli term using the field tensor ft, multiplies the
*     term by c and assigns the result to m[0] and m[1] (see the notes).
*
*   int sw_term(ptset_t set)
*     Computes the SW term for the current double-precision gauge field
*     and assigns the matrix to the global double-precision SW field. The
*     program inverts the matrices on the specified point set and returns
*     a non-zero value if some workspace allocations or matrix inversions
*     failed.
*
* The traditional expression for the SW term is
*
*  c(x0)+csw*(i/4)*sigma_{mu nu}*Fhat_{mu nu}(x),
*
* where
*
*  c(x0) = 4+m0+cF[0]-1     if x0=1 (open, SF or open-SF bc),
*          4+m0+cF[1]-1     if x0=NPROCO*L0-2 (open bc),
*                           or x0=NPROC0*L0-1 (SF or open-SF bc),
*          4+m0             otherwise,
*
*  sigma_{mu nu}=(i/2)*[gamma_mu,gamma_nu],
*
* and Fhat_{mu nu} is the standard (clover) expression for the gauge field
* tensor as computed by the program ftensor() [tcharge/ftensor.c]. The upper
* and lower 6x6 blocks of the matrix are stored in the pauli_dble structures
* swd[2*ix] and swd[2*ix+1], where ix is the label of the point x.
*
* If the alternative "exponential" expression is chosen for the SW term, the
* expression above gets replaced by
*
*  c(x0)*exp{[csw/(4+m0)]*(i/4)*sigma_{mu nu}*Fhat_{mu nu}(x)}.
*
* The quark mass m0, the improvement coefficients csw and cF as well as the
* flag that selects the type of SW term are obtained from the parameter data
* base by calling sw_parms() [flags/lat_parms.c].
*
* Along the boundaries of the lattice at global time
*
*  x0=0                (open, SF and open-SF boundary conditions),
*
*  x0=NPROC0*L0-1      (open boundary conditions),
*
* the SW term is set to unity. The program sw_term() checks the flags data
* base and computes only those parts of the SW field, which do not already
* have the correct values.
*
* The matrices m[0] and m[1] computed by pauli_term() are the upper and
* lower 6x6 submatrices on the diagonal of the matrix
*
*   -c*(i/2)*sigma_{mu,nu}*Fhat_{mu nu}
*
* at a given lattice point, assuming ft[0],..,ft[5] are the pointers to the
* (0,1),(0,2),(0,3),(2,3),(3,1) components of the field tensor Fhat_{mu nu}
* at this point.
*
* The order N returned by sw_order() is the order used by sw_term() if the
* exponential variant of the SW term is chosen. This is also the recommended
* order to be used when the quark forces are computed using the coefficients
* returned by sw_dexp() [swexp.c].
*
* The programs in this module are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define SW_TERM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "tcharge.h"
#include "sw_term.h"
#include "global.h"

#define N0 (NPROC0*L0)

static double c1,c2,c3[2];


int sw_order(void)
{
   int n;
   double a,b,c;
   sw_parms_t swp;

   swp=sw_parms();

   if (swp.m0!=DBL_MAX)
   {
      n=0;
      c=3.0*swp.csw/(4.0+swp.m0);
      a=c*exp(c);
      b=DBL_EPSILON;

      for (n=1;n<100;n++)
      {
         a*=c;
         b*=(double)(n+1);

         if (a<b)
            return n;
      }
   }

   error(1,1,"sw_order [sw_term.c]","SW parameters are out of range");

   return 0;
}


static void set_sw2unity(int n,pauli_dble *sw)
{
   int k;
   pauli_dble *sm;

   sm=sw+n;

   for (;sw<sm;sw++)
   {
      for (k=0;k<36;k++)
      {
         if (k<6)
            (*sw).u[k]=1.0;
         else
            (*sw).u[k]=0.0;
      }
   }
}


static void u3_alg2pauli1(u3_alg_dble *X,pauli_dble *m)
{
   (*m).u[10]=-(*X).c1;

   (*m).u[12]=-(*X).c5;
   (*m).u[13]= (*X).c4;
   (*m).u[14]=-(*X).c7;
   (*m).u[15]= (*X).c6;

   (*m).u[18]=-(*X).c5;
   (*m).u[19]=-(*X).c4;
   (*m).u[20]=-(*X).c2;

   (*m).u[22]=-(*X).c9;
   (*m).u[23]= (*X).c8;
   (*m).u[24]=-(*X).c7;
   (*m).u[25]=-(*X).c6;
   (*m).u[26]=-(*X).c9;
   (*m).u[27]=-(*X).c8;
   (*m).u[28]=-(*X).c3;
}


static void u3_alg2pauli2(u3_alg_dble *X,pauli_dble *m)
{
   (*m).u[11] =(*X).c1;
   (*m).u[12]+=(*X).c4;
   (*m).u[13]+=(*X).c5;
   (*m).u[14]+=(*X).c6;
   (*m).u[15]+=(*X).c7;

   (*m).u[18]-=(*X).c4;
   (*m).u[19]+=(*X).c5;

   (*m).u[21] =(*X).c2;
   (*m).u[22]+=(*X).c8;
   (*m).u[23]+=(*X).c9;
   (*m).u[24]-=(*X).c6;
   (*m).u[25]+=(*X).c7;
   (*m).u[26]-=(*X).c8;
   (*m).u[27]+=(*X).c9;

   (*m).u[29] =(*X).c3;
}


static void u3_alg2pauli3(u3_alg_dble *X,pauli_dble *m)
{
   (*m).u[ 0]=-(*X).c1;
   (*m).u[ 1]=-(*X).c2;
   (*m).u[ 2]=-(*X).c3;
   (*m).u[ 3]= (*X).c1;
   (*m).u[ 4]= (*X).c2;
   (*m).u[ 5]= (*X).c3;
   (*m).u[ 6]=-(*X).c5;
   (*m).u[ 7]= (*X).c4;
   (*m).u[ 8]=-(*X).c7;
   (*m).u[ 9]= (*X).c6;

   (*m).u[16]=-(*X).c9;
   (*m).u[17]= (*X).c8;

   (*m).u[30]= (*X).c5;
   (*m).u[31]=-(*X).c4;
   (*m).u[32]= (*X).c7;
   (*m).u[33]=-(*X).c6;
   (*m).u[34]= (*X).c9;
   (*m).u[35]=-(*X).c8;
}


void pauli_term(double c,u3_alg_dble **ft,pauli_dble *m)
{
   u3_alg_dble X;

   _u3_alg_mul_sub(X,c,ft[3][0],ft[0][0]);
   u3_alg2pauli1(&X,m);
   _u3_alg_mul_sub(X,c,ft[4][0],ft[1][0]);
   u3_alg2pauli2(&X,m);
   _u3_alg_mul_sub(X,c,ft[5][0],ft[2][0]);
   u3_alg2pauli3(&X,m);

   m+=1;

   _u3_alg_mul_add(X,c,ft[3][0],ft[0][0]);
   u3_alg2pauli1(&X,m);
   _u3_alg_mul_add(X,c,ft[4][0],ft[1][0]);
   u3_alg2pauli2(&X,m);
   _u3_alg_mul_add(X,c,ft[5][0],ft[2][0]);
   u3_alg2pauli3(&X,m);
}


static int set_swd(int N,int isw,int ofs,int ieo)
{
   int bc,ifail;
   int k,ix,t,n;
   double c,*u;
   pauli_dble *swb,*sw,*sm;
   u3_alg_dble **ft,*fr[6];
   pauli_wsp_t *pwsp;
   swexp_wsp_t *swsp;

   swb=swdfld();
   ft=ftensor();
   bc=bc_type();
   ifail=0;

#pragma omp parallel private(k,ix,t,n,c,u,sw,sm,fr,pwsp,swsp)   \
   reduction(| : ifail)
   {
      k=omp_get_thread_num();

      if (isw)
      {
         swsp=alloc_swexp_wsp(N);
         ifail|=(swsp==NULL);
      }
      else if (ieo)
      {
         pwsp=alloc_pauli_wsp();
         ifail|=(pwsp==NULL);
      }

      if (ifail==0)
      {
         n=ofs+k*(VOLUME_TRD/2);
         sw=swb+2*n;
         fr[0]=ft[0]+n;
         fr[1]=ft[1]+n;
         fr[2]=ft[2]+n;
         fr[3]=ft[3]+n;
         fr[4]=ft[4]+n;
         fr[5]=ft[5]+n;

         for (ix=0;ix<(VOLUME_TRD/2);ix++)
         {
            t=global_time(ix+n);

            if (((t==0)&&(bc!=3))||((t==(N0-1))&&(bc==0)))
            {
               set_sw2unity(2,sw);
               sw+=2;
            }
            else
            {
               pauli_term(c2,fr,sw);

               if ((t==1)&&(bc!=3))
                  c=c3[0];
               else if (((t==(N0-2))&&(bc==0))||
                        ((t==(N0-1))&&((bc==1)||(bc==2))))
                  c=c3[1];
               else
                  c=c1;

               sm=sw+2;

               for (;sw<sm;sw++)
               {
                  if (isw)
                  {
                     if (ieo)
                        sw_exp(1,sw,1.0/c,swsp,sw);
                     else
                        sw_exp(0,sw,c,swsp,sw);
                  }
                  else
                  {
                     u=(*sw).u;
                     u[0]+=c;
                     u[1]+=c;
                     u[2]+=c;
                     u[3]+=c;
                     u[4]+=c;
                     u[5]+=c;

                     if (ieo)
                        ifail|=inv_pauli_dble(0.0,sw,pwsp,sw);
                  }
               }
            }

            fr[0]+=1;
            fr[1]+=1;
            fr[2]+=1;
            fr[3]+=1;
            fr[4]+=1;
            fr[5]+=1;
         }

         if (isw)
            free_swexp_wsp(swsp);
         else if (ieo)
            free_pauli_wsp(pwsp);
      }
   }

   if ((NPROC>1)&&(isw==0)&&(ieo))
   {
      k=ifail;
      MPI_Allreduce(&k,&ifail,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
   }

   return ifail;
}


static int iswd(int ofs)
{
   int k,ifail;
   pauli_dble *swb,*sw,*sm;
   pauli_wsp_t *pwsp;

   swb=swdfld();
   ifail=0;

#pragma omp parallel private(k,sw,sm,pwsp) reduction(| : ifail)
   {
      k=omp_get_thread_num();
      pwsp=alloc_pauli_wsp();
      ifail|=(pwsp==NULL);

      if (ifail==0)
      {
         sw=swb+2*ofs+k*VOLUME_TRD;
         sm=sw+VOLUME_TRD;

         for (;sw<sm;sw++)
            ifail|=inv_pauli_dble(0.0,sw,pwsp,sw);

         free_pauli_wsp(pwsp);
      }
   }

   if (NPROC>1)
   {
      k=ifail;
      MPI_Allreduce(&k,&ifail,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
   }

   return ifail;
}


int sw_term(ptset_t set)
{
   int N,ie,io,isw,ifail,iprms[1];
   sw_parms_t swp;

   if (NPROC>1)
   {
      iprms[0]=(int)(set);
      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=(int)(set),1,"sw_term [sw_term.c]",
            "Parameter is not global");
   }

   swp=sw_parms();

   isw=swp.isw;
   c1=4.0+swp.m0;
   c2=-0.5*swp.csw;
   c3[0]=c1+swp.cF[0]-1.0;
   c3[1]=c1+swp.cF[1]-1.0;
   ifail=0;

   if (isw)
   {
      N=sw_order();
      c2/=c1;
   }
   else
      N=0;

   if (query_flags(SWD_UP2DATE)!=1)
   {
      if ((set==NO_PTS)||(set==ODD_PTS))
         (void)(set_swd(N,isw,0,0));
      else
         ifail|=set_swd(N,isw,0,1);

      if ((set==NO_PTS)||(set==EVEN_PTS))
         (void)(set_swd(N,isw,VOLUME/2,0));
      else
         ifail|=set_swd(N,isw,VOLUME/2,1);
   }
   else
   {
      ie=query_flags(SWD_E_INVERTED);
      io=query_flags(SWD_O_INVERTED);

      if ((ie==0)&&((set==ALL_PTS)||(set==EVEN_PTS)))
      {
         if (isw)
            (void)(set_swd(N,isw,0,1));
         else
            ifail|=iswd(0);
      }

      if ((ie==1)&&((set==NO_PTS)||(set==ODD_PTS)))
         (void)(set_swd(N,isw,0,0));

      if ((io==0)&&((set==ALL_PTS)||(set==ODD_PTS)))
      {
         if (isw)
            (void)(set_swd(N,isw,VOLUME/2,1));
         else
            ifail|=iswd(VOLUME/2);
      }

      if ((io==1)&&((set==NO_PTS)||(set==EVEN_PTS)))
         (void)(set_swd(N,isw,VOLUME/2,0));
   }

   set_flags(COMPUTED_SWD);

   if ((set==ALL_PTS)||(set==EVEN_PTS))
      set_flags(INVERTED_SWD_E);

   if ((set==ALL_PTS)||(set==ODD_PTS))
      set_flags(INVERTED_SWD_O);

   return ifail;
}
