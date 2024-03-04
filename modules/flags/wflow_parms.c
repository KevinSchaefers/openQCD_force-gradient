
/*******************************************************************************
*
* File wflow_parms.c
*
* Copyright (C) 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Wilson flow parameters.
*
*   wflow_parms_t set_wflow_parms(int rule,double eps,int ntot,int dnms,
*                                 int ntm,double *tm)
*     Sets the basic parameters of the gradient flow. The parameters are
*
*       rule        Integration rule (0: EULER, 1: RK2, 2: RK3).
*
*       eps         Integration step size. eps must be positive.
*
*       ntot        Total number of integration steps.
*
*       dnms        Separation of measurements along the flow in
*                   numbers of integration steps. dnms must divide
*                   ntot.
*
*       ntm         Number of special flow times.
*
*       tm          Array tm[0],..,tm[ntm-1] of special flow times.
*                   The times must be non-negative and monotonically
*                   increasing.
*
*     The program returns a structure containing the parameters listed
*     above.
*
*   wflow_parms_t wflow_parms(void)
*     Returns a structure containing the current values of the parameters
*     listed above.
*
*   void read_wflow_parms(char *section,int rflg)
*     Reads the Wilson flow parameters on MPI process 0 from the specified
*     parameter section on stdin. The tags are
*
*       integrator   EULER | RK2 | RK3
*       eps          <double>
*       ntot         <int>
*       dnms         <int>
*       tm           <double> [<double>]
*
*     The parameters ntot and dnms are read if the bit (rflg&0x1) is set
*     and the flow times tm[0],..,tm[ntm-1] if the bit (rflg&0x2) is set.
*     The tags rule and eps must always be present.
*
*   void print_wflow_parms(void)
*     Prints the Wilson flow  parameters to stdout on MPI process 0.
*
*   void write_wflow_parms(FILE *fdat)
*     Writes the Wilson flow parameters to the file fdat on MPI process 0.
*
*   void check_wflow_parms(FILE *fdat)
*     Compares the Wilson flow parameters with the values stored on the
*     file fdat on MPI process 0, assuming the latter were written to the
*     file by the program write_wflow_parms().
*
* The type wflow_parms_t is defined in the file flags.h.
*
* To ensure the consistency of the data base, the wflow parameters must be set
* simultaneously on all processes.
*
*******************************************************************************/

#define WFLOW_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

static wflow_parms_t wflow={0,0,0,0,0.0,NULL};


wflow_parms_t set_wflow_parms(int rule,double eps,int ntot,int dnms,
                              int ntm,double *tm)
{
   int i,ie,iprms[4];
   double dprms[1];

   ie=0;
   ie|=((rule<0)||(rule>3));
   ie|=(eps<=0.0);
   ie|=((ntot<0)||(dnms<0)||((dnms>0)&&(ntot%dnms)));
   ie|=(ntm<0);

   for (i=0;i<ntm;i++)
   {
      if (i==0)
         ie|=(tm[i]<0.0);
      else
         ie|=(tm[i]<tm[i-1]);
   }

   error_root(ie,1,"set_wflow_parms [wflow_parms.c]",
              "Parameters are out of range");

   if (NPROC>1)
   {
      iprms[0]=rule;
      iprms[1]=ntot;
      iprms[2]=dnms;
      iprms[3]=ntm;
      dprms[0]=eps;

      MPI_Bcast(iprms,4,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((iprms[0]!=rule)||(iprms[1]!=ntot)||(iprms[2]!=dnms)||
            (iprms[3]!=ntm)||(dprms[0]!=eps),1,
            "set_wflow_parms [wflow_parms.c]","Parameters are not global");

      ie=0;

      for (i=0;i<ntm;i++)
      {
         dprms[0]=tm[i];
         MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
         ie|=(dprms[0]!=tm[i]);
      }

      error(ie!=0,2,"set_wflow_parms [wflow_parms.c]",
            "Parameters are not global");
   }

   if (ntm>0)
   {
      wflow.tm=malloc(ntm*sizeof(double));
      error(wflow.tm==NULL,1,"set_wflow_parms [wflow_parms.c]",
            "Unable to allocate parameter array");
   }

   wflow.rule=rule;
   wflow.ntot=ntot;
   wflow.dnms=dnms;
   wflow.ntm=ntm;
   wflow.eps=eps;

   for (i=0;i<ntm;i++)
      wflow.tm[i]=tm[i];

   return wflow;
}


wflow_parms_t wflow_parms(void)
{
   return wflow;
}


void read_wflow_parms(char *section,int rflg)
{
   int my_rank,rule,ntot,dnms,ntm;
   double eps,*tm;
   char line[NAME_SIZE];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   rule=0;
   eps=0.0;
   ntot=0;
   dnms=0;
   ntm=0;

   if (my_rank==0)
   {
      find_section(section);

      read_line("integrator","%s",line);
      read_line("eps","%lf",&eps);

      if (rflg&0x1)
      {
         read_line("ntot","%d",&ntot);
         read_line("dnms","%d",&dnms);
      }

      if (rflg&0x2)
         ntm=count_tokens("tm");

      if (strcmp(line,"EULER")==0)
         rule=0;
      else if (strcmp(line,"RK2")==0)
         rule=1;
      else if (strcmp(line,"RK3")==0)
         rule=2;
      else
         error_root(1,1,"read_wflow_parms [wflow_parms.c]",
                    "Unknown integration rule");
   }

   if (NPROC>1)
   {
      MPI_Bcast(&rule,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&eps,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&ntot,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&dnms,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&ntm,1,MPI_INT,0,MPI_COMM_WORLD);
   }

   if (ntm)
   {
      tm=malloc(ntm*sizeof(*tm));
      error(tm==NULL,1,"read_wflow_parms [wflow_parms.c]",
            "Unable to allocate auxiliary array");

      if (my_rank==0)
         read_dprms("tm",ntm,tm);

      if (NPROC>1)
         MPI_Bcast(tm,ntm,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
      tm=NULL;

   (void)(set_wflow_parms(rule,eps,ntot,dnms,ntm,tm));

   if (ntm)
      free(tm);
}


void print_wflow_parms(void)
{
   int my_rank,n,i;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      printf("Wilson flow parameters:\n");

      if (wflow.rule==0)
         printf("Euler integrator\n");
      else if (wflow.rule==1)
         printf("2nd order RK integrator\n");
      else
         printf("3rd order RK integrator\n");

      n=fdigits(wflow.eps);
      printf("eps = %.*f\n",IMAX(n,1),wflow.eps);

      if (wflow.ntot)
      {
            printf("ntot = %d\n",wflow.ntot);
            printf("dnms = %d\n",wflow.dnms);
      }

      if (wflow.ntm)
      {
         n=fdigits(wflow.tm[0]);
         printf("tm = %.*f",IMAX(n,1),wflow.tm[0]);

         for (i=1;i<wflow.ntm;i++)
         {
            n=fdigits(wflow.tm[i]);
            printf(" %.*f",IMAX(n,1),wflow.tm[i]);
         }

         printf("\n");
      }

      printf("\n");
   }
}


void write_wflow_parms(FILE *fdat)
{
   int my_rank,endian;
   int ntm,i,iw;
   stdint_t istd[4];
   double *dstd;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if (my_rank==0)
   {
      istd[0]=(stdint_t)(wflow.rule);
      istd[1]=(stdint_t)(wflow.ntot);
      istd[2]=(stdint_t)(wflow.dnms);
      istd[3]=(stdint_t)(wflow.ntm);

      ntm=wflow.ntm;
      dstd=malloc((ntm+1)*sizeof(*dstd));
      error_root(dstd==NULL,1,"write_wflow_parms [wflow_parms.c]",
                 "Unable to allocate auxiliary array");
      dstd[0]=wflow.eps;

      for (i=0;i<ntm;i++)
         dstd[1+i]=wflow.tm[i];

      if (endian==BIG_ENDIAN)
      {
         bswap_int(4,istd);
         bswap_double(ntm+1,dstd);
      }

      iw=fwrite(istd,sizeof(stdint_t),4,fdat);
      iw+=fwrite(dstd,sizeof(double),ntm+1,fdat);
      error_root(iw!=(ntm+5),1,"write_wflow_parms [wflow_parms.c]",
                 "Incorrect write count");
      free(dstd);
   }
}


void check_wflow_parms(FILE *fdat)
{
   int my_rank,endian;
   int ntm,i,ie,ir;
   stdint_t istd[4];
   double *dstd;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if (my_rank==0)
   {
      ntm=wflow.ntm;
      dstd=malloc((ntm+1)*sizeof(*dstd));
      error_root(dstd==NULL,1,"check_wflow_parms [wflow_parms.c]",
                 "Unable to allocate auxiliary array");

      ir=fread(istd,sizeof(stdint_t),4,fdat);
      ir+=fread(dstd,sizeof(double),ntm+1,fdat);

      error_root(ir!=(ntm+5),1,"check_wflow_parms [wflow_parms.c]",
                 "Incorrect read count");

      if (endian==BIG_ENDIAN)
      {
         bswap_int(4,istd);
         bswap_double(ntm+1,dstd);
      }

      ie=0;
      ie|=(istd[0]!=(stdint_t)(wflow.rule));
      ie|=(istd[1]!=(stdint_t)(wflow.ntot));
      ie|=(istd[2]!=(stdint_t)(wflow.dnms));
      ie|=(istd[3]!=(stdint_t)(wflow.ntm));

      for (i=0;i<ntm;i++)
         ie|=(dstd[1+i]!=wflow.tm[i]);

      error_root(ie!=0,1,"check_wflow_parms [wflow_parms.c]",
                 "Parameters do not match");
      free(dstd);
   }
}
