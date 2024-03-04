
/*******************************************************************************
*
* File mdint_parms.c
*
* Copyright (C) 2011, 2012, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Molecular-dynamics integrator data base.
*
*   mdint_parms_t set_mdint_parms(int ilv,integrator_t integrator,double lambda,
*                                 int nstep,int nfr,int *ifr)
*     Sets the parameters of the molecular-dynamics integrator at level
*     ilv and returns a structure containing them (see the notes).
*
*   mdint_parms_t mdint_parms(int ilv)
*     Returns a structure containing the parameters of the integrator at
*     level ilv (see the notes).
*
*   void read_mdint_parms(int ilv)
*     Reads the parameter section [Level <int>] on MPI process 0 from
*     stdin, where <int> is set to the integrator level ilv. The expected
*     tags are
*
*       integrator   <integrator_t>
*       lambda       <double>
*       nstep        <int>
*       forces       <int> [<int>]
*
*     where "lambda" is required only if the specified integrator is the
*     2nd order OMF integrator. The line tagged "forces" must contain the
*     indices of the forces (separated by white space) to be integrated
*     at this level.
*
*   void read_all_mdint_parms(void)
*     Retrieves the number nlv of integrator levels from the SMD or HMC
*     parameter data base and calls read_mdint_parms(ilv) for all levels
*     ilv=0,..,nlv-1. An error occurs if neither the SMD nor the HMC
*     parameters are set.
*
*   void print_mdint_parms(void)
*     Prints the parameters of the defined integrator levels to stdout
*     on MPI process 0.
*
*   void write_mdint_parms(FILE *fdat)
*     Writes the parameters of the defined integrator levels to the file
*     fdat on MPI process 0.
*
*   void check_mdint_parms(FILE *fdat)
*     Compares the parameters of the defined integrator levels with those
*     stored on the file fdat on MPI process 0, assuming the latter were
*     written to the file by the program write_mdint_parms().
*
* A structure of type mdint_parms_t contains the parameters of a hierarchical
* molecular-dynamics integrator at a specified level (see update/README.mdint).
* Its elements are
*
*   integrator   Elementary integrator used. This parameter is an enum
*                type with one of the following values:
*
*                  LPFR     Leapfrog integrator
*
*                  OMF2     2nd order Omelyan-Mryglod-Folk integrator
*
*                  OMF4     4th order Omelyan-Mryglod-Folk integrator
*
*                   the other integrators are Hessian-free force-gradient integrators which are specified by alternating
*                   momentum updates (B) or Hessian-free force-gradient updates (D) and position updates (A).
*                   The coefficients are taken from the Hessian-free FGI paper (Schaefers et al., 2024)
*
*   lambda       Parameter of the 2nd order OMF integrator
*
*   nstep        Number of times the elementary integrator is applied
*                at this level
*
*   nfr          Number of forces integrated at this level
*
*   ifr          Force indices ifr[i] (i=0,..,nfr-1)
*
* The parameter lambda is not used in the case of the leapfrog and the 4th
* order OMF integrator. Up to 32 integrator levels, labeled by an index
* ilv=0,1,..,31, can be specified.
*
* An example of valid section in an input file which can be read by calling
* read_mdint(3) is
*
*  [Level 3]
*  integrator OMF2
*  lambda     0.2
*  nstep      12
*  forces     2 4 5
*
* In this case, there are three forces with index 2, 4 and 5.
*
* The programs set_mdint_parms() and read_mdint_parms() perform global
* operations and must be called simultaneously on all MPI processes.
*
*******************************************************************************/

#define MDINT_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

#define ILVMAX 32

static int init=0;
static mdint_parms_t mdp[ILVMAX+1]={{INTEGRATORS,0.0,0,0,NULL}};


static void init_mdp(void)
{
   int i;

   for (i=1;i<=ILVMAX;i++)
      mdp[i]=mdp[0];

   init=1;
}


static int *alloc_ifr(int ilv,int nfr)
{
   int *ifr,*old;

   if (nfr!=mdp[ilv].nfr)
   {
      if (nfr>0)
      {
         ifr=malloc(nfr*sizeof(*ifr));
         error(ifr==NULL,1,"alloc_ifr [mdint_parms.c]",
               "Unable to allocate index array");
      }
      else
         ifr=NULL;

      old=mdp[ilv].ifr;
      mdp[ilv].nfr=nfr;
      mdp[ilv].ifr=ifr;
   }
   else
      old=NULL;

   return old;
}


mdint_parms_t set_mdint_parms(int ilv,integrator_t integrator,double lambda,
                              int nstep,int nfr,int *ifr)
{
   int i,j,ie,*old,iprms[4];
   double dprms[1];

   if (init==0)
      init_mdp();

   if (integrator!=OMF2)
      lambda=0.0;

   if (NPROC>1)
   {
      iprms[0]=ilv;
      iprms[1]=(int)(integrator);
      iprms[2]=nstep;
      iprms[3]=nfr;
      dprms[0]=lambda;

      MPI_Bcast(iprms,4,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      ie=0;
      ie|=(iprms[0]!=ilv);
      ie|=(iprms[1]!=(int)(integrator));
      ie|=(iprms[2]!=nstep);
      ie|=(iprms[3]!=nfr);
      ie|=(dprms[0]!=lambda);

      for (i=0;i<nfr;i++)
      {
         iprms[0]=ifr[i];

         MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

         ie|=(iprms[0]!=ifr[i]);
      }

      error(ie!=0,1,"set_mdint_parms [mdint_parms.c]",
            "Parameters are not global");
   }

   ie=0;
   ie|=(ilv<0)||(ilv>=ILVMAX);
   ie|=(integrator==INTEGRATORS);
   ie|=(nstep<1);
   ie|=(nfr<1);

   for (i=0;i<nfr;i++)
      ie|=(ifr[i]<0);

   error_root(ie!=0,1,"set_mdint_parms [mdint_parms.c]",
              "Parameters are out of range");

   for (i=0;i<nfr;i++)
   {
      for (j=(i+1);j<nfr;j++)
         ie|=(ifr[i]==ifr[j]);
   }

   error_root(ie!=0,1,"set_mdint_parms [mdint_parms.c]",
              "Dublicate force indices");

   old=alloc_ifr(ilv,nfr);
   mdp[ilv].integrator=integrator;
   mdp[ilv].lambda=lambda;
   mdp[ilv].nstep=nstep;

   for (i=0;i<nfr;i++)
      mdp[ilv].ifr[i]=ifr[i];

   if (old!=NULL)
      free(old);

   return mdp[ilv];
}


mdint_parms_t mdint_parms(int ilv)
{
   if (init==0)
      init_mdp();

   if ((ilv>=0)&&(ilv<ILVMAX))
      return mdp[ilv];
   else
   {
      error_loc(1,1,"mdint_parms [mdint_parms.c]",
                "Level index is out of range");
      return mdp[ILVMAX];
   }
}


void read_mdint_parms(int ilv)
{
   int my_rank,idi;
   int nstep,nfr,*ifr;
   double lambda;
   char line[NAME_SIZE];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      sprintf(line,"Level %d",ilv);
      find_section(line);

      read_line("integrator","%s",line);

      if (strcmp(line,"LPFR")==0)
      {
         idi=0;
         lambda=0.0;
      }
      else if (strcmp(line,"OMF2")==0)
      {
         idi=1;
         read_line("lambda","%lf",&lambda);
      }
      else if (strcmp(line,"OMF4")==0)
      {
         idi=2;
         lambda=0.0;
      }
      else if (strcmp(line,"BAB")==0)
      {
          idi=3;
          lambda = 0.0;
      }
      else if (strcmp(line,"ABA")==0)
      {
          idi=4;
          lambda = 0.0;
      }
      else if (strcmp(line,"DAD")==0)
      {
          idi=5;
          lambda = 0.0;
      }
      else if (strcmp(line,"ADA")==0)
      {
          idi=6;
          lambda = 0.0;
      }
      else if (strcmp(line,"BABAB")==0)
      {
          idi=7;
          lambda = 0.0;
      }
      else if (strcmp(line,"ABABA")==0)
      {
          idi=8;
          lambda = 0.0;
      }
      else if (strcmp(line,"BADAB")==0)
      {
          idi=9;
          lambda = 0.0;
      }
      else if (strcmp(line,"DABAD")==0)
      {
          idi=10;
          lambda = 0.0;
      }
      else if (strcmp(line,"DADAD")==0)
      {
          idi=11;
          lambda = 0.0;
      }
      else if (strcmp(line,"ADADA")==0)
      {
          idi=12;
          lambda = 0.0;
      }
      else if (strcmp(line,"ABABABA")==0)
      {
          idi=13;
          lambda = 0.0;
      }
      else if (strcmp(line,"BABABAB")==0)
      {
          idi=14;
          lambda = 0.0;
      }
      else if (strcmp(line,"ABADABA")==0)
      {
          idi=15;
          lambda = 0.0;
      }
      else if (strcmp(line,"DABABAD")==0)
      {
          idi=16;
          lambda = 0.0;
      }
      else if (strcmp(line,"BADADAB")==0)
      {
          idi=17;
          lambda = 0.0;
      }
      else if (strcmp(line,"ADABADA")==0)
      {
          idi=18;
          lambda = 0.0;
      }
      else if (strcmp(line,"ADADADA")==0)
      {
          idi=19;
          lambda = 0.0;
      }
      else if (strcmp(line,"DADADAD")==0)
      {
          idi=20;
          lambda = 0.0;
      }
      else if (strcmp(line,"ABABABABA")==0)
      {
          idi=21;
          lambda = 0.0;
      }
      else if (strcmp(line,"BABABABAB")==0)
      {
          idi=22;
          lambda = 0.0;
      }
      else if (strcmp(line,"BABADABAB")==0)
      {
          idi=23;
          lambda = 0.0;
      }
      else if (strcmp(line,"DABABABAD")==0)
      {
          idi=24;
          lambda = 0.0;
      }
      else if (strcmp(line,"BADABADAB")==0)
      {
          idi=25;
          lambda = 0.0;
      }
      else if (strcmp(line,"DABADABAD")==0)
      {
          idi=26;
          lambda = 0.0;
      }
      else if (strcmp(line,"ABADADABA")==0)
      {
          idi=27;
          lambda = 0.0;
      }
      else if (strcmp(line,"ADABABADA")==0)
      {
          idi=28;
          lambda = 0.0;
      }
      else if (strcmp(line,"DADABADAD")==0)
      {
          idi=29;
          lambda = 0.0;
      }
      else if (strcmp(line,"ADADADADA")==0)
      {
          idi=30;
          lambda = 0.0;
      }
      else if (strcmp(line,"BADADADAB")==0)
      {
          idi=31;
          lambda = 0.0;
      }
      else if (strcmp(line,"BABABABABAB")==0)
      {
          idi=32;
          lambda = 0.0;
      }
      else if (strcmp(line,"ABABABABABA")==0)
      {
          idi=33;
          lambda = 0.0;
      }
      else if (strcmp(line,"ABABADABABA")==0)
      {
          idi=34;
          lambda = 0.0;
      }
      else if (strcmp(line,"DABABABABAD")==0)
      {
          idi=35;
          lambda = 0.0;
      }
      else if (strcmp(line,"ABADABADABA")==0)
      {
          idi=36;
          lambda = 0.0;
      }
      else if (strcmp(line,"BADABABADAB")==0)
      {
          idi=37;
          lambda = 0.0;
      }
      else if (strcmp(line,"ADABABABADA")==0)
      {
          idi=38;
          lambda = 0.0;
      }
      else if (strcmp(line,"BABADADABAB")==0)
      {
          idi=39;
          lambda = 0.0;
      }
      else if (strcmp(line,"ADABADABADA")==0)
      {
          idi=40;
          lambda = 0.0;
      }
      else if (strcmp(line,"DABADADABAD")==0)
      {
          idi=41;
          lambda = 0.0;
      }
      else if (strcmp(line,"DADABABADAD")==0)
      {
          idi=42;
          lambda = 0.0;
      }
      else if (strcmp(line,"ADADABADADA")==0)
      {
          idi=43;
          lambda = 0.0;
      }
      else if (strcmp(line,"BADADADADAB")==0)
      {
          idi=44;
          lambda = 0.0;
      }
      else if (strcmp(line,"ADADADADADA")==0)
      {
          idi=45;
          lambda = 0.0;
      }
      else if (strcmp(line,"BABABABABABABAB")==0)
      {
          idi=46;
          lambda = 0.0;
      }
      else if (strcmp(line,"ABABABABABABABA")==0)
      {
          idi=47;
          lambda = 0.0;
      }
      else
         error_root(1,1,"read_mdint_parms [mdint_parms.c]",
                    "Unknown integrator %s",line);

      read_line("nstep","%d",&nstep);
      error_root(nstep<1,1,"read_mdint [mdint_parms.c]",
                 "Parameter nstep out of range");

      nfr=count_tokens("forces");
      error_root(nfr==0,1,"read_mdint [mdint_parms.c]",
                 "No forces specified");
   }

   MPI_Bcast(&idi,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&lambda,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&nstep,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nfr,1,MPI_INT,0,MPI_COMM_WORLD);

   ifr=malloc(nfr*sizeof(*ifr));
   error(ifr==NULL,1,"read_mdint [mdint_parms.c]",
         "Unable to allocate index array");
   if (my_rank==0)
      read_iprms("forces",nfr,ifr);

   MPI_Bcast(ifr,nfr,MPI_INT,0,MPI_COMM_WORLD);

   if (idi==0)
      set_mdint_parms(ilv,LPFR,lambda,nstep,nfr,ifr);
   else if (idi==1)
      set_mdint_parms(ilv,OMF2,lambda,nstep,nfr,ifr);
   else if (idi==2)
      set_mdint_parms(ilv,OMF4,lambda,nstep,nfr,ifr);
   else if (idi==3)
      set_mdint_parms(ilv,BAB,lambda,nstep,nfr,ifr);
   else if (idi==4)
      set_mdint_parms(ilv,ABA,lambda,nstep,nfr,ifr);
   else if (idi==5)
      set_mdint_parms(ilv,DAD,lambda,nstep,nfr,ifr);
   else if (idi==6)
      set_mdint_parms(ilv,ADA,lambda,nstep,nfr,ifr);
   else if (idi==7)
      set_mdint_parms(ilv,BABAB,lambda,nstep,nfr,ifr);
   else if (idi==8)
      set_mdint_parms(ilv,ABABA,lambda,nstep,nfr,ifr);
   else if (idi==9)
      set_mdint_parms(ilv,BADAB,lambda,nstep,nfr,ifr);
   else if (idi==10)
      set_mdint_parms(ilv,DABAD,lambda,nstep,nfr,ifr);
   else if (idi==11)
      set_mdint_parms(ilv,DADAD,lambda,nstep,nfr,ifr);
   else if (idi==12)
      set_mdint_parms(ilv,ADADA,lambda,nstep,nfr,ifr);
   else if (idi==13)
      set_mdint_parms(ilv,ABABABA,lambda,nstep,nfr,ifr);
   else if (idi==14)
      set_mdint_parms(ilv,BABABAB,lambda,nstep,nfr,ifr);
   else if (idi==15)
      set_mdint_parms(ilv,ABADABA,lambda,nstep,nfr,ifr);
   else if (idi==16)
      set_mdint_parms(ilv,DABABAD,lambda,nstep,nfr,ifr);
   else if (idi==17)
      set_mdint_parms(ilv,BADADAB,lambda,nstep,nfr,ifr);
   else if (idi==18)
      set_mdint_parms(ilv,ADABADA,lambda,nstep,nfr,ifr);
   else if (idi==19)
      set_mdint_parms(ilv,ADADADA,lambda,nstep,nfr,ifr);
   else if (idi==20)
      set_mdint_parms(ilv,DADADAD,lambda,nstep,nfr,ifr);
   else if (idi==21)
      set_mdint_parms(ilv,ABABABABA,lambda,nstep,nfr,ifr);
   else if (idi==22)
      set_mdint_parms(ilv,BABABABAB,lambda,nstep,nfr,ifr);
   else if (idi==23)
      set_mdint_parms(ilv,BABADABAB,lambda,nstep,nfr,ifr);
   else if (idi==24)
      set_mdint_parms(ilv,DABABABAD,lambda,nstep,nfr,ifr);
   else if (idi==25)
      set_mdint_parms(ilv,BADABADAB,lambda,nstep,nfr,ifr);
   else if (idi==26)
      set_mdint_parms(ilv,DABADABAD,lambda,nstep,nfr,ifr);
   else if (idi==27)
      set_mdint_parms(ilv,ABADADABA,lambda,nstep,nfr,ifr);
   else if (idi==28)
      set_mdint_parms(ilv,ADABABADA,lambda,nstep,nfr,ifr);
   else if (idi==29)
      set_mdint_parms(ilv,DADABADAD,lambda,nstep,nfr,ifr);
   else if (idi==30)
      set_mdint_parms(ilv,ADADADADA,lambda,nstep,nfr,ifr);
   else if (idi==31)
      set_mdint_parms(ilv,BADADADAB,lambda,nstep,nfr,ifr);
   else if (idi==32)
      set_mdint_parms(ilv,BABABABABAB,lambda,nstep,nfr,ifr);
   else if (idi==33)
      set_mdint_parms(ilv,ABABABABABA,lambda,nstep,nfr,ifr);
   else if (idi==34)
      set_mdint_parms(ilv,ABABADABABA,lambda,nstep,nfr,ifr);
   else if (idi==35)
      set_mdint_parms(ilv,DABABABABAD,lambda,nstep,nfr,ifr);
   else if (idi==36)
      set_mdint_parms(ilv,ABADABADABA,lambda,nstep,nfr,ifr);
   else if (idi==37)
      set_mdint_parms(ilv,BADABABADAB,lambda,nstep,nfr,ifr);
   else if (idi==38)
      set_mdint_parms(ilv,ADABABABADA,lambda,nstep,nfr,ifr);
   else if (idi==39)
      set_mdint_parms(ilv,BABADADABAB,lambda,nstep,nfr,ifr);
   else if (idi==40)
      set_mdint_parms(ilv,ADABADABADA,lambda,nstep,nfr,ifr);
   else if (idi==41)
      set_mdint_parms(ilv,DABADADABAD,lambda,nstep,nfr,ifr);
   else if (idi==42)
      set_mdint_parms(ilv,DADABABADAD,lambda,nstep,nfr,ifr);
   else if (idi==43)
      set_mdint_parms(ilv,ADADABADADA,lambda,nstep,nfr,ifr);
   else if (idi==44)
      set_mdint_parms(ilv,BADADADADAB,lambda,nstep,nfr,ifr);
   else if (idi==45)
      set_mdint_parms(ilv,ADADADADADA,lambda,nstep,nfr,ifr);
   else if (idi==46)
      set_mdint_parms(ilv,BABABABABABABAB,lambda,nstep,nfr,ifr);
   else if (idi==47)
      set_mdint_parms(ilv,ABABABABABABABA,lambda,nstep,nfr,ifr);

   free(ifr);
}


void read_all_mdint_parms(void)
{
   int nlv,ilv;
   hmc_parms_t hmc;
   smd_parms_t smd;

   hmc=hmc_parms();
   smd=smd_parms();
   error_root((hmc.nlv==0)&&(smd.nlv==0),1,
              "read_all_mdint_parms [mdint_parms.c]",
              "SMD or HMC parameters must first be set");

   if (hmc.nlv)
      nlv=hmc.nlv;
   else
      nlv=smd.nlv;

   for (ilv=0;ilv<nlv;ilv++)
      read_mdint_parms(ilv);
}


void print_mdint_parms(void)
{
   int my_rank,i,j,n;
   int nfr,*ifr;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if ((my_rank==0)&&(init==1))
   {
      for (i=0;i<ILVMAX;i++)
      {
         if (mdp[i].integrator!=INTEGRATORS)
         {
            printf("Level %d:\n",i);

            if (mdp[i].integrator==LPFR)
               printf("Leapfrog integrator (BAB)\n");
            else if (mdp[i].integrator==OMF2)
            {
               n=fdigits(mdp[i].lambda);
               printf("2nd order OMF integrator (BABAB) with lambda = %.*f\n",
                      IMAX(n,1),mdp[i].lambda);
            }
            else if (mdp[i].integrator==OMF4)
               printf("4th order OMF integrator (BABABABABAB)\n");
            else if (mdp[i].integrator==BAB)
               printf("2nd order non-gradient integrator BAB\n");
            else if (mdp[i].integrator==ABA)
                printf("2nd order non-gradient integrator ABA\n");
            else if (mdp[i].integrator==DAD)
                printf("2nd order Hessian-free force-gradient integrator DAD\n");
            else if (mdp[i].integrator==ADA)
                printf("2nd order Hessian-free force-gradient integrator ADA\n");
            else if (mdp[i].integrator==BABAB)
                printf("2nd order non-gradient integrator BABAB\n");
            else if (mdp[i].integrator==ABABA)
                printf("2nd order non-gradient integrator ABABA\n");
            else if (mdp[i].integrator==BADAB)
                printf("4th order Hessian-free force-gradient integrator BADAB\n");
            else if (mdp[i].integrator==DABAD)
                printf("4th order Hessian-free force-gradient integrator DABAD\n");
            else if (mdp[i].integrator==DADAD)
                printf("4th order Hessian-free force-gradient integrator DADAD\n");
            else if (mdp[i].integrator==ADADA)
                printf("4th order Hessian-free force-gradient integrator ADADA\n");
            else if (mdp[i].integrator==ABABABA)
                printf("4th order non-gradient integrator ABABABA\n");
            else if (mdp[i].integrator==BABABAB)
                printf("4th order non-gradient integrator BABABAB\n");
            else if (mdp[i].integrator==ABADABA)
                printf("4th order Hessian-free force-gradient integrator ABADABA\n");
            else if (mdp[i].integrator==DABABAD)
                printf("4th order Hessian-free force-gradient integrator DABABAD\n");
            else if (mdp[i].integrator==BADADAB)
                printf("4th order Hessian-free force-gradient integrator BADADAB\n");
            else if (mdp[i].integrator==ADABADA)
                printf("4th order Hessian-free force-gradient integrator ADABADA\n");
            else if (mdp[i].integrator==ADADADA)
                printf("4th order Hessian-free force-gradient integrator ADADADA\n");
            else if (mdp[i].integrator==DADADAD)
                printf("4th order Hessian-free force-gradient integrator DADADAD\n");
            else if (mdp[i].integrator==ABABABABA)
                printf("4th order non-gradient integrator ABABABABA\n");
            else if (mdp[i].integrator==BABABABAB)
                printf("4th order non-gradient integrator BABABABAB\n");
            else if (mdp[i].integrator==BABADABAB)
                printf("4th order Hessian-free force-gradient integrator BABADABAB\n");
            else if (mdp[i].integrator==DABABABAD)
                printf("4th order Hessian-free force-gradient integrator DABABABAD\n");
            else if (mdp[i].integrator==BADABADAB)
                printf("4th order Hessian-free force-gradient integrator BADABADAB\n");
            else if (mdp[i].integrator==DABADABAD)
                printf("4th order Hessian-free force-gradient integrator DABADABAD\n");
            else if (mdp[i].integrator==ABADADABA)
                printf("4th order Hessian-free force-gradient integrator ABADADABA\n");
            else if (mdp[i].integrator==ADABABADA)
                printf("4th order Hessian-free force-gradient integrator ADABABADA\n");
            else if (mdp[i].integrator==DADABADAD)
                printf("4th order Hessian-free force-gradient integrator DADABADAD\n");
            else if (mdp[i].integrator==ADADADADA)
                printf("4th order Hessian-free force-gradient integrator ADADADADA\n");
            else if (mdp[i].integrator==BADADADAB)
                printf("6th order Hessian-free force-gradient integrator BADADADAB\n");
            else if (mdp[i].integrator==BABABABABAB)
                printf("4th order non-gradient integrator BABABABABAB\n");
            else if (mdp[i].integrator==ABABABABABA)
                printf("4th order non-gradient integrator ABABABABABA\n");
            else if (mdp[i].integrator==ABABADABABA)
                printf("4th order Hessian-free force-gradient integrator ABABADABABA\n");
            else if (mdp[i].integrator==DABABABABAD)
                printf("4th order Hessian-free force-gradient integrator DABABABABAD\n");
            else if (mdp[i].integrator==ABADABADABA)
                printf("4th order Hessian-free force-gradient integrator ABADABADABA\n");
            else if (mdp[i].integrator==BADABABADAB)
                printf("4th order Hessian-free force-gradient integrator BADABABADAB\n");
            else if (mdp[i].integrator==ADABABABADA)
                printf("4th order Hessian-free force-gradient integrator ADABABABADA\n");
            else if (mdp[i].integrator==BABADADABAB)
                printf("4th order Hessian-free force-gradient integrator BABADADABAB\n");
            else if (mdp[i].integrator==ADABADABADA)
                printf("4th order Hessian-free force-gradient integrator ADABADABADA\n");
            else if (mdp[i].integrator==DABADADABAD)
                printf("4th order Hessian-free force-gradient integrator DABADADABAD\n");
            else if (mdp[i].integrator==DADABABADAD)
                printf("4th order Hessian-free force-gradient integrator DADABABADAD\n");
            else if (mdp[i].integrator==ADADABADADA)
                printf("4th order Hessian-free force-gradient integrator ADADABADADA\n");
            else if (mdp[i].integrator==BADADADADAB)
                printf("6th order Hessian-free force-gradient integrator BADADADADAB\n");
            else if (mdp[i].integrator==ADADADADADA)
                printf("6th order Hessian-free force-gradient integrator ADADADADADA\n");
            else if (mdp[i].integrator==BABABABABABABAB)
               printf("6th order non-gradient 15-stage decomposition (velocity version)\n");
            else if (mdp[i].integrator==ABABABABABABABA)
                printf("6th order non-gradient 15-stage decomposition (position version)\n");
            else
               printf("Unknown integrator\n");

            printf("Number of steps = %d\n",mdp[i].nstep);

            nfr=mdp[i].nfr;
            ifr=mdp[i].ifr;

            if (nfr>0)
            {
               printf("Forces =");

               for (j=0;j<nfr;j++)
                  printf(" %d",ifr[j]);

               printf("\n");
            }

            printf("\n");
         }
      }
   }
}


void write_mdint_parms(FILE *fdat)
{
   int my_rank,endian;
   int nfr,nmx,n,iw,i,j;
   stdint_t *istd;
   double dstd[1];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if ((my_rank==0)&&(init==1))
   {
      nmx=0;

      for (i=0;i<ILVMAX;i++)
      {
         nfr=mdp[i].nfr;
         if (nfr>nmx)
            nmx=nfr;
      }

      istd=malloc((nmx+4)*sizeof(stdint_t));
      error_root(istd==NULL,1,"write_mdint_parms [mdint_parms.c]",
                 "Unable to allocate auxiliary array");

      for (i=0;i<ILVMAX;i++)
      {
         if (mdp[i].integrator!=INTEGRATORS)
         {
            nfr=mdp[i].nfr;

            istd[0]=(stdint_t)(i);
            istd[1]=(stdint_t)(mdp[i].integrator);
            istd[2]=(stdint_t)(mdp[i].nstep);
            istd[3]=(stdint_t)(mdp[i].nfr);

            for (j=0;j<nfr;j++)
               istd[4+j]=(stdint_t)(mdp[i].ifr[j]);

            dstd[0]=mdp[i].lambda;
            n=4+nfr;

            if (endian==BIG_ENDIAN)
            {
               bswap_int(n,istd);
               bswap_double(1,dstd);
            }

            iw=fwrite(istd,sizeof(stdint_t),n,fdat);
            iw+=fwrite(dstd,sizeof(double),1,fdat);
            error_root(iw!=(n+1),1,"write_mdint_parms [mdint_parms.c]",
                       "Incorrect write count");
         }
      }

      free(istd);
   }
}


void check_mdint_parms(FILE *fdat)
{
   int my_rank,endian;
   int nfr,nmx,n,ir,ie,i,j;
   stdint_t *istd;
   double dstd[1];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if ((my_rank==0)&&(init==1))
   {
      ie=0;
      nmx=0;

      for (i=0;i<ILVMAX;i++)
      {
         nfr=mdp[i].nfr;
         if (nfr>nmx)
            nmx=nfr;
      }

      istd=malloc((nmx+4)*sizeof(stdint_t));
      error_root(istd==NULL,1,"check_mdint_parms [mdint_parms.c]",
                 "Unable to allocate auxiliary array");

      for (i=0;i<ILVMAX;i++)
      {
         if (mdp[i].integrator!=INTEGRATORS)
         {
            nfr=mdp[i].nfr;
            n=4+nfr;

            ir=fread(istd,sizeof(stdint_t),n,fdat);
            ir+=fread(dstd,sizeof(double),1,fdat);
            error_root(ir!=(n+1),1,"check_mdint_parms [mdint_parms.c]",
                       "Incorrect read count");

            if (endian==BIG_ENDIAN)
            {
               bswap_int(n,istd);
               bswap_double(1,dstd);
            }

            ie|=(istd[0]!=(stdint_t)(i));
            ie|=(istd[1]!=(stdint_t)(mdp[i].integrator));
            ie|=(istd[2]!=(stdint_t)(mdp[i].nstep));
            ie|=(istd[3]!=(stdint_t)(mdp[i].nfr));

            for (j=0;j<nfr;j++)
               ie|=(istd[4+j]!=(stdint_t)(mdp[i].ifr[j]));

            ie|=(dstd[0]!=mdp[i].lambda);
         }
      }

      error_root(ie!=0,1,"check_mdint_parms [mdint_parms.c]",
                 "Parameters do not match");
      free(istd);
   }
}
