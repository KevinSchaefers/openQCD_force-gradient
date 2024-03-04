
/*******************************************************************************
*
* File counters.c
*
* Copyright (C) 2011-2018, 2020, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Solver iteration counters.
*
*   void setup_counters(void)
*     Creates the counters required for counting the solver iteration
*     numbers. The solvers for the Dirac equation used by the simulation
*     algorithm are inferred from the parameter data base.
*
*   void clear_counters(void)
*     Sets all counters to zero.
*
*   void add2counter(char *type,int idx,int *status)
*     Adds the status numbers "status" to the counter characterized by
*     "type" and index "idx" (see the notes).
*
*   int get_count(char *type,int idx,int *status)
*     Returns the number of times add2counter(type,idx,status) has been
*     called since the counter was last cleared. On exit the program
*     assigns the sum of the accumulated status values to the argument
*     status [the meaning of the arguments is otherwise the same as in
*     the case of add2counter()].
*
*   void print_avgstat(char *type,int idx)
*     Prints the average status values of the counter specified by "type"
*     and "idx" [as in add2counter()] to stdout on MPI process 0.
*
*   void print_all_avgstat(void)
*     Prints the average status values of all known counters to stdout on
*     MPI process 0.
*
* In most cases, the computation of the fermion actions and forces requires
* the Dirac equation to be solved a number of times. Depending on the solver
* used, the number of status values returned by the solver program may vary.

* The counters administered by this module are set up for all fermion actions
* and forces taking part in the simulation according to the parameter data
* base. In addition, the iteration numbers required for the solution of the
* little Dirac equation in the course of the generation and updates of the
* deflation subspace are monitored. The counters can only be set up once.
*
* The available counter types are "action", "field", "force" and "modes". In
* the first three cases, the index idx passed to add2counter() is the one of
* the action, pseudo-fermion field and force in the parameter data base (see
* flags/{action,force}_parms.c).
*
* If type="modes", the index idx selects the following:
*
*  idx=0       Counter for the solver iteration numbers performed by
*              the deflation mode generation program dfl_modes2().
*
*  idx=1       Same for the mode update program dfl_update2().
*
*  idx=2       Counter of the number of complete regenerations of the
*              deflation subspace in dfl_modes2(), dfl_update2() and
*              dfl_sap_gcr2().
*
* The status array passed to add2counter() is expected to contain all solver
* iteration numbers returned by the associated action, pseudo-fermion field
* generation, force and mode-generation program. In the case of the "modes"
* counter with idx=2, a single integer (the number of regenerations since
* the last addition) is passed.
*
* Printed status numbers use various formats depending on the counter type
* and the solvers used. Examples are
*
*  67
*  67;89
*  28[3|12]
*
* Semicolons ";" separate the two calls of the solver in the force programs,
* while the bar "|" separates the fgcr solver iteration numbers from the gcr
* ones reported by the solver for the little Dirac equation [dfl/ltl_gcr.c].
* The combination 28[3|12] is the printout of status values returned by the
* DFL_SAP_GCR solver [dfl/dfl_sap_gcr.c].
*
* The programs setup_counters(), clear_counters() and add2counter() are
* assumed to be called by the OpenMP master thread on all MPI processes
* simultaneously. All other programs may be called by the OpenMP master
* thread on any MPI process.
*
*******************************************************************************/

#define COUNTERS_C

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "update.h"
#include "global.h"

typedef struct
{
   int n,ipr;
   int ns,ms,ofs;
   int *status;
} counter_t;

static int nlv=0,nact=0,*iact=NULL;
static int nac=0,nfd=0,nfr=0,nmd=0;
static counter_t *act=NULL,*fld=NULL,*frc=NULL,*mds=NULL;


static void check_levels(void)
{
   int lv,ie;
   hmc_parms_t hmc;
   smd_parms_t smd;
   mdint_parms_t mdp;

   hmc=hmc_parms();
   smd=smd_parms();

   if (hmc.nlv!=0)
   {
      nlv=hmc.nlv;
      nact=hmc.nact;
      iact=hmc.iact;
   }
   else if (smd.nlv!=0)
   {
      nlv=smd.nlv;
      nact=smd.nact;
      iact=smd.iact;
   }
   else
      error_root(1,1,"check_levels [counters.c]",
                 "Simulation parameters are not set");

   ie=0;

   for (lv=0;lv<nlv;lv++)
   {
      mdp=mdint_parms(lv);
      ie|=(mdp.nstep==0);
   }

   error_root(ie!=0,1,"check_levels [counters.c]",
              "MD integrator parameters are not set");
}


static counter_t *alloc_cnt(int nc)
{
   int i;
   counter_t *cnt;

   if (nc>0)
   {
      cnt=malloc(nc*sizeof(*cnt));
      error(cnt==NULL,1,"alloc_cnt [counters.c]",
            "Unable to allocate counters");

      for (i=0;i<nc;i++)
      {
         cnt[i].n=0;
         cnt[i].ipr=0;
         cnt[i].ns=0;
         cnt[i].ms=0;
         cnt[i].ofs=0;
         cnt[i].status=NULL;
      }

      return cnt;
   }
   else
      return NULL;
}


static void set_nc(void)
{
   int i,j,k;
   action_parms_t ap;
   mdint_parms_t mdp;
   force_parms_t fp;
   solver_parms_t sp;

   nac=0;
   nfd=0;
   nfr=0;
   nmd=0;

   for (i=0;i<nact;i++)
   {
      j=iact[i];
      ap=action_parms(j);

      if ((ap.action==ACF_TM1)||
          (ap.action==ACF_TM1_EO)||
          (ap.action==ACF_TM1_EO_SDET)||
          (ap.action==ACF_TM2)||
          (ap.action==ACF_TM2_EO)||
          (ap.action==ACF_RAT)||
          (ap.action==ACF_RAT_SDET))
      {
         if (j>=nac)
            nac=j+1;

         sp=solver_parms(ap.isp[0]);
         if (sp.solver==DFL_SAP_GCR)
            nmd=3;

         if ((ap.action==ACF_TM2)||
             (ap.action==ACF_TM2_EO))
         {
            if (ap.ipf>=nfd)
               nfd=ap.ipf+1;

            sp=solver_parms(ap.isp[1]);
            if (sp.solver==DFL_SAP_GCR)
               nmd=3;
         }

         if ((ap.action==ACF_RAT)||
             (ap.action==ACF_RAT_SDET))
         {
            if (ap.ipf>=nfd)
               nfd=ap.ipf+1;
         }
      }
   }

   for (i=0;i<nlv;i++)
   {
      mdp=mdint_parms(i);

      for (j=0;j<mdp.nfr;j++)
      {
         k=mdp.ifr[j];
         fp=force_parms(k);

         if ((fp.force==FRF_TM1)||
             (fp.force==FRF_TM1_EO)||
             (fp.force==FRF_TM1_EO_SDET)||
             (fp.force==FRF_TM2)||
             (fp.force==FRF_TM2_EO)||
             (fp.force==FRF_RAT)||
             (fp.force==FRF_RAT_SDET))
         {
            if (k>=nfr)
               nfr=k+1;

            sp=solver_parms(fp.isp[0]);
            if (sp.solver==DFL_SAP_GCR)
               nmd=3;
         }
      }
   }
}


static void set_ns(void)
{
   int i,j,k;
   mdint_parms_t mdp;
   action_parms_t ap;
   force_parms_t fp;
   solver_parms_t sp;

   for (i=0;i<nact;i++)
   {
      j=iact[i];
      ap=action_parms(j);

      if ((ap.action==ACF_TM1)||
          (ap.action==ACF_TM1_EO)||
          (ap.action==ACF_TM1_EO_SDET)||
          (ap.action==ACF_TM2)||
          (ap.action==ACF_TM2_EO)||
          (ap.action==ACF_RAT)||
          (ap.action==ACF_RAT_SDET))
      {
         sp=solver_parms(ap.isp[0]);

         if ((sp.solver==CGNE)||(sp.solver==MSCG)||(sp.solver==SAP_GCR))
         {
            act[j].ns=1;
            act[j].ms=1;
         }
         else if (sp.solver==DFL_SAP_GCR)
         {
            act[j].ipr=1;
            act[j].ns=NSTD_STATUS/2;
            act[j].ms=NSTD_STATUS/2;
         }
         else
            error_root(1,1,"set_ns [counters.c]","Unknown solver");

         if ((ap.action==ACF_TM2)||
             (ap.action==ACF_TM2_EO)||
             (ap.action==ACF_RAT)||
             (ap.action==ACF_RAT_SDET))
         {
            k=ap.ipf;
            sp=solver_parms(ap.isp[1]);

            if ((sp.solver==CGNE)||(sp.solver==MSCG)||(sp.solver==SAP_GCR))
            {
               fld[k].ns=1;
               fld[k].ms=1;
            }
            else if (sp.solver==DFL_SAP_GCR)
            {
               fld[k].ipr=1;
               fld[k].ns=NSTD_STATUS/2;
               fld[k].ms=NSTD_STATUS/2;
            }
            else
               error_root(1,1,"set_ns [counters.c]","Unknown solver");
         }
      }
      else if (ap.action!=ACG)
         error_root(1,1,"set_ns [counters.c]","Unknown action");
   }

   for (i=0;i<nlv;i++)
   {
      mdp=mdint_parms(i);

      for (j=0;j<mdp.nfr;j++)
      {
         k=mdp.ifr[j];
         fp=force_parms(k);

         if ((fp.force==FRF_TM1)||
             (fp.force==FRF_TM1_EO)||
             (fp.force==FRF_TM1_EO_SDET)||
             (fp.force==FRF_TM2)||
             (fp.force==FRF_TM2_EO)||
             (fp.force==FRF_RAT)||
             (fp.force==FRF_RAT_SDET))
         {
            sp=solver_parms(fp.isp[0]);

            if ((sp.solver==CGNE)||(sp.solver==MSCG))
            {
               frc[k].ns=1;
               frc[k].ms=1;
            }
            else if (sp.solver==SAP_GCR)
            {
               frc[k].ns=2;
               frc[k].ms=1;
               frc[k].ofs=NSTD_STATUS/2;
            }
            else if (sp.solver==DFL_SAP_GCR)
            {
               frc[k].ipr=1;
               frc[k].ns=2*(NSTD_STATUS/2);
               frc[k].ms=NSTD_STATUS/2;
               frc[k].ofs=NSTD_STATUS/2;
            }
            else
               error_root(1,1,"set_ns [counters.c]","Unknown solver");
         }
         else if (fp.force!=FRG)
            error_root(1,1,"set_ns [counters.c]","Unknown force");
      }
   }

   for (i=0;i<nmd;i++)
   {
      if ((i==0)||(i==1))
      {
         mds[i].ipr=2;
         mds[i].ns=2;
         mds[i].ms=2;
      }
      else
      {
         mds[i].ipr=3;
         mds[i].ns=1;
         mds[i].ms=1;
      }
   }
}


static void alloc_stat(int nc,counter_t *cnt)
{
   int i,ns,*stat;

   if (nc>0)
   {
      ns=0;

      for (i=0;i<nc;i++)
         ns+=cnt[i].ns;

      if (ns>0)
      {
         stat=malloc(ns*sizeof(*stat));
         error(stat==NULL,1,"alloc_stat [counters.c]",
               "Unable to allocate status arrays");

         for (i=0;i<nc;i++)
         {
            if (cnt[i].ns>0)
            {
               cnt[i].status=stat;
               stat+=cnt[i].ns;
            }
         }
      }
   }
}


void setup_counters(void)
{
   error(nlv!=0,1,"setup_counters [counters.c]",
         "Attempt to setup the counters a second time");

   check_levels();

   set_nc();
   act=alloc_cnt(nac);
   fld=alloc_cnt(nfd);
   frc=alloc_cnt(nfr);
   mds=alloc_cnt(nmd);

   set_ns();
   alloc_stat(nac,act);
   alloc_stat(nfd,fld);
   alloc_stat(nfr,frc);
   alloc_stat(nmd,mds);

   clear_counters();
}


static void set_cnt2zero(int nc,counter_t *cnt)
{
   int i,j,ns,*stat;

   for (i=0;i<nc;i++)
   {
      cnt[i].n=0;
      ns=cnt[i].ns;
      stat=cnt[i].status;

      for (j=0;j<ns;j++)
         stat[j]=0;
   }
}


void clear_counters(void)
{
   set_cnt2zero(nac,act);
   set_cnt2zero(nfd,fld);
   set_cnt2zero(nfr,frc);
   set_cnt2zero(nmd,mds);
}


void add2counter(char *type,int idx,int *status)
{
   int i,nc,ns,ms,ofs,*stat;
   counter_t *cnt;

   if (strcmp(type,"action")==0)
   {
      nc=nac;
      cnt=act;
   }
   else if (strcmp(type,"field")==0)
   {
      nc=nfd;
      cnt=fld;
   }
   else if (strcmp(type,"force")==0)
   {
      nc=nfr;
      cnt=frc;
   }
   else if (strcmp(type,"modes")==0)
   {
      nc=nmd;
      cnt=mds;
   }
   else
   {
      error_loc(1,1,"add2counter [counters.c]","Unknown counter type");
      return;
   }

   if ((idx>=0)&&(idx<nc))
   {
      cnt[idx].n+=1;
      ns=cnt[idx].ns;
      ms=cnt[idx].ms;
      ofs=cnt[idx].ofs;
      stat=cnt[idx].status;

      for (i=0;i<ms;i++)
         stat[i]+=status[i];

      if (ofs>0)
      {
         for (i=ms;i<ns;i++)
            stat[i]+=status[ofs+i-ms];
      }
   }
   else
      error_loc(1,1,"add2counter [counters.c]","Counter index is out of range");
}


int get_count(char *type,int idx,int *status)
{
   int i,nc,ns,ms,ofs,*stat;
   counter_t *cnt;

   if (strcmp(type,"action")==0)
   {
      nc=nac;
      cnt=act;
   }
   else if (strcmp(type,"field")==0)
   {
      nc=nfd;
      cnt=fld;
   }
   else if (strcmp(type,"force")==0)
   {
      nc=nfr;
      cnt=frc;
   }
   else if (strcmp(type,"modes")==0)
   {
      nc=nmd;
      cnt=mds;
   }
   else
   {
      error_loc(1,1,"get_count [counters.c]","Unknown counter type");
      return 0;
   }

   if ((idx>=0)&&(idx<nc))
   {
      ns=cnt[idx].ns;
      ms=cnt[idx].ms;
      ofs=cnt[idx].ofs;
      stat=cnt[idx].status;

      for (i=0;i<ms;i++)
         status[i]=stat[i];

      if (ofs>0)
      {
         for (i=ms;i<ns;i++)
            status[ofs+i-ms]=stat[i];
      }

      return cnt[idx].n;
   }
   else
      error_loc(1,1,"get_count [counters.c]","Counter index out of range");

   return 0;
}


void print_avgstat(char *type,int idx)
{
   int my_rank,n,nh,i;
   int ipr,ns,ms,*stat;
   counter_t *cnt;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      cnt=NULL;

      if (strcmp(type,"action")==0)
      {
         if ((idx>=0)&&(idx<nac))
         {
            cnt=act+idx;
            if (cnt[0].ns)
               printf("Action %2d: <status> = ",idx);
         }
      }
      else if (strcmp(type,"field")==0)
      {
         if ((idx>=0)&&(idx<nfd))
         {
            cnt=fld+idx;
            if (cnt[0].ns)
               printf("Field  %2d: <status> = ",idx);
         }
      }
      else if (strcmp(type,"force")==0)
      {
         if ((idx>=0)&&(idx<nfr))
         {
            cnt=frc+idx;
            if (cnt[0].ns)
               printf("Force  %2d: <status> = ",idx);
         }
      }
      else if (strcmp(type,"modes")==0)
      {
         if ((idx>=0)&&(idx<nmd))
         {
            cnt=mds+idx;
            if (cnt[0].ns)
               printf("Modes  %2d: <status> = ",idx);
         }
      }
      else
         error_root(1,1,"print_avgstat [counters.c]","Unknown counter type");

      error_root((cnt==NULL)||(cnt[0].ns==0),1,"print_avgstat [counters.c]",
                 "Unknown counter or counter index is out of range");

      n=cnt[0].n;
      ipr=cnt[0].ipr;
      ns=cnt[0].ns;
      ms=cnt[0].ms;
      stat=cnt[0].status;
      if (n==0)
         n=1;
      nh=n/2;

      if (ipr==0)
      {
         for (i=0;i<ms;i++)
         {
            if (i>0)
               printf(",");
            printf("%d",(stat[i]+nh)/n);
         }

         if (ms<ns)
         {
            printf(";");

            for (i=ms;i<ns;i++)
            {
               if (i>ms)
                  printf(",");
               printf("%d",(stat[i]+nh)/n);
            }
         }
      }
      else if (ipr==1)
      {
         printf("%d[",(stat[0]+nh)/n);

         for (i=1;i<ms;i++)
         {
            if (i==((ms+1)/2))
               printf("|");
            else if (i>1)
               printf(",");
            printf("%d",(stat[i]+nh)/n);
         }

         printf("]");

         if (ms<ns)
         {
            printf(";%d[",(stat[ms]+nh)/n);

            for (i=(ms+1);i<ns;i++)
            {
               if (i==(ms+(ms+1)/2))
                  printf("|");
               else if (i>(ms+1))
                  printf(",");
               printf("%d",(stat[i]+nh)/n);
            }

            printf("]");
         }
      }
      else if (ipr==2)
      {
         for (i=0;i<ms;i++)
         {
            if (i==(ms/2))
               printf("|");
            else if (i>0)
               printf(",");
            printf("%d",(stat[i]+nh)/n);
         }
      }
      else if (ipr==3)
         printf("%d",stat[0]);

      if (strcmp(type,"modes")==0)
      {
         if (idx==0)
         {
            if (mds[2].n>0)
               printf(" (no of regenerations = %d)",mds[2].status[0]);
         }
         else if (idx==1)
         {
            if (mds[1].n>0)
               printf(" (no of updates = %d)",mds[1].n);
         }
      }

      printf("\n");
   }
}


void print_all_avgstat(void)
{
   int i;

   for (i=0;i<nac;i++)
   {
      if (act[i].ns>0)
         print_avgstat("action",i);
   }

   for (i=0;i<nfd;i++)
   {
      if (fld[i].ns>0)
         print_avgstat("field",i);
   }

   for (i=0;i<nfr;i++)
   {
      if (frc[i].ns>0)
         print_avgstat("force",i);
   }

   for (i=0;i<(nmd-1);i++)
   {
      if (mds[i].ns>0)
         print_avgstat("modes",i);
   }
}
