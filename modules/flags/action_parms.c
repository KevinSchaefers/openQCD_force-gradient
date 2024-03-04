
/*******************************************************************************
*
* File action_parms.c
*
* Copyright (C) 2011-2013, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Action parameter data base.
*
*   action_parms_t set_action_parms(int iact,action_t action,int ipf,
*                                   int im0,int *irat,int *imu,int *isp)
*     Sets the parameters in the action parameter set number iact and returns
*     a structure containing them (see the notes).
*
*   action_parms_t action_parms(int iact)
*     Returns a structure containing the action parameter set number iact
*     (see the notes).
*
*   void read_action_parms(int iact)
*     Reads the parameter section [Action <int>] on MPI process 0 from
*     stdin, where <int> is set to the action index iact. Depending on
*     the type of action, only a subset of the tags
*
*       action   <action_t>
*       ipf      <int>
*       im0      <int>
*       irat     <int> <int> <int>
*       imu      <int> [<int>]
*       isp      <int> [<int>]
*
*     must be present in the section.
*
*   void read_all_action_parms(void)
*     Reads the action parameters of all actions listed in the SMD or HMC
*     parameter data base from stdin on the MPI process 0. An error occurs
*     if both lists of actions are empty.
*
*   void print_action_parms(void)
*     Prints the parameters of the defined actions to stdout on MPI
*     process 0.
*
*   void write_action_parms(FILE *fdat)
*     Writes the parameters of the defined actions to the file fdat on
*     MPI process 0.
*
*   void check_action_parms(FILE *fdat)
*     Compares the parameters of the defined actions with those stored
*     on the file fdat on MPI process 0, assuming the latter were written
*     to the file by the program write_action_parms().
*
* For a description of the supported actions and their parameters see
* forces/README.forces.
*
* The elements of a structure of type action_parms_t are
*
*   action  Action program used. This parameter is an enum type with
*           one of the following values:
*
*            ACG             (program action0() [forces/force0.c]),
*
*            ACF_TM1         (program action1() [forces/force1.c]),
*
*            ACF_TM1_EO      (program action4() [forces/force4.c]),
*
*            ACF_TM1_EO_SDET (program action4() [forces/force4.c]),
*
*            ACF_TM2         (program action2() [forces/force2.c]),
*
*            ACF_TM2_EO      (program action5() [forces/force5.c]),
*
*            ACF_RAT         (program action3() [forces/force3.c]),
*
*            ACF_RAT_SDET    (program action3() [forces/force3.c]),
*
*   ipf     Pseudo-fermion field index (see mdflds/mdflds.c),
*
*   im0     Index of the bare sea quark mass in parameter data base
*           (see flags/lat_parms.c),
*
*   irat    Indices specifying a rational function (see ratfcts/ratfcts.c),
*
*   imu     Twisted mass indices (see flags/{hmc,smd}_parms.c),
*
*   isp     Solver parameter set indices (see flags/solver_parms.c).
*
* Depending on the action, some parameters are not used and are set to zero
* by set_action_parms() independently of the values of the arguments. In
* particular, for a given action, only the required number of integers are
* read from the arrays imu and isp passed to the program.
*
* The number of twisted mass indices and solver parameter set indices is
* 1 and 2 in the case of the actions ACF_TM1* and ACF_TM2*, respectively,
* where isp[k] is the solver parameter set used for the solution of the
* Dirac equation with twisted mass index imu[k]. Two solver parameter set
* indices are also required for the actions ACF_RAT*, the first being used
* when the action is computed and the other when the pseudo-fermion field
* is generated.
*
* Up to 32 action parameter sets, labeled by an index iact=0,1,..,31, can
* be specified. Once a set is specified, it cannot be changed by calling
* set_action_parms() again. Action parameters must be globally the same.
*
* Except for action_parms(), the programs in this module perform global
* operations and must be called simultaneously on all MPI processes.
*
*******************************************************************************/

#define ACTION_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

#define IACMAX 32

static int init=0;
static action_parms_t ap[IACMAX+1]={{ACTIONS,0,0,{0,0,0},{0,0,0,0},{0,0,0,0}}};


static void init_ap(void)
{
   int i;

   for (i=1;i<=IACMAX;i++)
      ap[i]=ap[0];

   init=1;
}


action_parms_t set_action_parms(int iact,action_t action,
                                int ipf,int im0,int *irat,int *imu,int *isp)
{
   int iprms[15],i,ie;
   int rat[3],mu[4],sp[4];

   if (init==0)
      init_ap();

   for (i=0;i<3;i++)
      rat[i]=0;

   for (i=0;i<4;i++)
   {
      mu[i]=0;
      sp[i]=0;
   }

   if ((action==ACG)||(action==ACTIONS))
   {
      ipf=0;
      im0=0;
   }
   else if ((action==ACF_TM1)||(action==ACF_TM1_EO)||(action==ACF_TM1_EO_SDET))
   {
      mu[0]=imu[0];
      sp[0]=isp[0];
   }
   else if ((action==ACF_TM2)||(action==ACF_TM2_EO))
   {
      mu[0]=imu[0];
      mu[1]=imu[1];
      sp[0]=isp[0];
      sp[1]=isp[1];
   }
   else if ((action==ACF_RAT)||(action==ACF_RAT_SDET))
   {
      rat[0]=irat[0];
      rat[1]=irat[1];
      rat[2]=irat[2];
      sp[0]=isp[0];
      sp[1]=isp[1];
   }

   if (NPROC>1)
   {
      iprms[0]=iact;
      iprms[1]=(int)(action);
      iprms[2]=ipf;
      iprms[3]=im0;

      for (i=0;i<3;i++)
         iprms[4+i]=rat[i];

      for (i=0;i<4;i++)
      {
         iprms[7+i]=mu[i];
         iprms[11+i]=sp[i];
      }

      MPI_Bcast(iprms,15,MPI_INT,0,MPI_COMM_WORLD);

      ie=0;
      ie|=(iprms[0]!=iact);
      ie|=(iprms[1]!=(int)(action));
      ie|=(iprms[2]!=ipf);
      ie|=(iprms[3]!=im0);

      for (i=0;i<3;i++)
         ie|=(iprms[4+i]!=rat[i]);

      for (i=0;i<4;i++)
      {
         ie|=(iprms[7+i]!=mu[i]);
         ie|=(iprms[11+i]!=sp[i]);
      }

      error(ie!=0,1,"set_action_parms [action_parms.c]",
            "Parameters are not global");
   }

   ie=0;
   ie|=((iact<0)||(iact>=IACMAX));
   ie|=(action==ACTIONS);
   ie|=((ipf<0)||(im0<0));

   for (i=0;i<3;i++)
      ie|=(rat[i]<0);

   for (i=0;i<4;i++)
      ie|=((mu[i]<0)||(sp[i]<0));

   error_root(ie!=0,1,"set_action_parms [action_parms.c]",
              "Parameters are out of range");

   error_root(ap[iact].action!=ACTIONS,1,"set_action_parms [action_parms.c]",
              "Attempt to reset already specified action parameters");

   ap[iact].action=action;
   ap[iact].ipf=ipf;
   ap[iact].im0=im0;

   for (i=0;i<3;i++)
      ap[iact].irat[i]=rat[i];

   for (i=0;i<4;i++)
   {
      ap[iact].imu[i]=mu[i];
      ap[iact].isp[i]=sp[i];
   }

   return ap[iact];
}


action_parms_t action_parms(int iact)
{
   if (init==0)
      init_ap();

   if ((iact>=0)&&(iact<IACMAX))
      return ap[iact];
   else
   {
      error_loc(1,1,"action_parms [action_parms.c]",
                "Action index is out of range");
      return ap[IACMAX];
   }
}


void read_action_parms(int iact)
{
   int my_rank,i,ida;
   int ipf,im0,irat[3],imu[4],isp[4];
   char line[NAME_SIZE];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   ida=0;
   ipf=0;
   im0=0;

   for (i=0;i<3;i++)
      irat[i]=0;

   for (i=0;i<4;i++)
   {
      imu[i]=0;
      isp[i]=0;
   }

   if (my_rank==0)
   {
      sprintf(line,"Action %d",iact);
      find_section(line);

      read_line("action","%s",line);

      if (strcmp(line,"ACG")==0)
         ida=(int)(ACG);
      else if (strcmp(line,"ACF_TM1")==0)
      {
         ida=(int)(ACF_TM1);
         read_line("ipf","%d",&ipf);
         read_line("im0","%d",&im0);
         read_line("imu","%d",imu);
         read_line("isp","%d",isp);
      }
      else if (strcmp(line,"ACF_TM1_EO")==0)
      {
         ida=(int)(ACF_TM1_EO);
         read_line("ipf","%d",&ipf);
         read_line("im0","%d",&im0);
         read_line("imu","%d",imu);
         read_line("isp","%d",isp);
      }
      else if (strcmp(line,"ACF_TM1_EO_SDET")==0)
      {
         ida=(int)(ACF_TM1_EO_SDET);
         read_line("ipf","%d",&ipf);
         read_line("im0","%d",&im0);
         read_line("imu","%d",imu);
         read_line("isp","%d",isp);
      }
      else if (strcmp(line,"ACF_TM2")==0)
      {
         ida=(int)(ACF_TM2);
         read_line("ipf","%d",&ipf);
         read_line("im0","%d",&im0);
         read_line("imu","%d %d",imu,imu+1);
         read_line("isp","%d %d",isp,isp+1);
      }
      else if (strcmp(line,"ACF_TM2_EO")==0)
      {
         ida=(int)(ACF_TM2_EO);
         read_line("ipf","%d",&ipf);
         read_line("im0","%d",&im0);
         read_line("imu","%d %d",imu,imu+1);
         read_line("isp","%d %d",isp,isp+1);
      }
      else if (strcmp(line,"ACF_RAT")==0)
      {
         ida=(int)(ACF_RAT);
         read_line("ipf","%d",&ipf);
         read_line("im0","%d",&im0);
         read_line("irat","%d %d %d",irat,irat+1,irat+2);
         read_line("isp","%d %d",isp,isp+1);
      }
      else if (strcmp(line,"ACF_RAT_SDET")==0)
      {
         ida=(int)(ACF_RAT_SDET);
         read_line("ipf","%d",&ipf);
         read_line("im0","%d",&im0);
         read_line("irat","%d %d %d",irat,irat+1,irat+2);
         read_line("isp","%d %d",isp,isp+1);
      }
      else
      {
         ida=(int)(ACTIONS);
         error_root(1,1,"read_action_parms [action_parms.c]",
                    "Unknown action %s",line);
      }
   }

   if (NPROC>1)
   {
      MPI_Bcast(&ida,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&ipf,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&im0,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(irat,3,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(imu,4,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(isp,4,MPI_INT,0,MPI_COMM_WORLD);
   }

   set_action_parms(iact,(action_t)(ida),ipf,im0,irat,imu,isp);
}


void read_all_action_parms(void)
{
   int i,j,k,nact,*iact;
   hmc_parms_t hmc;
   smd_parms_t smd;
   action_parms_t ac;
   rat_parms_t rp;

   hmc=hmc_parms();
   smd=smd_parms();
   error_root((hmc.nact==0)&&(smd.nact==0),1,
              "read_all_action_parms [action_parms.c]","Empty action lists");

   if (hmc.nact>0)
   {
      nact=hmc.nact;
      iact=hmc.iact;
   }
   else
   {
      nact=smd.nact;
      iact=smd.iact;
   }

   for (i=0;i<nact;i++)
   {
      j=iact[i];
      ac=action_parms(j);

      if (ac.action==ACTIONS)
      {
         read_action_parms(j);
         ac=action_parms(j);

         if ((ac.action==ACF_RAT)||(ac.action==ACF_RAT_SDET))
         {
            k=ac.irat[0];
            rp=rat_parms(k);

            if (rp.degree==0)
               read_rat_parms(k);
         }
      }
   }
}


void print_action_parms(void)
{
   int my_rank,i;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if ((my_rank==0)&&(init==1))
   {
      for (i=0;i<IACMAX;i++)
      {
         if (ap[i].action!=ACTIONS)
         {
            printf("Action %d:\n",i);

            if (ap[i].action==ACG)
               printf("ACG action\n\n");
            else if (ap[i].action==ACF_TM1)
            {
               printf("ACF_TM1 action\n");
               printf("ipf = %d\n",ap[i].ipf);
               printf("im0 = %d\n",ap[i].im0);
               printf("imu = %d\n",ap[i].imu[0]);
               printf("isp = %d\n\n",ap[i].isp[0]);
            }
            else if (ap[i].action==ACF_TM1_EO)
            {
               printf("ACF_TM1_EO action\n");
               printf("ipf = %d\n",ap[i].ipf);
               printf("im0 = %d\n",ap[i].im0);
               printf("imu = %d\n",ap[i].imu[0]);
               printf("isp = %d\n\n",ap[i].isp[0]);
            }
            else if (ap[i].action==ACF_TM1_EO_SDET)
            {
               printf("ACF_TM1_EO_SDET action\n");
               printf("ipf = %d\n",ap[i].ipf);
               printf("im0 = %d\n",ap[i].im0);
               printf("imu = %d\n",ap[i].imu[0]);
               printf("isp = %d\n\n",ap[i].isp[0]);
            }
            else if (ap[i].action==ACF_TM2)
            {
               printf("ACF_TM2 action\n");
               printf("ipf = %d\n",ap[i].ipf);
               printf("im0 = %d\n",ap[i].im0);
               printf("imu = %d %d\n",ap[i].imu[0],ap[i].imu[1]);
               printf("isp = %d %d\n\n",ap[i].isp[0],ap[i].isp[1]);
            }
            else if (ap[i].action==ACF_TM2_EO)
            {
               printf("ACF_TM2_EO action\n");
               printf("ipf = %d\n",ap[i].ipf);
               printf("im0 = %d\n",ap[i].im0);
               printf("imu = %d %d\n",ap[i].imu[0],ap[i].imu[1]);
               printf("isp = %d %d\n\n",ap[i].isp[0],ap[i].isp[1]);
            }
            else if (ap[i].action==ACF_RAT)
            {
               printf("ACF_RAT action\n");
               printf("ipf = %d\n",ap[i].ipf);
               printf("im0 = %d\n",ap[i].im0);
               printf("irat = %d %d %d\n",
                      ap[i].irat[0],ap[i].irat[1],ap[i].irat[2]);
               printf("isp = %d %d\n\n",ap[i].isp[0],ap[i].isp[1]);
            }
            else if (ap[i].action==ACF_RAT_SDET)
            {
               printf("ACF_RAT_SDET action\n");
               printf("ipf = %d\n",ap[i].ipf);
               printf("im0 = %d\n",ap[i].im0);
               printf("irat = %d %d %d\n",
                      ap[i].irat[0],ap[i].irat[1],ap[i].irat[2]);
               printf("isp = %d %d\n\n",ap[i].isp[0],ap[i].isp[1]);
            }
            else
               printf("UNKNOWN action\n\n");
         }
      }
   }
}


void write_action_parms(FILE *fdat)
{
   int my_rank,endian;
   int iw,i,j;
   stdint_t istd[15];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if ((my_rank==0)&&(init==1))
   {
      for (i=0;i<IACMAX;i++)
      {
         if (ap[i].action!=ACTIONS)
         {
            istd[0]=(stdint_t)(i);
            istd[1]=(stdint_t)(ap[i].action);
            istd[2]=(stdint_t)(ap[i].ipf);
            istd[3]=(stdint_t)(ap[i].im0);

            for (j=0;j<3;j++)
               istd[4+j]=(stdint_t)(ap[i].irat[j]);

            for (j=0;j<4;j++)
            {
               istd[7+j]=(stdint_t)(ap[i].imu[j]);
               istd[11+j]=(stdint_t)(ap[i].isp[j]);
            }

            if (endian==BIG_ENDIAN)
               bswap_int(15,istd);

            iw=fwrite(istd,sizeof(stdint_t),15,fdat);
            error_root(iw!=15,1,"write_action_parms [action_parms.c]",
                       "Incorrect write count");
         }
      }
   }
}


void check_action_parms(FILE *fdat)
{
   int my_rank,endian;
   int ir,ie,i,j;
   stdint_t istd[15];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if ((my_rank==0)&&(init==1))
   {
      ie=0;

      for (i=0;i<IACMAX;i++)
      {
         if (ap[i].action!=ACTIONS)
         {
            ir=fread(istd,sizeof(stdint_t),15,fdat);
            error_root(ir!=15,1,"check_action_parms [action_parms.c]",
                       "Incorrect read count");

            if (endian==BIG_ENDIAN)
               bswap_int(15,istd);

            ie|=(istd[0]!=(stdint_t)(i));
            ie|=(istd[1]!=(stdint_t)(ap[i].action));
            ie|=(istd[2]!=(stdint_t)(ap[i].ipf));
            ie|=(istd[3]!=(stdint_t)(ap[i].im0));

            for (j=0;j<3;j++)
               ie|=(istd[4+j]!=(stdint_t)(ap[i].irat[j]));

            for (j=0;j<4;j++)
            {
               ie|=(istd[7+j]!=(stdint_t)(ap[i].imu[j]));
               ie|=(istd[11+j]!=(stdint_t)(ap[i].isp[j]));
            }
         }
      }

      error_root(ie!=0,1,"check_action_parms [action_parms.c]",
                 "Parameters do not match");
   }
}
