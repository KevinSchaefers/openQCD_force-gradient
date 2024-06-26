

/*******************************************************************************
*
* File mdsteps.c
*
* Copyright (C) 2011, 2012, 2017, 2018 Martin Luescher
* 2024 Kevin Schaefers, Jacob Finkenrath
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Molecular-dynamics integrator.
*
*   void set_mdsteps(void)
*     Constructs the integrator from the data available in the parameter
*     data base (see the notes). The integrator is stored internally in
*     the form of an array of elementary operations (force computations
*     and gauge-field update steps).
*
*   mdstep_t *mdsteps(int *nop,int *ntu,int *itu)
*     Returns the array of elementary operations that describe the current
*     integrator. On exit the program assigns the total number of operations
*     to nop and the index of the gauge-field update operation to itu.
*
*   void print_mdsteps(int ipr)
*     Prints some information on the current integrator to stdout on MPI
*     process 0. The program always prints the available information on
*     the different levels of the integrator. Whether further information
*     is printed depends on the 3 low bits of the print flat ipr:
*
*      if (ipr&0x1): Force descriptions
*
*      if (ipr&0x2): List of elementary operations
*
*      if (ipr&0x4): Integration time check
*
*     The full information is thus printed if ipr=0x7.
*
* The structure of the MD integrator is explained in the file README.mdint
* in this directory. It is assumed here that the parameters of the integrator
* have been entered to the parameter data base.
*
* An elementary update step is described by a structure of type mdstep_t
* with the following elements:
*
*  iop     Operation index (0<=iop<=itu+1=iend). If iop<itu, the force number
*          iop is to be computed and to be assigned (gauge force) or added
*          (fermion forces) to the force field. If iop=itu, the momentum
*          and subsequently the gauge field are to be updated, using the
*          current force field. If iop=itu+1, the momentum field is to be
*          updated, using the current force, and the integration ends.
*
*  eps     Step size by which the force (iop<itu) or the momentum field
*          in the update of the gauge field must be multiplied.
*
*  lvl_id  level id. If lvl_id = 0, the force updates are just usual force 
*          updates without force-gradient step. If lvl_id > 0, the lvl_id 
*          is equal to ilv and the respective force update is part of a 
*          force-gradient step.
*
* The forces are described by the structures returned by force_parms(iop)
* if iop<itu (see flags/force_parms.c).
*
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "update.h"
#include "global.h"

static int nsmx,nmds=0,iend=1;
static mdstep_t *mds=NULL,*mdw[3];


static void set_nsmx(int nlv)
{
   int ntu,ilv;
   int nfr,*ifr,i;
   mdint_parms_t mdp;

   iend=0;
   ntu=1;

   for (ilv=0;ilv<nlv;ilv++)
   {
      mdp=mdint_parms(ilv);

      if (mdp.integrator==LPFR || mdp.integrator==BAB || mdp.integrator==ABA)
         ntu*=mdp.nstep;
      else if (mdp.integrator==OMF2 || mdp.integrator==DAD || mdp.integrator==ADA || mdp.integrator==BABAB || mdp.integrator==ABABA)
         ntu*=2*mdp.nstep;
      else if (mdp.integrator==BADAB || mdp.integrator==DABAD || mdp.integrator==ABABABA || mdp.integrator==BABABAB)
          ntu*=3*mdp.nstep;
      else if (mdp.integrator==DADAD || mdp.integrator==ADADA || mdp.integrator==ABADABA || mdp.integrator==DABABAD || mdp.integrator==ABABABABA || mdp.integrator==BABABABAB)
         ntu*=4*mdp.nstep;
      else if (mdp.integrator==OMF4 || mdp.integrator==BADADAB || mdp.integrator==ADABADA || mdp.integrator==BABADABAB || mdp.integrator==DABABABAD || mdp.integrator == BABABABABAB || mdp.integrator==ABABABABABA)
         ntu*=5*mdp.nstep;
      else if (mdp.integrator==ADADADA || mdp.integrator==DADADAD || mdp.integrator==BADABADAB || mdp.integrator==DABADABAD || mdp.integrator==ABADADABA || mdp.integrator==ADABABADA || mdp.integrator==ABABADABABA || mdp.integrator==DABABABABAD)
          ntu*=6*mdp.nstep;
      else if (mdp.integrator==DADABADAD || mdp.integrator==BADADADAB || mdp.integrator==ABADABADABA || mdp.integrator==BADABABADAB || mdp.integrator==ADABABABADA || mdp.integrator==BABADADABAB || mdp.integrator==BABABABABABABAB || mdp.integrator==ABABABABABABABA)
          ntu*=7*mdp.nstep;
      else if (mdp.integrator==ADADADADA || mdp.integrator==ADABADABADA || mdp.integrator==DABADADABAD || mdp.integrator==DADABABADAD)
          ntu*=8*mdp.nstep;
      else if (mdp.integrator==ADADABADADA || mdp.integrator==BADADADADAB)
          ntu*=9*mdp.nstep;
      else if (mdp.integrator==ADADADADADA)
          ntu*=10*mdp.nstep;
      else
         error(1,1,"set_nsmx [mdsteps.c]","Unknown integrator");

      nfr=mdp.nfr;
      ifr=mdp.ifr;

      for (i=0;i<nfr;i++)
      {
         if (ifr[i]>iend)
            iend=ifr[i];
      }
   }

   iend+=5;
   nsmx=(2*ntu+1)*iend;
}


static void alloc_mds(void)
{
   int k;

   if (mds!=NULL)
      free(mds);

   mds=malloc(4*nsmx*sizeof(*mds));
   error(mds==NULL,1,"alloc_mds [mdsteps.c]",
         "Unable to allocate mdsteps array");

   for (k=0;k<3;k++)
      mdw[k]=mds+(k+1)*nsmx;
}


static void set_steps2zero(int n,mdstep_t *s)
{
   int i;

   for (i=0;i<n;i++)
   {
      s[i].iop=iend;
      s[i].eps=0.0;
      s[i].lvl_id=-1;
   }
}


static void copy_steps(int n,double c,mdstep_t *s,mdstep_t *r, int lvl)
{
   int i;

   for (i=0;i<n;i++)
   {
      r[i].iop=s[i].iop;
      r[i].eps=c*s[i].eps;
      if (lvl != 0)
      	r[i].lvl_id=s[i].lvl_id;
      else 
        r[i].lvl_id=-1;
   }
}

static void copy_steps_fg(int n,double c,mdstep_t *s,mdstep_t *r)
{
   int i;

   for (i=0;i<n;i++)
   {
      r[i].iop=s[i].iop;
      r[i].eps=-c*s[i].eps*s[i].eps;
      r[i].lvl_id=s[i].lvl_id;
   }
}


static void expand_level(int ilv,double tau,mdstep_t *s,mdstep_t *ws)
{
   int nstep,nfr,*ifr;
   int itu,n,i,j;
   double r0,r1,r2,r3,r4,eps;
   mdint_parms_t mdp;

   mdp=mdint_parms(ilv);
   nstep=mdp.nstep;
   nfr=mdp.nfr;
   ifr=mdp.ifr;

   itu=iend-1;
   n=0;
    
   r0=mdp.lambda;
    
   r1=0.08398315262876693;
   r2=0.2539785108410595;
   r3=0.6822365335719091;
   r4=-0.03230286765269967;
    
   eps=tau/(double)(nstep);

   set_steps2zero(nsmx,s);
   set_steps2zero(nsmx,ws);

   for (i=0;i<nfr;i++)
   {
      for (j=0;j<n;j++)
      {
         if (ifr[i]==ws[j].iop)
         {
            ws[j].eps+=eps;
            break;
         }
      }

      if (j==n)
      {
         ws[n].iop=ifr[i];
         ws[n].eps=eps;
         ws[n].lvl_id=ilv;
         n+=1;
      }
   }

   if (mdp.integrator==LPFR)
   {
      copy_steps(n,0.5,ws,s,0);
      s+=n;

      for (i=1;i<=nstep;i++)
      {
         (*s).iop=itu-3;
         (*s).eps=eps;
         (*s).lvl_id=-1;
         s+=1;
         (*s).iop=itu;
         (*s).eps=eps;
         (*s).lvl_id=-1;
         s+=1;
         if (i<nstep)
            copy_steps(n,1.0,ws,s,0);
         else
            copy_steps(n,0.5,ws,s,0);
         s+=n;
      }
      (*s).iop=itu-3;
      (*s).eps=0.5*eps;
      (*s).lvl_id=-1;
      s+=1;
   }
   else if (mdp.integrator==OMF2)
   {
      copy_steps(n,r0,ws,s,0);
      s+=n;
  
      for (i=1;i<=nstep;i++)
      {
          (*s).iop=itu-3;
          (*s).eps=0.5*eps;
          (*s).lvl_id=-1;
          s+=1;
          (*s).iop=itu;
          (*s).eps=0.5*eps;
          (*s).lvl_id=-1;
          s+=1;
          
          copy_steps(n,1.0-2.0*r0,ws,s,0);
          s+=n;

          (*s).iop=itu-3;
          (*s).eps=0.5*eps;
          (*s).lvl_id=-1;
          s+=1;

          (*s).iop=itu;
          (*s).eps=0.5*eps;
          (*s).lvl_id=-1;
          s+=1;
          
          if (i<nstep)
              copy_steps(n,2.0*r0,ws,s,0);
          else
              copy_steps(n,r0,ws,s,0);
          s+=n;
      }
      (*s).iop=itu-3;
      (*s).eps=0.5*eps;
      (*s).lvl_id=-1;
      s+=1;
   }
   else if (mdp.integrator==OMF4)
   {
      copy_steps(n,r1,ws,s,0);
      s+=n;
 
      for (i=1;i<=nstep;i++)
      {
         (*s).iop=itu-3;
         (*s).eps=r2*eps;
         (*s).lvl_id=-1;
         s+=1;
          
         (*s).iop=itu;
         (*s).eps=r2*eps;
         (*s).lvl_id=-1;
         s+=1;
          
         copy_steps(n,r3,ws,s,0);
         s+=n;

         (*s).iop=itu-3;
         (*s).eps=r4*eps;
         (*s).lvl_id=-1;
         s+=1;
          
         (*s).iop=itu;
         (*s).eps=r4*eps;
         (*s).lvl_id=-1;
         s+=1;
          
         copy_steps(n,0.5-r1-r3,ws,s,0);
         s+=n;

         (*s).iop=itu-3;
         (*s).eps=(1.0-2.0*(r2+r4))*eps;
         (*s).lvl_id=-1;
         s+=1;
          
         (*s).iop=itu;
         (*s).eps=(1.0-2.0*(r2+r4))*eps;
         (*s).lvl_id=-1;
         s+=1;
          
         copy_steps(n,0.5-r1-r3,ws,s,0);
         s+=n;

         (*s).iop=itu-3;
         (*s).eps=r4*eps;
         (*s).lvl_id=-1;
         s+=1;
          
         (*s).iop=itu;
         (*s).eps=r4*eps;
         (*s).lvl_id=-1;
         s+=1;
          
         copy_steps(n,r3,ws,s,0);
         s+=n;

         (*s).iop=itu-3;
         (*s).eps=r2*eps;
         (*s).lvl_id=-1;
         s+=1;
          
         (*s).iop=itu;
         (*s).eps=r2*eps;
         (*s).lvl_id=-1;
         s+=1;                  
          
         if (i<nstep)
            copy_steps(n,2.0*r1,ws,s,0);
         else
            copy_steps(n,r1,ws,s,0);
         s+=n;
      }
      (*s).iop=itu-3;
      (*s).eps=0.5*eps;
      (*s).lvl_id=-1;
      s+=1;
   }
   else
   {
       int velocity;
       int d_a;
       int d_b;
       int d;
       
       double *a;
       double *b;
       double *c;
       
       velocity = 0;
       d_a = 0; d_b = 0; d = 0;
       
       if (mdp.integrator==BAB)
       {
           d_a = 1; a = (double *)malloc(d_a * sizeof(double)); a[0] = 1.0;
           d_b = 1; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.5;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0;
           d = 1;
           velocity = 1;
       }
       else if (mdp.integrator==ABA)
       {
           d_a = 1; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.5;
           d_b = 1; b = (double *)malloc(d_b * sizeof(double)); b[0] = 1.0;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0;
           d = 1;
           velocity = 0;
       }
       else if (mdp.integrator==DAD)
       {
           d_a = 1; a = (double *)malloc(d_a * sizeof(double)); a[0] = 1.0;
           d_b = 1; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.5;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = -1.0/48.0;
           d = 1;
           velocity = 1;
       }
       else if (mdp.integrator==ADA)
       {
           d_a = 1; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.5;
           d_b = 1; b = (double *)malloc(d_b * sizeof(double)); b[0] = 1.0;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 1.0/12.0;
           d = 1;
           velocity = 0;
       }
       else if (mdp.integrator==BABAB)
       {
           d_a = 1; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.5;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.1931833275037836; b[1] = 0.613633344992433;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.0;
           d = 1;
           velocity = 1;
       }
       else if (mdp.integrator==ABABA)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.1931833275037836; a[1] = 0.613633344992433;
           d_b = 1; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.5;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0;
           d = 1;
           velocity = 0;
       }
       else if (mdp.integrator==BADAB)
       {
           d_a = 1; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.5;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 1.0/6.0; b[1] = 2.0/3.0;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 1.0/72.0;
           d = 1;
           velocity = 1;
       }
       else if (mdp.integrator==DABAD)
       {
           d_a = 1; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.5;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 1.0/6.0; b[1] = 2.0/3.0;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 1.0/144.0; c[1] = 0.0;
           d = 1;
           velocity = 1;
       }
       else if (mdp.integrator==DADAD)
       {
           d_a = 1; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.5;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 1.0/6.0; b[1] = 2.0/3.0;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = -0.000881991367333; c[1] = 0.015652871623554;
           d = 1;
           velocity = 1;
       }
       else if (mdp.integrator==ADADA)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.211324865405187; a[1] = 0.577350269189626;
           d_b = 1; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.5;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.005582274842315;
           d = 1;
           velocity = 0;
       }
       else if (mdp.integrator==ABABABA)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.675603595979829; a[1] = -0.175603595979829;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 1.351207191959658; b[1] = -1.702414383919316;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.0;
           d = 2;
           velocity = 0;
       }
       else if (mdp.integrator==BABABAB)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 1.351207191959658; a[1] = -1.702414383919316;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.675603595979829; b[1] = -0.175603595979829;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.0;
           d = 2;
           velocity = 1;
       }
       else if (mdp.integrator==ABADABA)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.089775972994422; a[1] = 0.410224027005578;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.247597680043986; b[1] = 0.504804639912028;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.006911440413815;
           d = 2;
           velocity = 0;
       }
       else if (mdp.integrator==DABABAD)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.258529167713908; a[1] = 0.482941664572184;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.065274481323251; b[1] = 0.434725518676749;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.003595899064589; c[1] = 0.0;
           d = 2;
           velocity = 1;
       }
       else if (mdp.integrator==BADADAB)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.281473422092232; a[1] = 0.437053155815536;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.087960811032557; b[1] = 0.412039188967443;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.003060423791562;
           d = 2;
           velocity = 1;
       }
       else if (mdp.integrator==ADABADA)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.136458051118946; a[1] = 0.363541948881054;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.315267858070664; b[1] = 0.369464283858672;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.002427032834125; c[1] = 0.0;
           d = 2;
           velocity = 0;
       }
       else if (mdp.integrator==ADADADA)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.116438749543126; a[1] = 0.383561250456874;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.283216992495952; b[1] = 0.433566015008096;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.001247201195115; c[1] = 0.002974030329635;
           d = 2;
           velocity = 0;
       }
       else if (mdp.integrator==DADADAD)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.273005515864808; a[1] = 0.453988968270384;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.080128674198082; b[1] = 0.419871325801918;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.000271601364672; c[1] = 0.002959399979707;
           d = 2;
           velocity = 1;
       }
       else if (mdp.integrator==ABABABABA)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.178617895844809; a[1] = -0.066264582669818; a[2] = 0.775293373650018;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.712341831062606; b[1] = -0.212341831062606;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.0;
           d = 2;
           velocity = 0;
       }
       else if (mdp.integrator==BABABABAB)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.520943339103990; a[1] = -0.020943339103990;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.164498651557576; b[1] = 1.235692651138917; b[2] = -1.800382605392986;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
           d = 2;
           velocity = 1;
       }
       else if (mdp.integrator==BABADABAB)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.200395293638238; a[1] = 0.299604706361762;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.073943321445602; b[1] = 0.258244950046509; b[2] = 0.335623457015778;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.0; c[2] = 0.003147048491590;
           d = 2;
           velocity = 1;
       }
       else if (mdp.integrator==DABABABAD)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.190585159174513; a[1] = 0.309414840825487;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.036356798097337; b[1] = 0.340278911234329; b[2] = 0.246728581336668;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.002005691094612; c[1] = 0.0; c[2] = 0.0;
           d = 2;
           velocity = 1;
       }
       else if (mdp.integrator==BADABADAB)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.219039425103133; a[1] = 0.280960574896867;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.068466565514186; b[1] = 0.311000565033563; b[2] = 0.241065738904502;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.001602470431500; c[2] = 0.0;
           d = 2;
           velocity = 1;
       }
       else if (mdp.integrator==DABADABAD)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.197279141794602; a[1] = 0.302720858205398;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.060885008530668; b[1] = 0.288579639891554; b[2] = 0.301070703155556;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.000429756946246; c[1] = 0.0; c[2] = 0.002373498029145;
           d = 2;
           velocity = 1;
       }
       else if (mdp.integrator==ABADADABA)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.047802682977081; a[1] = 0.265994592108478; a[2] = 0.372405449828882;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.143282503449494; b[1] = 0.356717496550506;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.002065558490728;
           d = 2;
           velocity = 0;
       }
       else if (mdp.integrator==ADABABADA)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.118030603246046; a[1] = 0.295446189611111; a[2] = 0.173046414285686;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.273985556386628; b[1] = 0.226014443613372;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.001466561305710; c[1] = 0.0;
           d = 2;
           velocity = 0;
       }
       else if (mdp.integrator==DADABADAD)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.227758000273404; a[1] = 0.272241999726596;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.070935378258660; b[1] = 0.322911610232109; b[2] = 0.212306023018462;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.000067752132787; c[1] = 0.001597508440746; c[2] = 0.0;
           d = 2;
           velocity = 1;
       }
       else if (mdp.integrator==ADADADADA)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.094471605659163; a[1] = 0.281057227947299; a[2] = 0.248942332787076;
           d_b = 2; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.227712700174579; b[1] = 0.272287299825421;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.000577062053569; c[1] = 0.000817399268485;
           d = 2;
           velocity = 0;
       }
       else if (mdp.integrator==BADADADAB)
       {
           d_a = 2; a = (double *)malloc(d_a * sizeof(double)); a[0] = 1.079852426382431; a[1] = -0.579852426382431;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.359950808794144; b[1] = -0.143714727302654; b[2] = 0.567527837017021;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = -0.013965254224239; c[2] = -0.039247029382346;
           d = 2;
           velocity = 1;
       }
       else if (mdp.integrator==BABABABABAB)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.253978510841060; a[1] = -0.032302867652700; a[2] = 0.556648713623280;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.083983152628767; b[1] = 0.682236533571909; b[2] = -0.266219686200676;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
           d = 3;
           velocity = 1;
       }
       else if (mdp.integrator==ABABABABABA)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.275008121233242; a[1] = -0.134795009910679; a[2] = 0.359786888677437;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = -0.084429619507071; b[1] = 0.354900057157426; b[2] = 0.459059124699290;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.0; c[2] = 0.0;
           d = 3;
           velocity = 0;
       }
       else if (mdp.integrator==ABABADABABA)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.134257092137626; a[1] = -0.007010267216916; a[2] = 0.372753175079290;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = -0.485681409840328; b[1] = 0.767464037573892; b[2] = 0.436434744532872;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.0; c[2] = 0.002836723107629;
           d = 3;
           velocity = 0;
       }
       else if (mdp.integrator==DABABABABAD)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.282918304065611; a[1] = -0.002348009438292; a[2] = 0.438859410745362;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.080181913812571; b[1] = -1.372969015964262; b[2] = 1.792787102151691;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.000325098077953; c[1] = 0.0; c[2] = 0.0;
           d = 3;
           velocity = 1;
       }
       else if (mdp.integrator==ABADABADABA)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.062702644098210; a[1] = 0.193174566017780; a[2] = 0.244122789884010;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.149293739165427; b[1] = 0.220105234408407; b[2] = 0.261202052852332;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.000966194415594; c[2] = 0.0;
           d = 3;
           velocity = 0;
       }
       else if (mdp.integrator==BADABABADAB)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.201110227930330; a[1] = 0.200577842713366; a[2] = 0.196623858712608;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.065692416344302; b[1] = 0.264163604920340; b[2] = 0.170143978735358;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.001036943019757; c[2] = 0.0;
           d = 3;
           velocity = 1;
       }
       else if (mdp.integrator==ADABABABADA)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.115889910143319; a[1] = 0.388722377182381; a[2] = -0.004612287325700;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.282498420841510; b[1] = -0.625616553474143; b[2] = 1.686236265265266;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.001208219887746; c[1] = 0.0; c[2] = 0.0;
           d = 3;
           velocity = 0;
       }
       else if (mdp.integrator==BABADADABAB)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.122268182901557; a[1] = 0.203023211433263; a[2] = 0.349417211330360;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.055200549768959; b[1] = 0.127408150658963; b[2] = 0.317391299572078;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.0; c[2] = 0.001487834491987;
           d = 3;
           velocity = 1;
       }
       else if (mdp.integrator==ADABADABADA)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.083684971641549; a[1] = 0.225966488946428; a[2] = 0.190348539412023;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.199022868372193; b[1] = 0.197953981691206; b[2] = 0.206046299873202;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.000437056543403; c[1] = 0.0; c[2] = 0.000870457820984;
           d = 3;
           velocity = 0;
       }
       else if (mdp.integrator==DABADADABAD)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.068597474282941; a[1] = 0.284851197274498; a[2] = 0.293102656885122;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = -0.029456704762871; b[1] = 0.228751459942521; b[2] = 0.300705244820350;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.000410146066173; c[1] = 0.0; c[2] = 0.001249935251564;
           d = 3;
           velocity = 1;
       }
       else if (mdp.integrator==DADABABADAD)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.203263079324187; a[1] = 0.200698071607808; a[2] = 0.192077698136010;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.066202529912271; b[1] = 0.267856111220228; b[2] = 0.165941358867501;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.000012570620797; c[1] = 0.001042408779514; c[2] = 0.0;
           d = 3;
           velocity = 1;
       }
       else if (mdp.integrator==ADADABADADA)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.082541033171754; a[1] = 0.228637847036999; a[2] = 0.188821119791247;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.196785139280847; b[1] = 0.206783248777282; b[2] = 0.192863223883742;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.000317260402502; c[1] = 0.000555360763892; c[2] = 0.0;
           d = 3;
           velocity = 0;
       }
       else if (mdp.integrator==BADADADADAB)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.270990466773838; a[1] = 0.635374358266882; a[2] = -0.812729650081440;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.090330155591279; b[1] = 0.430978044876253; b[2] = -0.021308200467532;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.00263743598047; c[2] = -0.000586445610932;
           d = 3;
           velocity = 1;
       }
       else if (mdp.integrator==ADADADADADA)
       {
           d_a = 3; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.109534125980058; a[1] = 0.426279051773841; a[2] = -0.035813177753899;
           d_b = 3; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.268835839917653; b[1] = 0.529390037396794; b[2] = -0.596451754628894;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.000806354602850; c[1] = 0.007662601517364; c[2] = -0.011627206142396;
           d = 3;
           velocity = 0;
       }
       else if (mdp.integrator==BABABABABABABAB)
       {
           d_a = 4; a = (double *)malloc(d_a * sizeof(double)); a[0] = 0.246588187278614; a[1] = 0.604707387505781; a[2] = -0.400986903978801; a[3] = 0.099382658388812;
           d_b = 4; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.083333333333333; b[1] = 0.397767585954844; b[2] = -0.039333693144626; b[3] = 0.058232773856448;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.0; c[2] = 0.0; c[3] = 0.0;
           d = 4;
           velocity = 1;
       }
       else if (mdp.integrator==ABABABABABABABA)
       {
           d_a = 4; a = (double *)malloc(d_a * sizeof(double)); a[0] = -1.013087978917175; a[1] = 1.187429573732543; a[2] = -0.018335852096461; a[3] = 0.343994257281093;
           d_b = 4; b = (double *)malloc(d_b * sizeof(double)); b[0] = 0.000166006926500; b[1] = -0.379624214263774; b[2] = 0.689137411851811; b[3] = 0.380641590970926;
           c = (double *)malloc(d_b * sizeof(double)); c[0] = 0.0; c[1] = 0.0; c[2] = 0.0; c[3] = 0.0;
           d = 4;
           velocity = 0;
       }
       else
       {
           a = (double *)malloc(sizeof(double)); a[0] = 0.0;
           b = (double *)malloc(sizeof(double)); b[0] = 0.0;
           c = (double *)malloc(sizeof(double)); c[0] = 0.0;
       }
       
       /* setup integrator based on defined vectors a,b,c*/
       
       if (velocity==1) /* velocity version, starting with momentum or FG update */
       {
           if (c[0] == 0)
           {
               copy_steps(n,b[0],ws,s,0);
               s+=n;
           }
           else
           {
               copy_steps_fg(n,2*c[0]/b[0],ws,s);
               s+=n;
               
               (*s).iop = itu-2;
               (*s).eps = 1.0;
               (*s).lvl_id=ilv;
               s+=1;
               
               copy_steps(n,b[0],ws,s,1);
               s+=n;
           }
           
           for (i=1;i<=nstep;i++)
           {   
               if (c[0]!=0)
               {
                   (*s).iop = itu-1;
                   (*s).eps = 0.0;
                   (*s).lvl_id = ilv;
                   s+=1;
               }
               else 
               {
                   (*s).iop = itu-3;
                   (*s).eps = b[0]*eps;
                   (*s).lvl_id = -1;
                   s+=1;
               }
               
               (*s).iop = itu;
               (*s).eps = a[0]*eps;
               (*s).lvl_id=-1;
               s+=1;
               
               for (j=1;j<d;j++)
               {
                   if (c[j] == 0)
                   {
                       copy_steps(n,b[j],ws,s,0);
                       s+=n;
                   }
                   else
                   {
                       copy_steps_fg(n,2*c[j]/b[j],ws,s);
                       s+=n;
                       
                       (*s).iop = itu-2;
                       (*s).eps = 1.0;
                       (*s).lvl_id = ilv;
                       s+=1;
                       
                       copy_steps(n,b[j],ws,s,1);
                       s+=n;
                   }
                       
                   if (c[j]!=0)
                   {
                       (*s).iop = itu-1;
                       (*s).eps = 0.0;
                       (*s).lvl_id = ilv;
                       s+=1;
                   }
                   else 
                   {
                       (*s).iop = itu-3;
                       (*s).eps = b[j]*eps;
                       (*s).lvl_id=-1;
                       s+=1;
                   }
                   
                   (*s).iop = itu;
                   (*s).eps = a[j]*eps;
                   (*s).lvl_id = -1;
                   s+=1;
               }
               if (d_a < d_b)
               {
                   if (c[d_b-1] == 0)
                   {
                       copy_steps(n,b[d_b-1],ws,s,0);
                       s+=n;
                   }
                   else
                   {
                       copy_steps_fg(n,2*c[d_b-1]/b[d_b-1],ws,s);
                       s+=n;
                       
                       (*s).iop = itu-2;
                       (*s).eps = 1.0;
                       (*s).lvl_id = ilv;
                       s+=1;
                       
                       copy_steps(n,b[d_b-1],ws,s,1);
                       s+=n;
                   }
                   
                   if (c[d_b-1]!=0)
                   {
                       (*s).iop = itu-1;
                       (*s).eps = 0.0;
                       (*s).lvl_id = ilv;
                       s+=1;
                   }
                   else 
                   {
                       (*s).iop = itu-3;
                       (*s).eps = b[d_b-1]*eps;
                       (*s).lvl_id = -1;
                       s+=1;
                   }
               }
               for (j=d-1;j>0;j--)
               {
                   if ( (d_a == d_b && j==d-1)  == 0)
                   {
                       (*s).iop = itu;
                       (*s).eps = a[j]*eps;
                       (*s).lvl_id = -1;
                       s+=1;
                   }
                   
                   if (c[j] == 0)
                   {
                       copy_steps(n,b[j],ws,s,0);
                       s+=n;
                   }
                   else
                   {
                       copy_steps_fg(n,2*c[j]/b[j],ws,s);
                       s+=n;
                       
                       (*s).iop = itu-2;
                       (*s).eps = 1.0;
                       (*s).lvl_id = ilv;
                       s+=1;
                       
                       copy_steps(n,b[j],ws,s,1);
                       s+=n;
                   }
                       
                   if (c[j]!=0)
                   {
                       (*s).iop = itu-1;
                       (*s).eps = 0.0;
                       (*s).lvl_id = ilv;
                       s+=1;
                   }
                   else 
                   {
                       (*s).iop = itu-3;
                       (*s).eps = b[j]*eps;
                       (*s).lvl_id = -1;
                       s+=1;
                   }
               }

               if (d_a > 1 || d_a != d_b)               
               {
                   (*s).iop = itu;
                   (*s).eps = a[0]*eps;
                   (*s).lvl_id = -1;
                   s+=1;
               }
               
               if (i<nstep)
               {
                   if (c[0] == 0)
                   {
                       copy_steps(n,2*b[0],ws,s,0);
                       s+=n;
                   }
                   else
                   {
                       copy_steps_fg(n,2*c[0]/b[0],ws,s);
                       s+=n;
                       
                       (*s).iop = itu-2;
                       (*s).eps = 1.0;
                       (*s).lvl_id = ilv;
                       s+=1;
                       
                       copy_steps(n,2*b[0],ws,s,1);
                       s+=n;
                   }
               }
               else
               {
                   if (c[0] == 0)
                   {
                       copy_steps(n,b[0],ws,s,0);
                       s+=n;
                   }
                   else
                   {
                       copy_steps_fg(n,2*c[0]/b[0],ws,s);
                       s+=n;
                       
                       (*s).iop = itu-2;
                       (*s).eps = 1.0;
		       (*s).lvl_id = ilv;
                       s+=1;
                       
                       copy_steps(n,b[0],ws,s,1);
                       s+=n;
                   }
               }
           }
           
           if (c[0]!=0)
           {
               (*s).iop = itu-1;
               (*s).eps = 0.0;
               (*s).lvl_id = ilv;
               s+=1;
           }
           else 
           {
               (*s).iop = itu-3;
               (*s).eps = b[0]*eps;
               (*s).lvl_id = -1;
               s+=1;
           }
       }
       else /* position version, starting with link update */
       {
           (*s).iop = itu;
           (*s).eps = a[0]*eps;
           (*s).lvl_id = -1;
           s+=1;
           
           for (i=1;i<=nstep;i++)
           {
               if (c[0] == 0)
               {
                   copy_steps(n,b[0],ws,s,0);
                   s+=n;
               }
               else
               {
                   copy_steps_fg(n,2*c[0]/b[0],ws,s);
                   s+=n;
                   
                   (*s).iop = itu-2;
                   (*s).eps = 1.0;
                   (*s).lvl_id = ilv;
                   s+=1;
                   
                   copy_steps(n,b[0],ws,s,1);
                   s+=n;
               }
                   
               if (c[0]!=0)
               {
                   (*s).iop = itu-1;
                   (*s).eps = 0.0;
                   (*s).lvl_id = ilv;
                   s+=1;
               }
               else 
               {
                   (*s).iop = itu-3;
                   (*s).eps = b[0]*eps;
                   (*s).lvl_id = -1;
                   s+=1;
               }
               
               for (j=1;j<d;j++)
               {
                   (*s).iop = itu;
                   (*s).eps = a[j]*eps;
                   (*s).lvl_id = -1;
                   s+=1;
                   
                   if (c[j] == 0)
                   {
                       copy_steps(n,b[j],ws,s,0);
                       s+=n;
                   }
                   else
                   {
                       copy_steps_fg(n,2*c[j]/b[j],ws,s);
                       s+=n;
                       
                       (*s).iop = itu-2;
                       (*s).eps = 1.0;
                       (*s).lvl_id = ilv;
                       s+=1;
                       
                       copy_steps(n,b[j],ws,s,1);
                       s+=n;
                   }
                       
                   if (c[j]!=0)
                   {
                       (*s).iop = itu-1;
                       (*s).eps = 0.0;
                       (*s).lvl_id = ilv;
                       s+=1;
                   }
                   else 
                   {
                       (*s).iop = itu-3;
                       (*s).eps = b[j]*eps;
                       (*s).lvl_id = -1;
                       s+=1;
                   }
               }
               if (d_b < d_a)
               {
                   (*s).iop = itu;
                   (*s).eps = a[d_a-1]*eps;
                   (*s).lvl_id = -1;
                   s+=1;
               }
               for (j=d-1;j>0;j--)
               {
                   if ((d_a == d_b && j==d-1) == 0)
                   {
                       if (c[j] == 0)
                       {
                           copy_steps(n,b[j],ws,s,0);
                           s+=n;
                       }
                       else
                       {
                           copy_steps_fg(n,2*c[j]/b[j],ws,s);
                           s+=n;
                           
                           (*s).iop = itu-2;
                           (*s).eps = 1.0;
                           (*s).lvl_id = ilv;
                           s+=1;
                           
                           copy_steps(n,b[j],ws,s,1);
                           s+=n;
                       }
                           
                       if (c[j]!=0)
                       {
                           (*s).iop = itu-1;
                           (*s).eps = 0.0;
                           (*s).lvl_id = ilv;
                           s+=1;
                       }
                       else 
                       {
                           (*s).iop = itu-3;
                           (*s).eps = b[j]*eps;
                           (*s).lvl_id = -1;
                           s+=1;
                       }
                   }
                   
                   (*s).iop = itu;
                   (*s).eps = a[j]*eps;
                   (*s).lvl_id = -1; 
                   s+=1;
               }

               if (d_b > 1 || d_a != d_b)
	       {
                   if (c[0] == 0)
                   {
                       copy_steps(n,b[0],ws,s,0);
                       s+=n;
                   }
                   else
                   {
                       copy_steps_fg(n,2*c[0]/b[0],ws,s);
                       s+=n;
                   
                       (*s).iop = itu-2;
                       (*s).eps = 1.0;
                       (*s).lvl_id = ilv;
                       s+=1;
                   
                       copy_steps(n,b[0],ws,s,1);
                       s+=n;
                   }
                   
                   if (c[0]!=0)
                   {
                       (*s).iop = itu-1;
                       (*s).eps = 0.0;
                       (*s).lvl_id = ilv;
                       s+=1;
                   }
                   else 
                   {
                       (*s).iop = itu-3;
                       (*s).eps = b[0]*eps;
                       (*s).lvl_id = -1;
                       s+=1;
                   }
               }
               
               if (i<nstep)
               {
                   (*s).iop = itu;
                   (*s).eps = 2*a[0]*eps;
                   (*s).lvl_id = -1;
                   s+=1;
               }
               else
               {
                   (*s).iop = itu;
                   (*s).eps = a[0]*eps;
                   (*s).lvl_id = -1;
                   s+=1;
               }
           }
       }
       free(a);
       free(b);
       free(c);
   }
}


static int nfrc_steps(mdstep_t *s)
{
   int itu,n;

   itu=iend-1;
   n=0;

   while (s[n].iop<itu) {
       n+=1;
   }

   return n;
}


static int nall_steps(mdstep_t *s)
{
   int n;

   n=0;

   while (s[n].iop<iend)
      n+=1;

   return n;
}


static void add_frc_steps(double c,mdstep_t *s,mdstep_t *r)
{
   /*int n,m,i;

   n=nfrc_steps(s);
   m=nfrc_steps(r);

   for (i=0;i<n;i++)
   {
	 r[m].iop=s[i].iop;
	 r[m].eps=c*s[i].eps;
	 r[m].lvl_id=s[i].lvl_id;
	 m+=1;
   }*/
   int n,m,i,j;
   int index_mom_update_s, index_mom_update_r;
   mdstep_t *tmp;
   n=nfrc_steps(s);
   m=nfrc_steps(r);

   for(i=0,i<n,i++)
   {	   
   	if(s[i].lvl_id != -1)
   	{	   
	   index_mom_update_s = i-1;
	   break;
	}
   }
   for(i=0,i<m,i++)
   {	   
   	if(r[i].lvl_id != -1)
   	{	   
	   index_mom_update_r = i-1;
	   break;
	}
   }

   /* tmp will store the momentum update, as well as any operations belonging to force-gradient updates from r. 
      It will moreover already contain the momentum update from s, if it exists. */	
   tmp = (mdstep_t *)malloc((m - index_mom_update_r) * sizeof(mdstep_t));

   if (index_mom_update_r >= 0)
   {
	tmp[0].iop = r[index_mom_update_r].iop;
	tmp[0].eps = r[index_mom_update_r].eps;
	tmp[0].lvl_id = r[index_mom_update_r].lvl_id;
	if (index_mom_update s>=0)
	{
   		tmp[0].eps += c*s[index_mom_update_s].eps;
	}	
   }   
   else if (index_mom_update_s >= 0)
   {
	tmp[0].iop = s[index_mom_update_s].iop;
	tmp[0].eps = c*s[index_mom_update_s].eps;
	tmp[0].lvl_id = s[index_mom_update_s].lvl_id;   
   }   

   for (i=1;i<m-index_mom_update_r;i++)
   {
	tmp[i].iop = r[i+index_mom_update_r].iop;
	tmp[i].eps = r[i+index_mom_update_r].eps;
	tmp[i].lvl_id = r[i+index_mom_update_r].lvl_id;
   }

   /* in a next step, we will add all force updates from s to r that do not belong to any force-gradient update */
   for (i=0;i<index_mom_update_s;i++)
   {	   
   	for (j=0;j<index_mom_update_r;j++)
	{
		if (r[j].iop == s[i].iop)
		{
			r[j].eps += c*s[i].eps;
			break;
		}	
	}
	if (j==index_mom_update_r)
	{
		r[j].iop = s[i].iop;
		r[j].eps = c*s[i].eps;
		r[j].lvl_id = s[i].lvl_id;
		index_mom_update_r+=1;
	}
   }

   /* next, we will append the operations stored in tmp to r */
   for (i=0;i<m-index_mom_update_r;i++)
   {
	r[index_mom_update_r].iop = tmp[i].iop;
	r[index_mom_update_r].eps = tmp[i].eps;
	r[index_mom_update_r].lvl_id = tmp[i].lvl_id;
	index_mom_update_r+=1;
   }
   free(tmp);	

   /* finally, we will append the operations from s that belong to force-gradient updates */
   for (i=index_mom_update_s+1;i<n;i++)
   {
	r[index_mom_update_r].iop = s[i].iop;
	r[index_mom_update_r].eps = c*s[i].eps;
	r[index_mom_update_r].lvl_id = s[i].lvl_id;
	index_mom_update_r+=1;
   }
}


static void insert_level(mdstep_t *s1,mdstep_t *s2,mdstep_t *r)
{
   int itu,nfrc,nall;
   double eps;

   set_steps2zero(nsmx,r);

   itu=iend-1;
   nfrc=nfrc_steps(s1);
   nall=nall_steps(s1+nfrc);

   add_frc_steps(1.0,s2,r);
   s2+=nfrc_steps(s2);

   while ((*s2).iop==itu)
   {
      eps=(*s2).eps;
      add_frc_steps(eps,s1,r);
      r+=nfrc_steps(r);
      copy_steps(nall,eps,s1+nfrc,r,1);
      r+=nall-nfrc;

      s2+=1;
      add_frc_steps(1.0,s2,r);
      s2+=nfrc_steps(s2);
   }
}

static void set_nlv(int *nlv,double *tau)
{
   hmc_parms_t hmc;
   smd_parms_t smd;

   hmc=hmc_parms();
   smd=smd_parms();

   if (hmc.nlv!=0)
   {
      (*nlv)=hmc.nlv;
      (*tau)=hmc.tau;
   }
   else if (smd.nlv!=0)
   {
      (*nlv)=smd.nlv;
      (*tau)=smd.eps;
   }
   else
      error(1,1,"set_nlv [mdsteps.c]","Simulation parameters are not set");
}


void set_mdsteps(void)
{
   int nlv,ilv,n;
   double tau;

   set_nlv(&nlv,&tau);
   set_nsmx(nlv);
   alloc_mds();
   expand_level(nlv-1,tau,mds,mdw[0]);

   for (ilv=(nlv-2);ilv>=0;ilv--)
   {
      n=nall_steps(mds);
      copy_steps(n,1.0,mds,mdw[0],1);
      expand_level(ilv,1.0,mdw[1],mdw[2]);
      insert_level(mdw[1],mdw[0],mds);
   }

   nmds=nall_steps(mds)+1;
}


mdstep_t *mdsteps(int *nop,int *itu)
{
   (*nop)=nmds;
   (*itu)=iend-1;

   return mds;
}


static void print_ops(void)
{
   int i,itu;
   double t;

   printf("List of elementary operations:\n");
   itu=iend-1;
   t=0.0;

   for (i=0;i<nmds;i++)
   {  
      if (mds[i].iop<itu-3)
      { 
          if (mds[i].lvl_id==-1)
          {
              printf("TP: force %2d, eps = % .2e, t = %.2e\n",
                   mds[i].iop,mds[i].eps,t);
          }
          else 
          {
              printf("FG: force %2d, level = %2d, eps = % .2e, t= %.2e\n",
                   mds[i].iop,mds[i].lvl_id,mds[i].eps,t);
          }
      }
      else if (mds[i].iop == itu-3)
          printf("momentum-update\n");
      else if (mds[i].iop == itu-2)
      {
          printf("creation of temporary link field\n");
          i+=1;
          while (mds[i].iop != itu-1)
          {
              printf("TP: force %2d, eps = % .2e, t = %.2e\n",
                   mds[i].iop,mds[i].eps,t);
              i+=1;
          }
          printf("momentum update + restoring original link field\n");
      }
      else if (mds[i].iop == itu-1)
          printf("momentum update + restoring original link field\n");
      else if (mds[i].iop==itu)
      {
         printf("TU:           eps = % .2e, t = %.2e\n",mds[i].eps,t);
         t+=mds[i].eps;
      }
      else if (mds[i].iop==iend)
         printf("END\n\n");
      else {
          printf("iend = %d, itu = %d, operation index = %d",iend,itu,mds[i].iop);
          error_root(1,1,"print_ops [mdsteps.c]","Unknown operation");
      }
   }
}


static void print_times(double tau)
{
   int i,j,it,COUNT_FG_FORCES;
   double seps;

   COUNT_FG_FORCES = 0; /* is a force update with lvl_id > 0 a time step? 0 -> no, 1 -> yes */

   printf("Total integration times:\n");

   for (i=0;i<iend-4;i++)
   {
      it=0;
      seps=0.0;

      for (j=0;j<nmds;j++)
      {
         if (mds[j].iop==iend-3)
         {
             COUNT_FG_FORCES = 1; /* upcoming force updates with lvl_id > 0 are time steps of size b[j]*eps */    
         }
         else if (mds[j].iop==iend-2)
         {
             COUNT_FG_FORCES = 0; /* upcoming force updates with lvl_id >0 are temporary updates of size 2*c[j]/b[j]*eps and will be ignored */
         }
         else if (mds[j].iop==i)
         {
            if (mds[j].lvl_id == -1 || COUNT_FG_FORCES == 1)
            {
                it=1;
                seps+=mds[j].eps;
            }
         }
      }

      seps/=tau;

      if (it==1)
         printf("Force %2d: sum(eps)/tau = %.3e\n",i,seps);
   }
   seps=0.0;
   for (j=0;j<nmds;j++)
   {
	if (mds[j].iop==iend-1)
        {
  	    seps+=mds[j].eps;		
        }
   }
   seps/=tau;
   printf("TU:       sum(eps)/tau = %.3e\n",seps);

   printf("\n");
}


void print_mdsteps(int ipr)
{
   int my_rank,nlv;
   double tau;

   set_nlv(&nlv,&tau);

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      printf("Molecular-dynamics integrator:\n\n");

      printf("Trajectory length tau = %.4e\n",tau);
      printf("Number of levels = %d\n\n",nlv);

      print_mdint_parms();

      if (ipr&0x1)
         print_force_parms(0x1);

      if (ipr&0x2)
         print_ops();

      if (ipr&0x4)
         print_times(tau);
   }
}
