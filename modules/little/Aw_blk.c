
/*******************************************************************************
*
* File Aw_blk.c
*
* Copyright (C) 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the diagonal terms of the little Dirac operator.
*
*   void set_Awblk(double mu)
*     Computes the diagonal terms of the double-precision little Dirac
*     operator. Apart from the twisted mass mu, the parameters of the
*     little Dirac operator are set to the values returned by sw_parms()
*     (see flags/lat_parms.c).
*
*   int update_Awblk(double mu)
*     Updates the diagonal terms of the double-precision little Dirac
*     operator after a change of the twisted mass or the other parameters
*     at fixed gauge field. The program returns 0 if the diagonal terms
*     did not need to be updated and 1 otherwise.
*
* For a description of the little Dirac operator and the associated data
* structures see README.Aw.
*
* The programs in this module are called by the programs in Aw_ops.c and are
* not intended to be called from anywhere else (the prototypes in little.h
* are masked accordingly). They are assumed to be called by the OpenMP master
* thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define AW_BLK_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "flags.h"
#include "utils.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "block.h"
#include "dfl.h"
#include "little.h"
#include "global.h"

#define MAX_UPDATE 128

static int old_eo,nupd=0;
static double old_m0,old_mu;
static complex_dble **Smat,**Pmat=NULL;


static void alloc_Pmat(int Ns,int nb)
{
   int isw,n;
   complex_dble *wd;
   lat_parms_t lat;

   lat=lat_parms();
   isw=lat.isw;

   if (isw==1)
   {
      Pmat=malloc(nb*sizeof(*Pmat));
      Smat=NULL;
      wd=amalloc(nb*Ns*Ns*sizeof(*wd),6);

      for (n=0;n<nb;n++)
      {
         Pmat[n]=wd;
         wd+=Ns*Ns;
      }
   }
   else
   {
      Pmat=malloc(2*nb*sizeof(*Pmat));
      Smat=Pmat+nb;
      wd=amalloc(2*nb*Ns*Ns*sizeof(*wd),6);

      for (n=0;n<(2*nb);n++)
      {
         Pmat[n]=wd;
         wd+=Ns*Ns;
      }
   }

   error((Pmat==NULL)||(wd==NULL),1,"alloc_Pmat [Aw_blk.c]",
         "Unable to allocate auxiliary matrix arrays");
}


static void set_Awblk_SP(int isp,double mu)
{
   int Ns,nb,isw,nbh,vol,volh;
   int k,l,j,n;
   complex_dble *S,*P,*A,**Ablk;
   complex_qflt cqsm;
   spinor **s;
   spinor_dble **sd;
   block_t *b;
   Aw_dble_t Aw;
   tm_parms_t tm;
   sw_parms_t swp;

   Aw=Awop_dble();
   Ns=Aw.Ns;
   Ablk=Aw.Ablk;

   if ((query_flags(SWD_UP2DATE)!=1)||
       (query_flags(SWD_E_INVERTED)==1)||
       (query_flags(SWD_O_INVERTED)==1))
      sw_term(NO_PTS);

   b=blk_list(DFL_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;
   volh=vol/2;

   if (Pmat==NULL)
      alloc_Pmat(Ns,nb);

#pragma omp parallel private(k,l,j,n,cqsm,s,sd,S,P,A)
   {
      k=omp_get_thread_num();
      S=NULL;
      P=NULL;

      for (n=k;n<nb;n+=NTHREAD)
      {
         assign_ud2udblk(DFL_BLOCKS,n);
         assign_swd2swdblk(DFL_BLOCKS,n);

         s=b[n].s;
         sd=b[n].sd;

         if (n<nbh)
         {
            if (isp)
            {
               if (Smat!=NULL)
                  S=Smat[n+isw*nbh];
               P=Pmat[n+isw*nbh];
            }

            A=Ablk[n+isw*nbh];
         }
         else
         {
            if (isp)
            {
               if (Smat!=NULL)
                  S=Smat[n-isw*nbh];
               P=Pmat[n-isw*nbh];
            }

            A=Ablk[n-isw*nbh];
         }

         for (l=0;l<Ns;l++)
         {
            assign_s2sd(vol,0,s[l+1],sd[0]);
            Dw_blk_dble(DFL_BLOCKS,n,mu,0,1);

            cqsm=spinor_prod_dble(vol,0,sd[0],sd[1]);
            A[Ns*l+l].re=cqsm.re.q[0];
            A[Ns*l+l].im=cqsm.im.q[0];

            for (j=0;j<Ns;j++)
            {
               if (j!=l)
               {
                  assign_s2sd(vol,0,s[j+1],sd[0]);
                  cqsm=spinor_prod_dble(vol,0,sd[0],sd[1]);
                  A[Ns*j+l].re=cqsm.re.q[0];
                  A[Ns*j+l].im=cqsm.im.q[0];
               }
            }

            if (isp)
            {
               assign_s2sd(vol,0,s[l+1],sd[0]);

               cqsm=spinor_prod5_dble(volh,0,sd[0],sd[0]);
               P[Ns*l+l].re=cqsm.re.q[0];

               cqsm=spinor_prod5_dble(volh,0,sd[0]+volh,sd[0]+volh);
               P[Ns*l+l].im=cqsm.re.q[0];

               if (S!=NULL)
               {
                  cqsm.re=norm_square_dble(vol,0,sd[0]);
                  S[Ns*l+l].re=cqsm.re.q[0];
                  S[Ns*l+l].im=0.0;
               }

               for (j=0;j<l;j++)
               {
                  assign_s2sd(vol,0,s[j+1],sd[1]);

                  cqsm=spinor_prod5_dble(volh,0,sd[0],sd[1]);
                  P[Ns*l+j].re=cqsm.re.q[0];
                  P[Ns*l+j].im=cqsm.im.q[0];
                  P[Ns*j+l].re=cqsm.re.q[0];
                  P[Ns*j+l].im=-cqsm.im.q[0];

                  cqsm=spinor_prod5_dble(volh,0,sd[0]+volh,sd[1]+volh);
                  P[Ns*l+j].re-=cqsm.im.q[0];
                  P[Ns*l+j].im+=cqsm.re.q[0];
                  P[Ns*j+l].re+=cqsm.im.q[0];
                  P[Ns*j+l].im+=cqsm.re.q[0];

                  if (S!=NULL)
                  {
                     cqsm=spinor_prod_dble(vol,0,sd[0],sd[1]);
                     S[Ns*l+j].re=cqsm.re.q[0];
                     S[Ns*l+j].im=cqsm.im.q[0];
                     S[Ns*j+l].re=cqsm.re.q[0];
                     S[Ns*j+l].im=-cqsm.im.q[0];
                  }
               }
            }
         }
      }
   }

   tm=tm_parms();
   swp=sw_parms();

   nupd=0;
   old_eo=tm.eoflg;
   old_m0=swp.m0;
   old_mu=mu;
}


void set_Awblk(double mu)
{
   set_Awblk_SP(1,mu);
}


int update_Awblk(double mu)
{
   int Ns,nb,isw,nbh,nbt,eo;
   int k,l,j,n;
   double m0,dm0,dme,dmo;
   complex_dble z,*S,*P,*A,**Ablk;
   sw_parms_t swp;
   tm_parms_t tm;
   Aw_dble_t Aw;

   tm=tm_parms();
   swp=sw_parms();
   eo=tm.eoflg;
   m0=swp.m0;

   if ((eo==old_eo)&&(m0==old_m0)&&(mu==old_mu))
      return 0;

   dm0=m0-old_m0;
   dme=mu-old_mu;

   if (eo==1)
   {
      if (old_eo==1)
         dmo=0.0;
      else
         dmo=-old_mu;
   }
   else
   {
      if (old_eo==1)
         dmo=mu;
      else
         dmo=dme;
   }

   if ((nupd<MAX_UPDATE)&&
       (fabs(dm0)<1.0)&&(fabs(dme)<1.0)&&(fabs(dmo)<1.0)&&
       ((swp.isw==0)||(dm0==0.0)))
   {
      Aw=Awop_dble();
      Ns=Aw.Ns;
      Ablk=Aw.Ablk;

      (void)(blk_list(DFL_BLOCKS,&nb,&isw));
      nbh=nb/2;
      nbt=nb/NTHREAD;

#pragma omp parallel private(k,l,j,n,z,S,P,A)
      {
         k=omp_get_thread_num();
         S=NULL;

         for (n=k*nbt;n<(k+1)*nbt;n++)
         {
            if (n<nbh)
            {
               if (Smat!=NULL)
                  S=Smat[n+isw*nbh];
               P=Pmat[n+isw*nbh];
               A=Ablk[n+isw*nbh];
            }
            else
            {
               if (Smat!=NULL)
                  S=Smat[n-isw*nbh];
               P=Pmat[n-isw*nbh];
               A=Ablk[n-isw*nbh];
            }

            for (j=0;j<Ns;j++)
            {
               if (dm0!=0.0)
               {
                  A[Ns*j+j].re+=dm0*S[Ns*j+j].re;

                  for (l=0;l<j;l++)
                  {
                     A[Ns*j+l].re+=dm0*S[Ns*j+l].re;
                     A[Ns*j+l].im+=dm0*S[Ns*j+l].im;
                     A[Ns*l+j].re+=dm0*S[Ns*l+j].re;
                     A[Ns*l+j].im+=dm0*S[Ns*l+j].im;
                  }
               }

               if (dme!=0.0)
               {
                  A[Ns*j+j].im+=dme*P[Ns*j+j].re;

                  for (l=0;l<j;l++)
                  {
                     z.re=0.5*dme*(P[l*Ns+j].im-P[j*Ns+l].im);
                     z.im=0.5*dme*(P[l*Ns+j].re+P[j*Ns+l].re);

                     A[Ns*j+l].re+=z.re;
                     A[Ns*j+l].im+=z.im;
                     A[Ns*l+j].re-=z.re;
                     A[Ns*l+j].im+=z.im;
                  }
               }

               if (dmo!=0.0)
               {
                  A[Ns*j+j].im+=dmo*P[Ns*j+j].im;

                  for (l=0;l<j;l++)
                  {
                     z.re=0.5*dmo*(P[j*Ns+l].re-P[l*Ns+j].re);
                     z.im=0.5*dmo*(P[j*Ns+l].im+P[l*Ns+j].im);

                     A[Ns*j+l].re+=z.re;
                     A[Ns*j+l].im+=z.im;
                     A[Ns*l+j].re-=z.re;
                     A[Ns*l+j].im+=z.im;
                  }
               }
            }
         }
      }

      nupd+=1;
      old_eo=eo;
      old_m0=m0;
      old_mu=mu;
   }
   else
      set_Awblk_SP(0,mu);

   return 1;
}
