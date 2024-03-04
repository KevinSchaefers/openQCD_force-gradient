
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2015, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the DFT and DFT4D parameter setting.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "utils.h"
#include "lattice.h"
#include "dft.h"
#include "global.h"

static const int nproc[4]={NPROC0,NPROC1,NPROC2,NPROC3};
static int ns[6]={10,16,24,32,40,48};


static int cmp_bit(int p,int a,int b)
{
   int ia,ib;

   while (p>0)
   {
      ia=a&0x1;
      ib=b&0x1;

      if (ia>ib)
         return 1;
      else if (ia<ib)
         return -1;

      p-=1;
      a>>=1;
      b>>=1;
   }

   return 0;
}


static void check_r(dft_parms_t *dp)
{
   int n,p,m;
   int i,j,r1,r2;

   n=(*dp).n;
   if ((*dp).type!=EXP)
      n*=2;

   p=0;
   m=n;

   while ((m>4)&&((m%2)==0))
   {
      p+=1;
      m/=2;
   }

   for (i=0;i<n;i++)
   {
      j=(*dp).r[i];
      error((j<0)||(j>=n),1,"check_r [check1.c]",
            "Incorrect range of reordering array");

      if (i<(n-1))
         error((*dp).r[i+1]==j,1,"check_r [check1.c]",
               "Incorrect range of reordering array");
   }

   j=0;

   for (i=m;i<=n;i+=m)
   {
      while (j<(i-1))
      {
         r1=(*dp).r[j];
         r2=(*dp).r[j+1];

         error((cmp_bit(p,r1,r2)!=0)||(r1>=r2),1,
               "check_r [check1.c]","Incorrect reordering array");
         j+=1;
      }

      if (i<n)
      {
         r1=(*dp).r[j];
         r2=(*dp).r[j+1];

         error(cmp_bit(p,r1,r2)!=-1,2,
               "check_r [check1.c]","Incorrect reordering array");
         j+=1;
      }
   }
}


static void check_w(dft_parms_t *dp)
{
   int n,b,c,d,*r;
   int i,nh;
   double del,dmx,pi,phi;
   complex_dble *w,*wb,*wc,*iwb,*iwc;

   n=(*dp).n;
   b=(*dp).b;
   c=(*dp).c;
   d=((*dp).type==SIN);
   r=(*dp).r;

   w=(*dp).w;
   wb=(*dp).wb;
   wc=(*dp).wc;
   iwb=(*dp).iwb;
   iwc=(*dp).iwc;

   if ((*dp).type!=EXP)
      n*=2;

   nh=n/2;
   pi=4.0*atan(1.0);
   dmx=0.0;

   for (i=0;i<=n;i++)
   {
      phi=(double)(i)*2.0*pi/(double)(n);
      del=fabs(w[i].re-cos(phi))+fabs(w[i].im-sin(phi));

      if (del>dmx)
         dmx=del;
   }

   for (i=0;i<n;i++)
   {
      phi=(double)(b*r[i])*pi/(double)(n);
      del=fabs(wb[i].re-cos(phi))+fabs(wb[i].im-sin(phi));

      if ((*dp).type!=EXP)
      {
         if (((r[i]==0)&&(c==0)&&(d==1))||((r[i]==nh)&&(c==0)&&((b+d)==1)))
            del=fabs(wb[i].re)+fabs(wb[i].im);
         else if ((r[i]>=nh)&&((b+d)==1))
            del=fabs(wb[i].re+cos(phi))+fabs(wb[i].im+sin(phi));
      }

      if (del>dmx)
         dmx=del;
   }

   for (i=0;i<n;i++)
   {
      phi=(double)(c*(2*i+b))*0.5*pi/(double)(n);
      del=fabs(wc[i].re-cos(phi))+fabs(wc[i].im-sin(phi));

      if (del>dmx)
         dmx=del;
   }

   for (i=0;i<n;i++)
   {
      phi=-(double)(b*(2*i+c))*0.5*pi/(double)(n);
      del=fabs(iwb[i].re-cos(phi))+fabs(iwb[i].im-sin(phi));

      if (del>dmx)
         dmx=del;
   }

   for (i=0;i<n;i++)
   {
      phi=-(double)(c*r[i])*pi/(double)(n);
      del=fabs(iwc[i].re-cos(phi)/(double)(n))+
         fabs(iwc[i].im-sin(phi)/(double)(n));

      if ((*dp).type!=EXP)
      {
         if (((r[i]==0)&&(b==0)&&(d==1))||((r[i]==nh)&&(b==0)&&((c+d)==1)))
            del=fabs(iwc[i].re)+fabs(iwc[i].im);
         else if ((r[i]>=nh)&&((c+d)==1))
            del=fabs(iwc[i].re+cos(phi)/(double)(n))+
               fabs(iwc[i].im+sin(phi)/(double)(n));
      }

      if (del>dmx)
         dmx=del;
   }

   error(dmx>(16.0*DBL_EPSILON),1,"check_w [check1.c]",
         "Incorrect w arrays");
}


static void check_dft4d_parms(void)
{
   int idp[4],nx[4],csize;
   int mu,i,id,np,nf,m,r,ie;
   dft4d_parms_t *dp;

   csize=8;

   idp[0]=set_dft_parms(SIN,NPROC0*L0,0,1);
   idp[1]=set_dft_parms(EXP,NPROC1*L1,0,0);
   idp[2]=set_dft_parms(EXP,NPROC2*L2,0,0);
   idp[3]=set_dft_parms(EXP,NPROC3*L3,0,0);

   if (cpr[0]==NPROC0-1)
      nx[0]=L0+1;
   else
      nx[0]=L0;

   nx[1]=L1;
   nx[2]=L2;
   nx[3]=L3;

   id=set_dft4d_parms(idp,nx,csize);
   dp=dft4d_parms(id);
   ie=0;
   ie|=(csize!=(*dp).csize);

   for (mu=0;mu<4;mu++)
   {
      ie|=(nx[mu]!=(*dp).nx[mu][cpr[mu]]);
      ie|=(((nx[0]*nx[1]*nx[2]*nx[3])/nx[mu])!=(*dp).ny[mu]);
      ie|=((csize*(*dp).ny[mu])!=(*dp).nf[mu]);

      np=nproc[mu];
      nf=(*dp).nf[mu];
      m=nf/np;
      r=nf%np;

      for (i=0;i<np;i++)
      {
         if (mu!=0)
            ie|=((*dp).nx[mu][i]!=nx[mu]);
         else if (i==(np-1))
            ie|=((*dp).nx[mu][i]!=(L0+1));
         else
            ie|=((*dp).nx[mu][i]!=L0);

         ie|=((*dp).mf[mu][i]!=(m+(i<r)));
      }

      ie|=(dft_parms(idp[mu])!=(*dp).dp[mu]);
   }

   error(ie!=0,1,"check_dft4d_parms [check1.c]",
         "Parameters are not correctly set");
}


int main(int argc,char *argv[])
{
   int my_rank,np,t,in,n,b,c,id;
   dft_parms_t *dp;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   MPI_Comm_size(MPI_COMM_WORLD,&np);

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);

      printf("\n");
      printf("DFT parameter setting\n");
      printf("---------------------\n\n");

      printf("Number of processes = %d\n\n",np);
      fflush(flog);
   }

   geometry();

   for (t=0;t<(int)(DFT_TYPES);t++)
   {
      for (in=0;in<6;in++)
      {
         for (b=0;b<2;b++)
         {
            for (c=0;c<2;c++)
            {
               n=ns[in];
               id=set_dft_parms((dft_type_t)(t),n,b,c);
               dp=dft_parms(id);

               error((t!=(int)((*dp).type))||(n!=(*dp).n)||(b!=(*dp).b)||
                     (c!=(*dp).c),1,"main [check1.c]",
                     "Parameters type,n,b,c not correctly set");

               if (my_rank==0)
               {
                  if ((*dp).type==EXP)
                     printf("Type = EXP\n");
                  else if ((*dp).type==SIN)
                     printf("Type = SIN\n");
                  else if ((*dp).type==COS)
                     printf("Type = COS\n");
                  else
                     error_root(1,1,"main [check1.c]",
                                "Unknown transformation type");

                  printf("n,b,c = %d,%d,%d\n",(*dp).n,(*dp).b,(*dp).c);
                  printf("id = %d\n\n",id);
                  fflush(flog);
               }

               check_r(dp);
               check_w(dp);
            }
         }
      }
   }

   check_dft4d_parms();

   if (my_rank==0)
   {
      printf("No errors discovered\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
