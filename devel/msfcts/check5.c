
/*******************************************************************************
*
* File check5.c
*
* Copyright (C) 2017, 2018, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Correlation function of E(x).
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "flags.h"
#include "random.h"
#include "su3fcts.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "archive.h"
#include "tcharge.h"
#include "wflow.h"
#include "msfcts.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int my_rank,first,last,step;
static int i3d,tmin,tmax,dmax;
static double **av,**ava,**avs;
static double ***sm,***sma,***sms;
static double **f;
static complex_dble **rf;
static u3_alg_dble **ft;
static iodat_t iodat[1];
static char nbase[NAME_SIZE],cnfg_file[NAME_SIZE];
static char end_file[NAME_SIZE];
static FILE *flog=NULL,*fin=NULL;


static void read_run_parms(void)
{
   int n;
   double zero[3];

   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   read_iodat("Configurations","i",iodat);

   if (my_rank==0)
   {
      read_line("first","%d",&first);
      read_line("last","%d",&last);
      read_line("step","%d",&step);

      error_root((first<1)||(last<first)||(step<1)||((last-first)%step!=0),1,
                 "read_run_parms [check5.c]","Improper configuration range");
   }

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   n=0;
   zero[0]=0.0;
   zero[1]=0.0;
   zero[2]=0.0;
   (void)(set_hmc_parms(1,&n,0,0,NULL,1,1.0));
   (void)(set_bc_parms(3,1.0,1.0,1.0,1.0,zero,zero,zero));
}


static void read_obs_parms(void)
{
   if (my_rank==0)
   {
      find_section("Correlation function");
      read_line("i3d","%d",&i3d);
      read_line("times","%d %d",&tmin,&tmax);
      read_line("dmax","%d",&dmax);
   }

   error_root((i3d<0)||(i3d>1),1,"read_obs_parms [check5.c]",
              "Dimension flag i3d must be 0 or 1");
   error_root((tmin<0)||(tmax<tmin),1,"read_obs_parms [check5.c]",
              "Improper radius range");
   error_root(dmax<0,1,"read_obs_parms [check5.c]",
              "Maximal distance dmax must be non-negative");

   MPI_Bcast(&i3d,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&tmin,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&tmax,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&dmax,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void check_files(void)
{
   error(name_size("%sn%d",nbase,last)>=NAME_SIZE,1,
         "check_files [check5.c]","Configuration base name is too long");
   sprintf(cnfg_file,"%sn%d",nbase,last);
   check_iodat(iodat,"i",0x1,cnfg_file);
}


static void alloc_flds(int nf,int nrf)
{
   int i;
   double *f1,**f2;
   complex_dble *rf1,**rf2;

   f1=malloc(nf*VOLUME*sizeof(*f1));
   f2=malloc(nf*sizeof(*f2));

   rf1=malloc(nrf*VOLUME*sizeof(*rf1));
   rf2=malloc(nrf*sizeof(*rf2));

   error((f1==NULL)||(f2==NULL)||(rf1==NULL)||(rf2==NULL),1,
         "alloc_flds [check1.c]","Unable to allocate fields");

   f=f2;
   rf=rf2;

   for (i=0;i<nf;i++)
   {
      f2[0]=f1;
      f1+=VOLUME;
      f2+=1;
   }

   for (i=0;i<nrf;i++)
   {
      rf2[0]=rf1;
      rf1+=VOLUME;
      rf2+=1;
   }
}


static void alloc_av(int ntm)
{
   int i,tmx;
   double *p1,**p2;

   tmx=tmax-tmin+1;

   p2=malloc(3*ntm*sizeof(*p2));
   p1=malloc(3*ntm*tmx*sizeof(*p1));
   error((p1==NULL)||(p2==NULL),1,"alloc_av [check5.c]",
         "Unable to allocate auxiliary arrays");

   av=p2;
   ava=av+ntm;
   avs=ava+ntm;

   for (i=0;i<(3*ntm*tmx);i++)
      p1[i]=0.0;

   for (i=0;i<(3*ntm);i++)
   {
      p2[i]=p1;
      p1+=tmx;
   }
}


static void alloc_sm(int ntm)
{
   int i,tmx,dmx;
   double *p1,**p2,***p3;

   tmx=tmax-tmin+1;
   dmx=dmax+1;

   p3=malloc(3*ntm*sizeof(*p3));
   p2=malloc(3*ntm*tmx*sizeof(*p2));
   p1=malloc(3*ntm*tmx*dmx*sizeof(*p1));

   error((p1==NULL)||(p2==NULL)||(p3==NULL),1,"alloc_sm [check5.c]",
         "Unable to allocate auxiliary arrays");

   sm=p3;
   sma=sm+ntm;
   sms=sma+ntm;

   for (i=0;i<(3*ntm*tmx*dmx);i++)
      p1[i]=0.0;

   for (i=0;i<(3*ntm*tmx);i++)
   {
      p2[i]=p1;
      p1+=dmx;
   }

   for (i=0;i<(3*ntm);i++)
   {
      p3[i]=p2;
      p2+=tmx;
   }
}


static double prodXX(u3_alg_dble *X)
{
   double p;

   p=(-2.0/3.0)*((*X).c1+(*X).c2+(*X).c3)*((*X).c1+(*X).c2+(*X).c3)+
      2.0*((*X).c1*(*X).c1+(*X).c2*(*X).c2+(*X).c3*(*X).c3)+
      4.0*((*X).c4*(*X).c4+(*X).c5*(*X).c5+(*X).c6*(*X).c6+
           (*X).c7*(*X).c7+(*X).c8*(*X).c8+(*X).c9*(*X).c9);

   return p;
}


static double density(int ix)
{
   double dn;

   dn=prodXX(ft[0]+ix)+prodXX(ft[1]+ix)+prodXX(ft[2]+ix)+
      prodXX(ft[3]+ix)+prodXX(ft[4]+ix)+prodXX(ft[5]+ix);

   return 0.5*dn;
}


static double set_f0(double t)
{
   int k,ofs,vol,ix;
   double r,s;

   ft=ftensor();
   r=t*t;

#pragma omp parallel private(k,ofs,vol,ix)
   {
      k=omp_get_thread_num();

      vol=VOLUME_TRD;
      ofs=k*vol;

      for (ix=ofs;ix<(ofs+vol);ix++)
         f[0][ix]=r*density(ix);
   }

   s=avg_msfld(f[0]);

#pragma omp parallel private(k,ofs,vol,ix)
   {
      k=omp_get_thread_num();

      vol=VOLUME_TRD;
      ofs=k*vol;

      for (ix=ofs;ix<(ofs+vol);ix++)
         f[0][ix]-=s;
   }

   return s;
}


static void acc_sm(int ntm)
{
   int i,j,k;
   double r,rv;

   rv=1.0/((double)(N0*N1)*(double)(N2*N3));

   for (i=0;i<ntm;i++)
   {
      for (j=0;j<=(tmax-tmin);j++)
      {
         ava[i][j]+=av[i][j];
         avs[i][j]+=av[i][j]*av[i][j];

         for (k=0;k<=dmax;k++)
         {
            sm[i][j][k]*=rv;
            r=sm[i][j][k];
            sma[i][j][k]+=r;
            sms[i][j][k]+=r*r;
         }
      }
   }
}


static int check_end(void)
{
   int iend;
   FILE *end;

   if (my_rank==0)
   {
      iend=0;
      end=fopen(end_file,"r");

      if (end!=NULL)
      {
         fclose(end);
         remove(end_file);
         iend=1;
         printf("End flag set, run stopped\n\n");
      }
   }

   MPI_Bcast(&iend,1,MPI_INT,0,MPI_COMM_WORLD);

   return iend;
}


int main(int argc,char *argv[])
{
   int icnfg,ncnfg;
   int n,i,j,k,mu,s[4];
   int ntm,rule;
   double dt,rv,r,eps,*tm;
   double wt1,wt2,wtavg;
   wflow_parms_t wfl;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check5.log","w",stdout);
      fin=freopen("check5.in","r",stdin);

      printf("\n");
      printf("Correlation function of E(x)\n");
      printf("----------------------------\n\n");

      print_lattice_sizes();
   }

   read_run_parms();
   read_wflow_parms("Wilson flow",0x2);
   read_obs_parms();

   if (my_rank==0)
      fclose(fin);

   check_machine();
   sprintf(end_file,"check5.end");

   if (my_rank==0)
   {
      print_bc_parms(0x0);
      print_wflow_parms();

      printf("Correlation function:\n");
      printf("i3d = %d\n",i3d);
      printf("times = %d,..,%d\n",tmin,tmax);
      printf("dmax = %d\n\n",dmax);

      print_iodat("i",iodat);
      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   start_ranlux(0,12345);
   geometry();
   check_files();

   wfl=wflow_parms();
   rule=wfl.rule;
   eps=wfl.eps;
   ntm=wfl.ntm;
   tm=wfl.tm;

   if (rule>1)
      alloc_wfd(1);
   alloc_flds(4,1);
   alloc_av(ntm);
   alloc_sm(ntm);

   wtavg=0.0;
   rv=1.0/((double)(N0*N1)*(double)(N2*N3));
   s[0]=0;
   s[1]=0;
   s[2]=0;
   s[3]=0;

   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      if (my_rank==0)
      {
         printf("Configuration no %d\n\n",icnfg);
         fflush(flog);
      }

      sprintf(cnfg_file,"%sn%d",nbase,icnfg);
      read_flds(iodat,cnfg_file,0x0,0x1);

      for (i=0;i<ntm;i++)
      {
         if (i==0)
         {
            n=(int)(tm[0]/eps);
            dt=tm[0]-(double)(n)*eps;
         }
         else
         {
            n=(int)((tm[i]-tm[i-1])/eps);
            dt=tm[i]-tm[i-1]-(double)(n)*eps;
         }

         if (rule==1)
         {
            if (n>0)
               fwd_euler(n,eps);
            if (dt>0.0)
               fwd_euler(1,dt);
         }
         else if (rule==2)
         {
            if (n>0)
               fwd_rk2(n,eps);
            if (dt>0.0)
               fwd_rk2(1,dt);
         }
         else
         {
            if (n>0)
               fwd_rk3(n,eps);
            if (dt>0.0)
               fwd_rk3(1,dt);
         }

         set_f0(tm[i]);

         for (j=0;j<=(tmax-tmin);j++)
         {
            for (mu=0;mu<4;mu++)
            {
               s[mu]=(tmin+j)/2;
               shift_msfld(s,f[0],rf[0],f[2]);
               s[mu]-=(tmin+j);

               if (mu==0)
               {
                  shift_msfld(s,f[0],rf[0],f[1]);
                  mul_msfld(f[1],f[2]);
               }
               else
               {
                  shift_msfld(s,f[0],rf[0],f[3]);
                  mul_msfld(f[2],f[3]);
                  add_msfld(f[1],f[2]);
               }

               s[mu]=0;
            }

            mulr_msfld(0.25,f[1]);
            av[i][j]=center_msfld(f[1]);
            convolute_msfld(NULL,f[1],f[1],rf[0],rf[0],f[2]);
            mulr_msfld(rv,f[2]);

            if (i3d)
               sphere3d_sum(dmax,f[2],sm[i][j]);
            else
               sphere_sum(dmax,f[2],sm[i][j]);
         }
      }

      acc_sm(ntm);
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wtavg+=(wt2-wt1);

      if (my_rank==0)
      {
         for (i=0;i<ntm;i++)
         {
            n=fdigits(tm[i]);
            printf("t = %.*f\n",IMAX(n,1),tm[i]);

            for (j=0;j<=(tmax-tmin);j++)
            {
               printf(" x0 = %d, cor(x0) = %.4e\n",
                      tmin+j,av[i][j]);

               for (k=0;k<=dmax;k++)
                  printf(" var[%3d]  ",k);
               printf("\n");
               for (k=0;k<=dmax;k++)
                  printf("% .2e  ",sm[i][j][k]);
               printf("\n");
            }

            printf("\n");
         }

         printf("Configuration no %d fully processed in %.2e sec ",
                icnfg,wt2-wt1);
         printf("(average = %.2e sec)\n\n",
                wtavg/(double)((icnfg-first)/step+1));
         fflush(flog);
      }

      if (check_end())
         break;
   }

   if (my_rank==0)
   {
      ncnfg=(last-first)/step+1;
      r=1.0/(double)(ncnfg);

      for (i=0;i<ntm;i++)
      {
         for (j=0;j<=(tmax-tmin);j++)
         {
            ava[i][j]*=r;
            avs[i][j]*=r;
            avs[i][j]-=ava[i][j]*ava[i][j];
            avs[i][j]=sqrt(fabs(avs[i][j]));

            for (k=0;k<=dmax;k++)
            {
               sma[i][j][k]*=r;
               sms[i][j][k]*=r;
               sms[i][j][k]-=sma[i][j][k]*sma[i][j][k];
               sms[i][j][k]=sqrt(fabs(sms[i][j][k]));
            }
         }
      }

      printf("\n");
      printf("Test summary\n");
      printf("------------\n\n");

      printf("Processed %d configurations\n\n",ncnfg);

      for (i=0;i<ntm;i++)
      {
         n=fdigits(tm[i]);
         printf("t = %.*f\n",IMAX(n,1),tm[i]);

         for (j=0;j<=(tmax-tmin);j++)
         {
            printf(" x0 = %d, cor(x) = %.4e (%.1e)\n",
                   tmin+j,ava[i][j],avs[i][j]);

            for (k=0;k<=dmax;k++)
               printf(" var[%3d]            ",k);
            printf("\n");
            for (k=0;k<=dmax;k++)
               printf("% .2e (%.1e)  ",sma[i][j][k],sms[i][j][k]);
            printf("\n");
         }

         printf("\n");
      }

      printf("\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
