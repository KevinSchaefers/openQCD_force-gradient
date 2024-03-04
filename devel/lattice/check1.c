
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2005, 2011, 2013, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Consistency checks on the global index arrays cpr,..,map.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "lattice.h"
#include "global.h"

#define NPROC_BLK (NPROC0_BLK*NPROC1_BLK*NPROC2_BLK*NPROC3_BLK)
#define NTHREAD0 (L0/L0_TRD)
#define NTHREAD1 (L1/L1_TRD)
#define NTHREAD2 (L2/L2_TRD)
#define NTHREAD3 (L3/L3_TRD)

static int ip_test[NPROC];
static int ix_test[VOLUME];
static int ia[2][9];


static void set_ia(void)
{
   int ifc;

   ia[0][0]=0;
   ia[0][1]=ia[0][0]+(FACE0/2);
   ia[0][2]=ia[0][1]+(FACE0/2);
   ia[0][3]=ia[0][2]+(FACE1/2);
   ia[0][4]=ia[0][3]+(FACE1/2);
   ia[0][5]=ia[0][4]+(FACE2/2);
   ia[0][6]=ia[0][5]+(FACE2/2);
   ia[0][7]=ia[0][6]+(FACE3/2);
   ia[0][8]=ia[0][7]+(FACE3/2);

   for (ifc=0;ifc<9;ifc++)
      ia[1][ifc]=ia[0][ifc]+(BNDRY/2);
}


int main(int argc,char *argv[])
{
   int my_rank,itest;
   int in,ir,n[4];
   int k,mu,ix,x0,x1,x2,x3,y0,y1,y2,y3,ofs[2][2];
   int iy0,iy1,iy2,iy3,iz0,iz1,iz2,iz3;
   FILE *flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);

      printf("\n");
      printf("Consistency checks on the global index arrays cpr,...,map\n");
      printf("---------------------------------------------------------\n\n");

      print_lattice_sizes();
   }

   geometry();
   set_ia();

   error(my_rank!=ipr_global(cpr),1,
         "main [check1.c]","Processor coordinates are incorrect");

   if (my_rank==0)
   {
      for (in=0;in<NPROC;in++)
      {
         ir=in;
         n[0]=ir%NPROC0;
         ir/=NPROC0;
         n[1]=ir%NPROC1;
         ir/=NPROC1;
         n[2]=ir%NPROC2;
         ir/=NPROC2;
         n[3]=ir%NPROC3;

         ip_test[in]=ipr_global(n);
      }
   }

   MPI_Bcast(ip_test,NPROC,MPI_INT,0,MPI_COMM_WORLD);

   itest=0;

   for (in=0;in<NPROC;in++)
   {
      ir=in;
      n[0]=ir%NPROC0;
      ir/=NPROC0;
      n[1]=ir%NPROC1;
      ir/=NPROC1;
      n[2]=ir%NPROC2;
      ir/=NPROC2;
      n[3]=ir%NPROC3;

      if (ip_test[in]!=ipr_global(n))
         itest=1;

      n[0]-=(n[0]%NPROC0_BLK);
      n[1]-=(n[1]%NPROC1_BLK);
      n[2]-=(n[2]%NPROC2_BLK);
      n[3]-=(n[3]%NPROC3_BLK);
      ir=ipr_global(n);

      if ((ip_test[in]<ir)||(ip_test[in]>=(ir+NPROC_BLK)))
         itest=2;
   }

   error(itest==1,1,
         "main [check1.c]","ipr_global is process dependent");

   error(itest==2,1,
         "main [check1.c]","Processes are not properly blocked");

   n[0]=cpr[0];
   n[1]=cpr[1];
   n[2]=cpr[2];
   n[3]=cpr[3];

   for (mu=0;mu<4;mu++)
   {
      n[mu]-=1;
      if (npr[2*mu]!=ipr_global(n))
         itest=1;
      n[mu]+=2;
      if (npr[2*mu+1]!=ipr_global(n))
         itest=1;
      n[mu]-=1;
   }

   error(itest==1,1,
         "main [check1.c]","npr is incorrect");

   for (k=0;k<16;k++)
   {
      if (sbvol[k]<16)
         itest=1;

      if ((sbofs[0]!=0)||((k>0)&&(sbofs[k]!=(sbofs[k-1]+sbvol[k-1]))))
         itest=1;
   }

   if ((sbofs[15]+sbvol[15])!=VOLUME_TRD)
      itest=1;

   error(itest==1,1,
         "main [check1.c]","sbofs or sbvol is incorrect");

   for (ix=0;ix<VOLUME;ix++)
      ix_test[ix]=0;

   for (x0=0;x0<L0;x0++)
   {
      for (x1=0;x1<L1;x1++)
      {
         for (x2=0;x2<L2;x2++)
         {
            for (x3=0;x3<L3;x3++)
            {
               ix=ipt[x3+L3*x2+L2*L3*x1+L1*L2*L3*x0];

               if ((ix<0)||(ix>=VOLUME))
                  itest=1;
               else
                  ix_test[ix]+=1;
            }
         }
      }
   }

   error(itest==1,1,
         "main [check1.c]","The index ipt is out of range");

   for (ix=0;ix<VOLUME;ix++)
   {
      if (ix_test[ix]!=1)
         itest=1;
   }

   error(itest==1,1,
         "main [check1.c]","The index ipt is not one-to-one");

   for (x0=0;x0<L0;x0++)
   {
      for (x1=0;x1<L1;x1++)
      {
         for (x2=0;x2<L2;x2++)
         {
            for (x3=0;x3<L3;x3++)
            {
               ix=ipt[x3+L3*x2+L2*L3*x1+L1*L2*L3*x0];
               ir=(x0+x1+x2+x3)%2;

               if (((ir==0)&&(ix>=(VOLUME/2)))||((ir==1)&&(ix<(VOLUME/2))))
                  itest=1;

               ir^=0x1;
               iy0=iup[ix][0];
               iz0=ipt[x3+L3*x2+L2*L3*x1+L1*L2*L3*((x0+1)%L0)];

               if ((x0==(L0-1))&&(NPROC0>1))
               {
                  iy0-=VOLUME;
                  if ((iy0<ia[ir][1])||(iy0>=ia[ir][2]))
                     itest=2;
                  else
                     iy0=map[iy0];
               }

               iy1=iup[ix][1];
               iz1=ipt[x3+L3*x2+L2*L3*((x1+1)%L1)+L1*L2*L3*x0];

               if ((x1==(L1-1))&&(NPROC1>1))
               {
                  iy1-=VOLUME;
                  if ((iy1<ia[ir][3])||(iy1>=ia[ir][4]))
                     itest=2;
                  else
                     iy1=map[iy1];
               }

               iy2=iup[ix][2];
               iz2=ipt[x3+L3*((x2+1)%L2)+L2*L3*x1+L1*L2*L3*x0];

               if ((x2==(L2-1))&&(NPROC2>1))
               {
                  iy2-=VOLUME;
                  if ((iy2<ia[ir][5])||(iy2>=ia[ir][6]))
                     itest=2;
                  else
                     iy2=map[iy2];
               }

               iy3=iup[ix][3];
               iz3=ipt[((x3+1)%L3)+L3*x2+L2*L3*x1+L1*L2*L3*x0];

               if ((x3==(L3-1))&&(NPROC3>1))
               {
                  iy3-=VOLUME;
                  if ((iy3<ia[ir][7])||(iy3>=ia[ir][8]))
                     itest=2;
                  else
                     iy3=map[iy3];
               }

               if ((iy0!=iz0)||(iy1!=iz1)||(iy2!=iz2)||(iy3!=iz3))
                  itest=3;

               iy0=idn[ix][0];
               iz0=ipt[x3+L3*x2+L2*L3*x1+L1*L2*L3*((x0+L0-1)%L0)];

               if ((x0==0)&&(NPROC0>1))
               {
                  iy0-=VOLUME;
                  if ((iy0<ia[ir][0])||(iy0>=ia[ir][1]))
                     itest=4;
                  else
                     iy0=map[iy0];
               }

               iy1=idn[ix][1];
               iz1=ipt[x3+L3*x2+L2*L3*((x1+L1-1)%L1)+L1*L2*L3*x0];

               if ((x1==0)&&(NPROC1>1))
               {
                  iy1-=VOLUME;
                  if ((iy1<ia[ir][2])||(iy1>=ia[ir][3]))
                     itest=4;
                  else
                     iy1=map[iy1];
               }

               iy2=idn[ix][2];
               iz2=ipt[x3+L3*((x2+L2-1)%L2)+L2*L3*x1+L1*L2*L3*x0];

               if ((x2==0)&&(NPROC2>1))
               {
                  iy2-=VOLUME;
                  if ((iy2<ia[ir][4])||(iy2>=ia[ir][5]))
                     itest=4;
                  else
                     iy2=map[iy2];
               }

               iy3=idn[ix][3];
               iz3=ipt[((x3+L3-1)%L3)+L3*x2+L2*L3*x1+L1*L2*L3*x0];

               if ((x3==0)&&(NPROC3>1))
               {
                  iy3-=VOLUME;
                  if ((iy3<ia[ir][6])||(iy3>=ia[ir][7]))
                     itest=4;
                  else
                     iy3=map[iy3];
               }

               if ((iy0!=iz0)||(iy1!=iz1)||(iy2!=iz2)||(iy3!=iz3))
                  itest=5;

               y0=x0/L0_TRD;
               y1=x1/L1_TRD;
               y2=x2/L2_TRD;
               y3=x3/L3_TRD;

               k=y3+y2*NTHREAD3+y1*NTHREAD2*NTHREAD3+
                  y0*NTHREAD1*NTHREAD2*NTHREAD3;

               ir^=0x1;
               ofs[0][0]=k*(VOLUME_TRD/2);
               ofs[0][1]=(k+1)*(VOLUME_TRD/2);
               ofs[1][0]=ofs[0][0]+VOLUME/2;
               ofs[1][1]=ofs[0][1]+VOLUME/2;

               if ((ix<ofs[ir][0])||(ix>=ofs[ir][1]))
                  itest=6;

               y0=x0%L0_TRD;
               y1=x1%L1_TRD;
               y2=x2%L2_TRD;
               y3=x3%L3_TRD;

               n[0]=(y0>=((L0_TRD/2)+((L0_TRD/2)%2)));
               n[1]=(y1>=((L1_TRD/2)+((L1_TRD/2)%2)));
               n[2]=(y2>=((L2_TRD/2)+((L2_TRD/2)%2)));
               n[3]=(y3>=((L3_TRD/2)+((L3_TRD/2)%2)));

               k=n[3]+n[2]*2+n[1]*4+n[0]*8;

               if ((ix<(ofs[ir][0]+sbofs[k]/2))||
                   (ix>=(ofs[ir][0]+sbofs[k]/2+sbvol[k]/2)))
                  itest=7;
            }
         }
      }
   }

   error(itest==1,1,
         "main [check1.c]","The index ipt does not respect eo ordering");
   error(itest==2,1,
         "main [check1.c]","The index iup is out of range at the boundaries");
   error(itest==3,1,
         "main [check1.c]","The index iup (combined with map) is incorrect");
   error(itest==4,1,
         "main [check1.c]","The index idn is out of range at the boundaries");
   error(itest==5,1,
         "main [check1.c]","The index idn (combined with map) is incorrect");
   error(itest==6,1,
         "main [check1.c]","The index ipt violates the thread-block ordering");
   error(itest==7,1,
         "main [check1.c]","The index ipt violates the subblock ordering");

   if (my_rank==0)
   {
      printf("The lattice is correctly mapped by the global arrays\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
