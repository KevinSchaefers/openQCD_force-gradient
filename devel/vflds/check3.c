
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2007-2013, 2016, 2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the communication program cpv_int_bnd().
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "linalg.h"
#include "dfl.h"
#include "vflds.h"
#include "global.h"

static int bs[4],Ns,nv,nvec;
static int nb,nbb,*nbbe,*nbbo,*obbe,*obbo;
static int (*inn)[8],*ipp;


static void set_field(complex *v)
{
   int n[4],no[4],c[4];
   int i0,i1,i2,i3,ibe,ibo;

   n[0]=L0/bs[0];
   n[1]=L1/bs[1];
   n[2]=L2/bs[2];
   n[3]=L3/bs[3];

   no[0]=cpr[0]*n[0];
   no[1]=cpr[1]*n[1];
   no[2]=cpr[2]*n[2];
   no[3]=cpr[3]*n[3];

   set_v2zero(nv,0,v);
   ibe=0;
   ibo=(n[0]*n[1]*n[2]*n[3])/2;

   for (i0=0;i0<n[0];i0++)
   {
      for (i1=0;i1<n[1];i1++)
      {
         for (i2=0;i2<n[2];i2++)
         {
            for (i3=0;i3<n[3];i3++)
            {
               c[0]=no[0]+i0;
               c[1]=no[1]+i1;
               c[2]=no[2]+i2;
               c[3]=no[3]+i3;

               if (((c[0]+c[1]+c[2]+c[3])&0x1)==0x0)
               {
                  v[ibe*Ns  ].re=(double)(c[0]);
                  v[ibe*Ns+1].re=(double)(c[1]);
                  v[ibe*Ns+2].re=(double)(c[2]);
                  v[ibe*Ns+3].re=(double)(c[3]);
                  ibe+=1;
               }
               else
               {
                  v[ibo*Ns  ].re=(double)(c[0]);
                  v[ibo*Ns+1].re=(double)(c[1]);
                  v[ibo*Ns+2].re=(double)(c[2]);
                  v[ibo*Ns+3].re=(double)(c[3]);
                  ibo+=1;
               }
            }
         }
      }
   }
}


static int chk_ext_bnd(int ieo,complex *v)
{
   int np[4],bc,ofs,vol;
   int ifc,ib,ibb,mu,i,ie;
   float c[4],n[4];

   bc=bc_type();

   np[0]=NPROC0;
   np[1]=NPROC1;
   np[2]=NPROC2;
   np[3]=NPROC3;

   n[0]=(double)((NPROC0*L0)/bs[0]);
   n[1]=(double)((NPROC1*L1)/bs[1]);
   n[2]=(double)((NPROC2*L2)/bs[2]);
   n[3]=(double)((NPROC3*L3)/bs[3]);
   ie=0;

   for (ifc=0;ifc<8;ifc++)
   {
      if (ieo)
      {
         ofs=obbo[ifc]-(nbb/2);
         vol=nbbo[ifc];
      }
      else
      {
         ofs=obbe[ifc];
         vol=nbbe[ifc];
      }

      if ((ifc>1)||
          ((ifc==0)&&(cpr[0]!=0))||
          ((ifc==1)&&(cpr[0]!=(NPROC0-1)))||
          (bc==3))
      {
         for (ibb=ofs;ibb<(ofs+vol);ibb++)
         {
            ib=ipp[ibb+ieo*(nbb/2)];

            for (mu=0;mu<4;mu++)
            {
               c[mu]=v[nv+ibb*Ns+mu].re-v[ib*Ns+mu].re;

               if (mu==(ifc/2))
               {
                  if ((ifc&0x1)==0x0)
                  {
                     c[mu]+=1.0;

                     if (cpr[mu]==0)
                        c[mu]-=n[mu];
                  }
                  else
                  {
                     c[mu]-=1.0;

                     if (cpr[mu]==(np[mu]-1))
                        c[mu]+=n[mu];
                  }
               }
            }

            if ((c[0]!=0.0)||(c[1]!=0.0)||(c[2]!=0.0)||(c[3]!=0.0))
               ie=1;
         }
      }
      else
      {
         for (ibb=ofs;ibb<(ofs+vol);ibb++)
         {
            for (i=0;i<Ns;i++)
            {
               if ((v[nv+Ns*ibb+i].re!=0.0)||(v[nv+Ns*ibb+i].im!=0.0))
                  ie=2;
            }
         }
      }
   }

   return ie;
}


int main(int argc,char *argv[])
{
   int my_rank,bc,i,ie;
   float nrm;
   double phi[2],phi_prime[2],theta[3];
   complex **wv,z;
   dfl_grid_t *dgr;
   FILE *fin=NULL,*flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      fin=freopen("check1.in","r",stdin);

      printf("\n");
      printf("Check of the communication program cpv_int_bnd()\n");
      printf("------------------------------------------------\n\n");

      print_lattice_sizes();

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      fclose(fin);

      printf("bs = %d %d %d %d\n\n",bs[0],bs[1],bs[2],bs[3]);

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check3.c]",
                    "Syntax: check3 [-bc <type>]");
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   phi[0]=0.0;
   phi[1]=0.0;
   phi_prime[0]=0.0;
   phi_prime[1]=0.0;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;
   set_bc_parms(bc,1.0,1.0,1.0,1.0,phi,phi_prime,theta);
   print_bc_parms(0x0);

   start_ranlux(0,123456);
   geometry();
   Ns=4;
   set_dfl_parms(bs,Ns);
   set_dfl_geometry();
   dgr=dfl_geometry();
   nb=(*dgr).nb;
   nbb=(*dgr).nbb;
   nbbe=(*dgr).nbbe;
   nbbo=(*dgr).nbbo;
   obbe=(*dgr).obbe;
   obbo=(*dgr).obbo;
   inn=(*dgr).inn;
   ipp=(*dgr).ipp;

   alloc_wv(3);
   wv=reserve_wv(3);

   nv=Ns*nb;
   nvec=Ns*(nb+nbb/2);
   z.re=-1.0f;
   z.im=0.0f;

   for (i=0;i<2;i++)
   {
      random_v(nvec,0,wv[i],1.0f);
      set_field(wv[i]);
      assign_v2v(nv,0,wv[i],wv[i+1]);

      cpv_int_bnd(i,wv[i]);
      mulc_vadd(nv,0,wv[i+1],wv[i],z);
      nrm=vnorm_square(nv,1,wv[i+1]);
      error_root(nrm!=0.0f,1,"main [check3.c]",
                 "cpv_int_bnd() modifies the input field on the local grid");

      ie=chk_ext_bnd(i,wv[i]);
      error(ie==1,1,"main [check3.c]",
            "Boundary values are incorrectly mapped by cpv_int_bnd()");
      error(ie==2,1,"main [check3.c]",
            "Boundary values are not set to zero where they should");
   }

   if (my_rank==0)
   {
      printf("No errors detected\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
