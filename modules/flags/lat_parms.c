
/*******************************************************************************
*
* File lat_parms.c
*
* Copyright (C) 2009-2013, 2016, 2022 Martin Luescher, Isabel Campos
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Lattice parameters and boundary conditions.
*
*   lat_parms_t set_lat_parms(double beta,double c0,
*                             int nk,double *kappa,int isw,double csw)
*     Sets the basic lattice parameters. The parameters are
*
*       beta           Inverse bare coupling (beta=6/g0^2).
*
*       c0             Coefficient of the plaquette loops in the gauge
*                      action (see doc/gauge_action.pdf).
*
*       nk             Number of hopping parameter values.
*
*       kappa          Array of hopping parameter values.
*
*       isw            Implementation of the Sheikholeslami-Wohlert term
*                      (0: traditional, 1: exponential).
*
*       csw            Coefficient of the Sheikholeslami-Wohlert term.
*
*     The return value is a structure that contains the lattice parameters
*     and the associated bare quark masses m0[0],..,m0[nk-1].
*
*   lat_parms_t lat_parms(void)
*     Returns the current lattice parameters in a structure that contains
*     the above parameters plus the bare quark masses.
*
*   void read_lat_parms(char *section,int rflg)
*     Reads the lattice parameters on MPI process 0 from the specified
*     parameter section on stdin. The tags are
*
*       beta         <double>
*       c0           <double>
*       kappa        <double> [<double>]
*       isw          <int>
*       csw          <double>
*
*     The parameters beta and c0 are read if the bit (rflg&0x1) is set and
*     the parameters kappa, isw and csw if the bit (rflg&0x2) is set.
*
*   void print_lat_parms(int ipr)
*     Prints the lattice parameters to stdout on MPI process 0. The
*     parameters beta and c0 are printed if the flag (ipr&0x1) is set
*     and the remaining parameters if the flag (ipr&0x2) is set.
*
*   void write_lat_parms(FILE *fdat)
*     Writes the global lattices sizes and lattice parameters to the
*     file fdat on MPI process 0.
*
*   void check_lat_parms(FILE *fdat)
*     Compares the global lattice sizes and the lattice parameters with
*     the values stored on the file fdat on MPI process 0, assuming the
*     latter were written to the file by the program write_lat_parms().
*
*   bc_parms_t set_bc_parms(int type,
*                           double cG,double cG_prime,
*                           double cF,double cF_prime,
*                           double *phi,double *phi_prime,
*                           double *theta)
*     Sets the boundary conditions and the associated parameters of the
*     action. The parameters are
*
*       type           Chosen type of boundary condition (0: open, 1: SF,
*                      2: open-SF, 3: periodic).
*
*       cG,cG_prime    Gauge action improvement coefficients at time 0
*                      and T, respectively.
*
*       cF,cF_prime    Fermion action improvement coefficients at time 0
*                      and T, respectively.
*
*       phi[0],        First two angles that define the boundary values of
*       phi[1]         the gauge field at time 0.
*
*       phi_prime[0],  First two angles that define the boundary values of
*       phi_prime[1]   the gauge field at time T.
*
*       theta[0],      Angles specifying the phase-periodic boundary
*       theta[1],      conditions for the quark fields in direction 1,2,3.
*       theta[2]
*
*     The return value is a structure that contains these parameters plus
*     the third angles. In this structure, the improvement coefficients and
*     the angles are stored in the form of arrays cG[2],cF[2] and phi[2][3],
*     where cG[0],cF[0],phi[0][3] and cG[1],cF[1],phi[1][3] are the para-
*     meters at time 0 and T, respectively
*      Parameters that are not required for the specification of the chosen
*     boundary conditions are not read and are set to their default values
*     in the data base (angles to 0, improvement coefficients to 1). In the
*     case of SF boundary conditions (type 1), the program only reads cG,cF
*     and the angles phi,phi_prime and then sets cG_prime=cG,cF_prime=cF.
*     When open-SF boundary conditions are chosen, all parameters except for
*     the angles phi are read.
*
*   bc_parms_t bc_parms(void)
*     Returns a structure that contains the boundary parameters.
*
*   void read_bc_parms(char *section,int rflg)
*     Reads the boundary-condition parameters on MPI process 0 from the
*     specified parameter section on stdin. The tags are
*
*       type         <int>
*       phi          <double> <double>
*       phi'         <double> <double>
*       cG           <double>
*       cG'          <double>
*       cF           <double>
*       cF'          <double>
*       theta        <double> <double> <double>
*
*     The parameters cG and cG' are read if the bit (rflg&0x1) is set
*     and the ones related to the quark fields (cF,cF',theta) if the bit
*     (rflg&0x2) is set. The tag "type" must always be present, while
*     the other tags must only be present if required by the boundary
*     conditions and the value of the flag rflg.
*
*   void print_bc_parms(int ipr)
*     Prints the boundary parameters to stdout on MPI process 0. The
*     improvement coefficients cG,cG' are printed if the flag (ipr&0x1)
*     is set and the parameters referring to the quark fields are printed
*     if the flag (ipr&0x2) is set.
*
*   void write_bc_parms(FILE *fdat)
*     Writes the boundary parameters to the file fdat on MPI process 0.
*
*   void check_bc_parms(FILE *fdat)
*     Compares the currently set boundary parameters with the values stored
*     on the file fdat on MPI process 0, assuming the latter were written to
*     the file by the program write_bc_parms().
*
*   double sea_quark_mass(int im0)
*     Returns the bare sea quark mass m0[im0] stored in the lattice
*     parameter data base or DBL_MAX if the index im0 is out of range.
*
*   int bc_type(void)
*     Returns the type of the chosen boundary conditions (0: open, 1: SF,
*     2: open-SF, 3: periodic).
*
*   sw_parms_t set_sw_parms(double m0)
*     Sets the parameters of the SW term. The adjustable parameter is
*
*       m0             Bare quark mass.
*
*     The return value is a structure that contains the mass m0, the type
*     isw of the SW term and the improvement coefficients csw and cF[2],
*     all except m0 being copied from the list of the lattice and boundary
*     parameters, respectively.
*
*   sw_parms_t sw_parms(void)
*     Returns the parameters currently set for the SW term. The values
*     of the SW type isw and the coefficients csw and cF[2] are copied from
*     the lattice and boundary parameter lists.
*
*   tm_parms_t set_tm_parms(int eoflg)
*     Sets the twisted-mass flag. The parameter is
*
*       eoflg          If the flag is set (eoflg!=0), the twisted-mass term
*                      in the Dirac operator, the SAP preconditioner and the
*                      little Dirac operator is turned off on the odd lattice
*                      sites.
*
*     The return value is a structure that contains the twisted-mass flag.
*
*   tm_parms_t tm_parms(void)
*     Returns a structure containing the twisted-mass flag.
*
* To ensure the consistency of the data base, the parameters must be set
* simultaneously on all processes. The data types lat_parms_t,..,tm_parms_t
* are defined in the file flags.h.
*
* The programs set_lat_parms() and set_bc_parms() may be called at most once.
* Moreover, they may not be called after the geometry arrays are set up. The
* default values of the lattice parameters beta=0.0, c0=1.0, nk=0, isw=0 and
* csw=1.0 are used if set_lat_parms() is not called. The default in the case
* of set_bc_parms() are open boundary conditions with cG=cF=1.0.
*
* See the notes doc/gauge_action.pdf and doc/dirac.pdf for the detailed
* description of the lattice action and the boundary conditions.
*
*******************************************************************************/

#define LAT_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int flg_lat=0,flg_bc=0;
static lat_parms_t lat={0,0,0.0,1.0,0.0,NULL,NULL,1.0};
static bc_parms_t bc={0,{1.0,1.0},{1.0,1.0},{{0.0,0.0,0.0},{0.0,0.0,0.0}},
                      {0.0,0.0,0.0}};
static sw_parms_t sw={0,DBL_MAX,1.0,{1.0,1.0}};
static tm_parms_t tm={0};


lat_parms_t set_lat_parms(double beta,double c0,
                          int nk,double *kappa,int isw,double csw)
{
   int iprms[2],ik,ie;
   double dprms[3],*k;

   error(flg_lat!=0,1,"set_lat_parms [lat_parms.c]",
         "Attempt to reset the lattice parameters");

   error(ipt!=NULL,1,"set_lat_parms [lat_parms.c]",
         "Geometry arrays are already set");

   if (NPROC>1)
   {
      iprms[0]=nk;
      iprms[1]=isw;
      dprms[0]=beta;
      dprms[1]=c0;
      dprms[2]=csw;

      MPI_Bcast(iprms,2,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,3,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((iprms[0]!=nk)||(iprms[1]!=isw)||(dprms[0]!=beta)||
            (dprms[1]!=c0)||(dprms[2]!=csw),1,
            "set_lat_parms [lat_parms.c]","Parameters are not global");
   }

   error_root(nk<0,1,"set_lat_parms [lat_parms.c]",
              "Number of kappa values must be non-negative");

   error_root((isw<0)||(isw>1),1,"set_lat_parms [lat_parms.c]",
              "The SW type isw must be either 0 or 1");

   error_root(c0<=0.0,1,"set_lat_parms [lat_parms.c]",
              "Parameter c0 must be positive");

   if (nk>0)
   {
      k=malloc(2*nk*sizeof(*k));
      error(k==NULL,1,"set_lat_parms [lat_parms.c]",
            "Unable to allocate parameter array");
   }
   else
      k=NULL;

   lat.kappa=k;
   lat.m0=k+nk;

   for (ik=0;ik<nk;ik++)
      lat.kappa[ik]=kappa[ik];

   if ((NPROC>1)&&(nk>0))
   {
      for (ik=0;ik<nk;ik++)
         lat.m0[ik]=lat.kappa[ik];

      MPI_Bcast(lat.m0,nk,MPI_DOUBLE,0,MPI_COMM_WORLD);

      ie=0;

      for (ik=0;ik<nk;ik++)
         ie|=(lat.m0[ik]!=lat.kappa[ik]);

      error(ie!=0,1,"set_lat_parms [lat_parms.c]",
            "Hopping parameters are not global");
   }

   lat.nk=nk;
   lat.isw=isw;
   lat.beta=beta;
   lat.c0=c0;
   lat.c1=0.125*(1.0-c0);
   lat.csw=csw;

   for (ik=0;ik<nk;ik++)
   {
      if (lat.kappa[ik]!=0.0)
         lat.m0[ik]=1.0/(2.0*lat.kappa[ik])-4.0;
      else
         lat.m0[ik]=DBL_MAX;
   }

   flg_lat=1;

   return lat;
}


lat_parms_t lat_parms(void)
{
   return lat;
}


void read_lat_parms(char *section,int rflg)
{
   int my_rank,nk,isw;
   double beta,c0,csw,*kappa;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   beta=0.0;
   c0=1.0;
   nk=0;
   isw=0;
   csw=1.0;

   if (my_rank==0)
   {
      find_section(section);

      if (rflg&0x1)
      {
         read_line("beta","%lf",&beta);
         read_line("c0","%lf",&c0);
      }

      if (rflg&0x2)
      {
         nk=count_tokens("kappa");
         read_line("isw","%d",&isw);
         read_line("csw","%lf",&csw);
      }
   }

   if (NPROC>1)
   {
      MPI_Bcast(&beta,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&c0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&nk,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&isw,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }

   if (nk>0)
   {
      kappa=malloc(nk*sizeof(*kappa));
      error(kappa==NULL,1,"read_lat_parms [lat_parms.c]",
            "Unable to allocate parameter array");
      if (my_rank==0)
         read_dprms("kappa",nk,kappa);
      if (NPROC>1)
         MPI_Bcast(kappa,nk,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
      kappa=NULL;

   (void)(set_lat_parms(beta,c0,nk,kappa,isw,csw));

   if (kappa!=NULL)
      free(kappa);
}


void print_lat_parms(int ipr)
{
   int my_rank,n,ik;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      printf("Lattice parameters:\n");

      if (ipr&0x1)
      {
         n=fdigits(lat.beta);
         printf("beta = %.*f\n",IMAX(n,1),lat.beta);
         n=fdigits(lat.c0);
         printf("c0 = %.*f, ",IMAX(n,1),lat.c0);
         n=fdigits(lat.c1);
         printf("c1 = %.*f\n",IMAX(n,1),lat.c1);
      }

      if (ipr&0x2)
      {
         for (ik=0;ik<lat.nk;ik++)
         {
            n=fdigits(lat.kappa[ik]);

            if (lat.nk>=11)
               printf("kappa[%2d] = %.*f\n",ik,IMAX(n,6),lat.kappa[ik]);
            else
               printf("kappa[%1d] = %.*f\n",ik,IMAX(n,6),lat.kappa[ik]);
         }

         n=fdigits(lat.csw);
         printf("isw = %d, csw = %.*f\n",lat.isw,IMAX(n,1),lat.csw);
      }

      printf("\n");
   }
}


void write_lat_parms(FILE *fdat)
{
   int my_rank,endian;
   int iw,ik;
   stdint_t istd[6];
   double dstd[4];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if (my_rank==0)
   {
      istd[0]=(stdint_t)(N0);
      istd[1]=(stdint_t)(N1);
      istd[2]=(stdint_t)(N2);
      istd[3]=(stdint_t)(N3);
      istd[4]=(stdint_t)(lat.nk);
      istd[5]=(stdint_t)(lat.isw);

      dstd[0]=lat.beta;
      dstd[1]=lat.c0;
      dstd[2]=lat.c1;
      dstd[3]=lat.csw;

      if (endian==BIG_ENDIAN)
      {
         bswap_int(6,istd);
         bswap_double(4,dstd);
      }

      iw=fwrite(istd,sizeof(stdint_t),6,fdat);
      iw+=fwrite(dstd,sizeof(double),4,fdat);

      for (ik=0;ik<lat.nk;ik++)
      {
         dstd[0]=lat.kappa[ik];
         dstd[1]=lat.m0[ik];

         if (endian==BIG_ENDIAN)
            bswap_double(2,dstd);

         iw+=fwrite(dstd,sizeof(double),2,fdat);
      }

      error_root(iw!=(10+2*lat.nk),1,"write_lat_parms [lat_parms.c]",
                 "Incorrect write count");
   }
}


void check_lat_parms(FILE *fdat)
{
   int my_rank,endian;
   int ir,ik,ie;
   stdint_t istd[6];
   double dstd[4];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if (my_rank==0)
   {
      ir=fread(istd,sizeof(stdint_t),6,fdat);
      ir+=fread(dstd,sizeof(double),4,fdat);

      if (endian==BIG_ENDIAN)
      {
         bswap_int(6,istd);
         bswap_double(4,dstd);
      }

      ie=0;
      ie|=(istd[0]!=(stdint_t)(N0));
      ie|=(istd[1]!=(stdint_t)(N1));
      ie|=(istd[2]!=(stdint_t)(N2));
      ie|=(istd[3]!=(stdint_t)(N3));
      ie|=(istd[4]!=(stdint_t)(lat.nk));
      ie|=(istd[5]!=(stdint_t)(lat.isw));

      ie|=(dstd[0]!=lat.beta);
      ie|=(dstd[1]!=lat.c0);
      ie|=(dstd[2]!=lat.c1);
      ie|=(dstd[3]!=lat.csw);

      for (ik=0;ik<lat.nk;ik++)
      {
         ir+=fread(dstd,sizeof(double),2,fdat);

         if (endian==BIG_ENDIAN)
            bswap_double(2,dstd);

         ie|=(dstd[0]!=lat.kappa[ik]);
         ie|=(dstd[1]!=lat.m0[ik]);
      }

      error_root(ir!=(10+2*lat.nk),1,"check_lat_parms [lat_parms.c]",
                 "Incorrect read count");

      error_root(ie!=0,1,"check_lat_parms [lat_parms.c]",
                 "Parameters do not match");
   }
}


bc_parms_t set_bc_parms(int type,
                        double cG,double cG_prime,
                        double cF,double cF_prime,
                        double *phi,double *phi_prime,
                        double *theta)
{
   int iprms[1],ie,i;
   double dprms[9];

   error(flg_bc!=0,1,"set_bc_parms [lat_parms.c]",
         "Attempt to reset the boundary conditions");

   error(ipt!=NULL,1,"set_bc_parms [lat_parms.c]",
         "Geometry arrays are already set");

   if (NPROC>1)
   {
      iprms[0]=type;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=type,1,"set_bc_parms [lat_parms.c]",
            "Parameters are not global");

      for (i=0;i<5;i++)
         dprms[i]=0.0;

      if ((type>=0)&&(type<3))
      {
         dprms[0]=cG;
         dprms[1]=cF;

         if (type==1)
         {
            dprms[2]=phi[0];
            dprms[3]=phi[1];
            dprms[4]=phi_prime[0];
            dprms[5]=phi_prime[1];
         }
         else if (type==2)
         {
            dprms[2]=cG_prime;
            dprms[3]=cF_prime;
            dprms[4]=phi_prime[0];
            dprms[5]=phi_prime[1];
         }
      }

      dprms[6]=theta[0];
      dprms[7]=theta[1];
      dprms[8]=theta[2];

      MPI_Bcast(dprms,9,MPI_DOUBLE,0,MPI_COMM_WORLD);

      ie=((dprms[6]!=theta[0])||(dprms[7]!=theta[1])||(dprms[8]!=theta[2]));

      if ((type>=0)&&(type<3))
      {
         ie|=((dprms[0]!=cG)||(dprms[1]!=cF));

         if (type==1)
         {
            ie|=((dprms[2]!=phi[0])||(dprms[3]!=phi[1]));
            ie|=((dprms[4]!=phi_prime[0])||(dprms[5]!=phi_prime[1]));
         }
         else if (type==2)
         {
            ie|=((dprms[2]!=cG_prime)||(dprms[3]!=cF_prime));
            ie|=((dprms[4]!=phi_prime[0])||(dprms[5]!=phi_prime[1]));
         }
      }

      error(ie!=0,1,"set_bc_parms [lat_parms.c]","Parameters are not global");
   }

   error_root((type<0)||(type>3),1,"set_bc_parms [lat_parms.c]",
              "Unknown type of boundary condition");

   bc.type=type;

   if ((type>=0)&&(type<3))
   {
      bc.cG[0]=cG;
      bc.cF[0]=cF;

      if (type==0)
      {
         bc.cG[1]=cG;
         bc.cF[1]=cF;
      }
      else if (type==1)
      {
         bc.cG[1]=cG;
         bc.cF[1]=cF;

         bc.phi[0][0]=phi[0];
         bc.phi[0][1]=phi[1];
         bc.phi[0][2]=-phi[0]-phi[1];

         bc.phi[1][0]=phi_prime[0];
         bc.phi[1][1]=phi_prime[1];
         bc.phi[1][2]=-phi_prime[0]-phi_prime[1];
      }
      else if (type==2)
      {
         bc.cG[1]=cG_prime;
         bc.cF[1]=cF_prime;

         bc.phi[1][0]=phi_prime[0];
         bc.phi[1][1]=phi_prime[1];
         bc.phi[1][2]=-phi_prime[0]-phi_prime[1];
      }
   }

   bc.theta[0]=theta[0];
   bc.theta[1]=theta[1];
   bc.theta[2]=theta[2];

   flg_bc=1;

   return bc;
}


bc_parms_t bc_parms(void)
{
   return bc;
}


void read_bc_parms(char *section,int rflg)
{
   int my_rank,bcn;
   double cG,cG_prime,cF,cF_prime;
   double phi[2],phi_prime[2],theta[3];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      find_section(section);
      read_line("type","%d",&bcn);

      phi[0]=0.0;
      phi[1]=0.0;
      phi_prime[0]=0.0;
      phi_prime[1]=0.0;
      cG=1.0;
      cG_prime=1.0;
      cF=1.0;
      cF_prime=1.0;
      theta[0]=0.0;
      theta[1]=0.0;
      theta[2]=0.0;

      if (bcn==1)
         read_dprms("phi",2,phi);

      if ((bcn==1)||(bcn==2))
         read_dprms("phi'",2,phi_prime);

      if (rflg&0x1)
      {
         if (bcn!=3)
            read_line("cG","%lf",&cG);

         if (bcn==2)
            read_line("cG'","%lf",&cG_prime);
      }

      if (rflg&0x2)
      {
         if (bcn!=3)
            read_line("cF","%lf",&cF);

         if (bcn==2)
            read_line("cF'","%lf",&cF_prime);

         read_dprms("theta",3,theta);
      }
   }

   if (NPROC>1)
   {
      MPI_Bcast(&bcn,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(phi,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(phi_prime,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&cG,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&cG_prime,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&cF_prime,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(theta,3,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }

   (void)(set_bc_parms(bcn,cG,cG_prime,cF,cF_prime,phi,phi_prime,theta));
}


void print_bc_parms(int ipr)
{
   int my_rank,n[3];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      if (bc.type==0)
      {
         printf("Open boundary conditions\n");

         if (ipr&0x1)
         {
            n[0]=fdigits(bc.cG[0]);
            printf("cG = %.*f\n",IMAX(n[0],1),bc.cG[0]);
         }

         if (ipr&0x2)
         {
            n[0]=fdigits(bc.cF[0]);
            printf("cF = %.*f\n",IMAX(n[0],1),bc.cF[0]);
         }
      }
      else if (bc.type==1)
      {
         printf("SF boundary conditions\n");

         if (ipr&0x1)
         {
            n[0]=fdigits(bc.cG[0]);
            printf("cG = %.*f\n",IMAX(n[0],1),bc.cG[0]);
         }

         if (ipr&0x2)
         {
            n[0]=fdigits(bc.cF[0]);
            printf("cF = %.*f\n",IMAX(n[0],1),bc.cF[0]);
         }

         n[0]=fdigits(bc.phi[0][0]);
         n[1]=fdigits(bc.phi[0][1]);
         n[2]=fdigits(bc.phi[0][2]);
         printf("phi = %.*f,%.*f,%.*f\n",IMAX(n[0],1),bc.phi[0][0],
                IMAX(n[1],1),bc.phi[0][1],IMAX(n[2],1),bc.phi[0][2]);

         n[0]=fdigits(bc.phi[1][0]);
         n[1]=fdigits(bc.phi[1][1]);
         n[2]=fdigits(bc.phi[1][2]);
         printf("phi' = %.*f,%.*f,%.*f\n",IMAX(n[0],1),bc.phi[1][0],
                IMAX(n[1],1),bc.phi[1][1],IMAX(n[2],1),bc.phi[1][2]);
      }
      else if (bc.type==2)
      {
         printf("Open-SF boundary conditions\n");

         if (ipr&0x1)
         {
            n[0]=fdigits(bc.cG[0]);
            printf("cG = %.*f\n",IMAX(n[0],1),bc.cG[0]);
         }

         if (ipr&0x2)
         {
            n[0]=fdigits(bc.cF[0]);
            printf("cF = %.*f\n",IMAX(n[0],1),bc.cF[0]);
         }

         if (ipr&0x1)
         {
            n[1]=fdigits(bc.cG[1]);
            printf("cG' = %.*f\n",IMAX(n[1],1),bc.cG[1]);
         }

         if (ipr&0x2)
         {
            n[1]=fdigits(bc.cF[1]);
            printf("cF' = %.*f\n",IMAX(n[1],1),bc.cF[1]);
         }

         n[0]=fdigits(bc.phi[1][0]);
         n[1]=fdigits(bc.phi[1][1]);
         n[2]=fdigits(bc.phi[1][2]);
         printf("phi' = %.*f,%.*f,%.*f\n",IMAX(n[0],1),bc.phi[1][0],
                IMAX(n[1],1),bc.phi[1][1],IMAX(n[2],1),bc.phi[1][2]);
      }
      else
         printf("Periodic boundary conditions\n");

      if (ipr&0x2)
      {
         n[0]=fdigits(bc.theta[0]);
         n[1]=fdigits(bc.theta[1]);
         n[2]=fdigits(bc.theta[2]);
         printf("theta = %.*f,%.*f,%.*f\n",IMAX(n[0],1),bc.theta[0],
                IMAX(n[1],1),bc.theta[1],IMAX(n[2],1),bc.theta[2]);
      }

      printf("\n");
   }
}


void write_bc_parms(FILE *fdat)
{
   int my_rank,endian,iw;
   stdint_t istd[1];
   double dstd[13];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if (my_rank==0)
   {
      istd[0]=(stdint_t)(bc.type);

      dstd[0]=bc.cG[0];
      dstd[1]=bc.cG[1];
      dstd[2]=bc.cF[0];
      dstd[3]=bc.cF[1];
      dstd[4]=bc.phi[0][0];
      dstd[5]=bc.phi[0][1];
      dstd[6]=bc.phi[0][2];
      dstd[7]=bc.phi[1][0];
      dstd[8]=bc.phi[1][1];
      dstd[9]=bc.phi[1][2];
      dstd[10]=bc.theta[0];
      dstd[11]=bc.theta[1];
      dstd[12]=bc.theta[2];

      if (endian==BIG_ENDIAN)
      {
         bswap_int(1,istd);
         bswap_double(13,dstd);
      }

      iw=fwrite(istd,sizeof(stdint_t),1,fdat);
      iw+=fwrite(dstd,sizeof(double),13,fdat);

      error_root(iw!=14,1,"write_bc_parms [bc_parms.c]",
                 "Incorrect write count");
   }
}


void check_bc_parms(FILE *fdat)
{
   int my_rank,endian,ir,ie;
   stdint_t istd[1];
   double dstd[13];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();

   if (my_rank==0)
   {
      ir=fread(istd,sizeof(stdint_t),1,fdat);
      ir+=fread(dstd,sizeof(double),13,fdat);

      if (endian==BIG_ENDIAN)
      {
         bswap_int(1,istd);
         bswap_double(13,dstd);
      }

      ie=0;
      ie|=(istd[0]!=(stdint_t)(bc.type));

      ie|=(dstd[0]!=bc.cG[0]);
      ie|=(dstd[1]!=bc.cG[1]);
      ie|=(dstd[2]!=bc.cF[0]);
      ie|=(dstd[3]!=bc.cF[1]);
      ie|=(dstd[4]!=bc.phi[0][0]);
      ie|=(dstd[5]!=bc.phi[0][1]);
      ie|=(dstd[6]!=bc.phi[0][2]);
      ie|=(dstd[7]!=bc.phi[1][0]);
      ie|=(dstd[8]!=bc.phi[1][1]);
      ie|=(dstd[9]!=bc.phi[1][2]);
      ie|=(dstd[10]!=bc.theta[0]);
      ie|=(dstd[11]!=bc.theta[1]);
      ie|=(dstd[12]!=bc.theta[2]);

      error_root(ir!=14,1,"check_bc_parms [bc_parms.c]",
                 "Incorrect read count");

      error_root(ie!=0,1,"check_bc_parms [bc_parms.c]",
                 "Parameters do not match");
   }
}


double sea_quark_mass(int im0)
{
   if ((im0>=0)&&(im0<lat.nk))
      return lat.m0[im0];
   else
      return DBL_MAX;
}


int bc_type(void)
{
   return bc.type;
}


sw_parms_t set_sw_parms(double m0)
{
   double dprms[1];

   if (NPROC>1)
   {
      dprms[0]=m0;

      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error(dprms[0]!=m0,1,
            "set_sw_parms [lat_parms.c]","Parameter is not global");
   }

   if (m0!=sw.m0)
   {
      set_flags(ERASED_SW);
      set_flags(ERASED_SWD);
      set_grid_flags(SAP_BLOCKS,ERASED_SW);
      set_flags(ERASED_AWHAT);
   }

   sw.m0=m0;
   sw.isw=lat.isw;
   sw.csw=lat.csw;
   sw.cF[0]=bc.cF[0];
   sw.cF[1]=bc.cF[1];

   return sw;
}


sw_parms_t sw_parms(void)
{
   sw.isw=lat.isw;
   sw.csw=lat.csw;
   sw.cF[0]=bc.cF[0];
   sw.cF[1]=bc.cF[1];

   return sw;
}


tm_parms_t set_tm_parms(int eoflg)
{
   int iprms[1];

   if (NPROC>1)
   {
      iprms[0]=eoflg;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=eoflg,1,
            "set_tm_parms [lat_parms.c]","Parameter is not global");
   }

   if (eoflg!=tm.eoflg)
      set_flags(ERASED_AWHAT);

   tm.eoflg=eoflg;

   return tm;
}


tm_parms_t tm_parms(void)
{
   return tm;
}
