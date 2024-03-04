
/*******************************************************************************
*
* File check4.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Test of the program name_size().
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "utils.h"


int main(void)
{
   char nbase[1024],cnfg_dir[1024];

   printf("\n");
   printf("Test of the program name_size()\n");
   printf("-------------------------------\n\n");

   sprintf(nbase,"%s","48x24x24x24-002");
   sprintf(cnfg_dir,"%s/%s/%s_b%.4f_c%d","home",nbase,nbase,5.3,125);

   printf("name = %s\n",cnfg_dir);
   printf("length    = %d\n",(int)(strlen(cnfg_dir)));
   printf("name_size = %d\n\n",
          name_size("%s/%s/%s_b%.4f_c%d","home",nbase,nbase,5.3,125));

   exit(0);
}
