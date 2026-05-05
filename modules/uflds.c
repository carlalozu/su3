#define UFLDS_C

#include "lattice.h"
#include "utils.h"
#include "su3.h"
#include "uflds.h"
#include "global.h"

static su3_dble *udb=NULL;

static void alloc_ud(void)
{
   int bc;
   size_t n;

   error_root(sizeof(su3_dble)!=(18*sizeof(double)),1,"alloc_ud [uflds.c]",
              "The su3_dble structures are not properly packed");

   error(ipt==NULL,1,"alloc_ud [uflds.c]","Geometry arrays are not set");

   bc=bc_type();
   n=4*VOLUME;

   udb=amalloc(n*sizeof(*udb),ALIGN);
   error(udb==NULL,1,"alloc_ud [uflds.c]",
         "Unable to allocate memory space for the gauge field");

   set_bc();
   #pragma omp target enter data map(to: udb[:n])
}


su3_dble *udfld(void)
{
   if (udb==NULL)
      alloc_ud();

   return udb;
}


void random_ud(void)
{
   int bc;
   int k,t,ifc,mu,ix;
   su3_dble *ub;

   ub=udfld();
   bc=bc_type();

#pragma omp parallel private(k,ix,t,ifc,mu)
   {
      k=omp_get_thread_num();
      for (ix=k*VOLUME_TRD;ix<(k+1)*VOLUME_TRD;ix++)
      {
         t=global_time(ix);

         if ((t==(N0-1))&&(bc!=0))
         {
            ifc=offset(ix,0);
            random_su3_dble(ub+ifc);
         }

         for (mu=1;mu<4;mu++)
         {
            ifc=offset(ix,mu);
            random_su3_dble(ub+ifc);
         }
      }
   }

   #pragma omp target update to(udb[:4*VOLUME])
}