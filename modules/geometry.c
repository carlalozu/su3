#define GEOMETRY_C

#include "global.h"
#include "lattice.h"


static void alloc_ipt(void)
{
   ipt=malloc(VOLUME*sizeof(*ipt));

   error(ipt==NULL,1,"alloc_ipt [geometryv.c]",
         "Unable to allocate index array");
}

static void set_ipt(void)
{
   int y0,y1,y2,y3,lex;

   alloc_ipt();

   int mem=0;
   for (y0=0;y0<L0;y0++){
      for (y1=0;y1<L1;y1++){
         for (y2=0;y2<L2;y2++){
            for (y3=0;y3<L3;y3++){
               lex=y3+y2*L3+y1*L2*L3+y0*L1*L2*L3;
               ipt[lex]=mem;
               mem+=1;
            }
         }
      }
   }
   #pragma omp target enter data map(to : ipt[:VOLUME])
}

static void alloc_tms(void)
{
   tms=malloc(VOLUME*sizeof(*tms));

   error(tms==NULL,1,"alloc_tms [geometry.c]",
         "Unable to allocate time array");
   #pragma omp target enter data map(to : tms[:VOLUME])
}


static void set_tms(void)
{
   int k,ix,iy,x0;

   alloc_tms();

#pragma omp parallel private(k,ix,iy,x0)
   {
      k=omp_get_thread_num();

      for (iy=(k*VOLUME_TRD);iy<((k+1)*VOLUME_TRD);iy++)
      {
         x0=iy/(L1*L2*L3);
         ix=ipt[iy];

         tms[ix]=x0;
      }
   }
   #pragma omp target update to(tms[:VOLUME])
}

#pragma omp declare target
int global_time(int ix)
{
   if ((ix>=0)&&(ix<VOLUME))
      return tms[ix];
      
   else
      return NPROC0*L0;
}
#pragma omp end declare target


void geometry(void)
{
   if (ipt==NULL)
   {
      set_ipt();
      set_iupdn();
      set_tms();
   }
}