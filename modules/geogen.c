#define GEOGEN_C

#include "global.h"
#include "lattice.h"

static void alloc_iupdn(void)
{
   iup=malloc(VOLUME*sizeof(*iup));
   iupT=malloc(4*sizeof(*iupT));
   idn=malloc(VOLUME*sizeof(*idn));

   error((iup==NULL)||(idn==NULL)||(iupT==NULL),1,"alloc_iupdn [geogen.c]",
         "Unable to allocate index arrays");
}

static int index(int x0,int x1,int x2,int x3)
{
   int y0,y1,y2,y3;

   y0=safe_mod(x0,L0);
   y1=safe_mod(x1,L1);
   y2=safe_mod(x2,L2);
   y3=safe_mod(x3,L3);

   return ipt[y3+y2*L3+y1*L2*L3+y0*L1*L2*L3];
}

static int cart_index(int x0,int x1,int x2,int x3)
{
   int y0,y1,y2,y3;

   y0=safe_mod(x0,L0);
   y1=safe_mod(x1,L1);
   y2=safe_mod(x2,L2);
   y3=safe_mod(x3,L3);

   return y3+y2*L3+y1*L2*L3+y0*L1*L2*L3;
}


void set_iupdn(void)
{
   int k,x0,x1,x2,x3;
   int ix,iy,iz;

   if (iup==NULL)
   {
      error(ipt==NULL,1,"set_iupdn [geogen.c]",
            "Index array ipt[VOLUME] is not allocated");
      alloc_iupdn();

#pragma omp parallel private(k,x0,x1,x2,x3,ix,iy,iz)
      {
         k=omp_get_thread_num();

         for (iy=k*VOLUME_TRD;iy<(k+1)*VOLUME_TRD;iy++)
         {
            iz=iy;
            x3=iz%L3;
            iz/=L3;
            x2=iz%L2;
            iz/=L2;
            x1=iz%L1;
            iz/=L1;
            x0=iz;

            ix=index(x0,x1,x2,x3);

            iup[ix][0]=index(x0+1,x1,x2,x3);
            iupT[0][ix]=index(x0+1,x1,x2,x3);
            idn[ix][0]=index(x0-1,x1,x2,x3);

            iup[ix][1]=index(x0,x1+1,x2,x3);
            iupT[1][ix]=index(x0,x1+1,x2,x3);
            idn[ix][1]=index(x0,x1-1,x2,x3);

            iup[ix][2]=index(x0,x1,x2+1,x3);
            iupT[2][ix]=index(x0,x1,x2+1,x3);
            idn[ix][2]=index(x0,x1,x2-1,x3);

            iup[ix][3]=index(x0,x1,x2,x3+1);
            iupT[3][ix]=index(x0,x1,x2,x3+1);
            idn[ix][3]=index(x0,x1,x2,x3-1);

            if ((x0==(L0-1))&&(NPROC0>1)){
               iup[ix][0]=VOLUME;
               iupT[0][ix]=VOLUME;}
            if ((x0==0)&&(NPROC0>1))
               idn[ix][0]=VOLUME;

            if ((x1==(L1-1))&&(NPROC1>1)){
               iup[ix][1]=VOLUME;
               iupT[1][ix]=VOLUME;}
            if ((x1==0)&&(NPROC1>1))
               idn[ix][1]=VOLUME;

            if ((x2==(L2-1))&&(NPROC2>1)){
               iup[ix][2]=VOLUME;
               iupT[2][ix]=VOLUME;}
            if ((x2==0)&&(NPROC2>1))
               idn[ix][2]=VOLUME;

            if ((x3==(L3-1))&&(NPROC3>1)){
               iup[ix][3]=VOLUME;
               iupT[3][ix]=VOLUME;}
            if ((x3==0)&&(NPROC3>1))
               idn[ix][3]=VOLUME;
         }
      }
      #pragma omp target enter data map(to: iup[:VOLUME], iupT[:4])
   }
}
