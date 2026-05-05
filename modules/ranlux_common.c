
/*******************************************************************************
*
* File ranlux_common.c
*
* Copyright (C) 2019, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Collection of common functions used by the programs implementing the
* random number generator RANLUX.
*
*   void rlx_alloc_state(int n,rlx_state_t *s)
*     Allocates the state arrays in the structures s[0],..,s[n-1]. The
*     arrays are aligned to a 64 byte boundary and are padded at the end
*     such that their size is a multiple of 64.
*
*   void rlx_free_state(int n,int rlx_state_t *s)
*     Frees the state arrays in the structures s[0],..,s[n-1], assuming
*     they have been allocated by rlx_alloc_state().
*
*   void rlx_init(rlx_state_t *s,int seed,int flag)
*     Initializes the state "s" of RANLUX using a bit generator with the
*     given "seed" (seed>0). Setting "flag" to a non-zero value indicates
*     that the initialization is for the double-precision generator.
*
*   void rlx_get_state(rlx_state_t *s,int *is)
*     Maps the state vectors and the carry bits contained in the state "s"
*     to an array is[0],..,is[99] of integers in the range from 0 to 2^24-1.
*
*   void rlx_set_state(int *is,rlx_state_t *s)
*     Restores the state vectors and the carry bits in the state "s" from
*     an array is[0],..,is[99] of integers, assuming the array was obtained
*     from a previous call of rlx_get_state(). Various checks are performed
*     to ensure that the state "s" is properly restored.
*
*   void rlx_update(rlx_state_t *s)
*     Updates the state "s" of the random number generator (see the notes).
*
*   void rlx_converts(rlx_state_t *s,float *rs)
*     Extracts 96 random single-precision floating-point numbers from the
*     state "s" and assigns them to rs[0],..,rs[95]. Each number is of the
*     form n/2^24, where n is an integer ranging from 0 to 2^24-1. If SSE2
*     instructions are used, rs must be aligned to a 16 byte boundary.
*
*   void rlx_convertd(rlx_state_t *state,double *rd)
*     Extracts 48 random double-precision floating-point numbers from the
*     state "s" and assigns them to rd[0],..,rd[47]. Each number is of the
*     form n/2^48, where n is an integer ranging from 0 to 2^48-1. If SSE2
*     instructions are used, rd must be aligned to a 16 byte boundary.
*
* These programs are used by the functions in the modules ranlux.c, ranlxd.c
* and ranlxs.c. They are not intended to be called from any other program and
* their prototypes in ranlux.h are masked accordingly.
*
* A structure of type rlx_state_t contains the states of 4 copies of the
* RANLUX generator, which are initialized differently and updated in parallel.
* The double-word algorithm is used, where the state of a single generator
* is an array of 12 integers plus a carry bit. The elements of the structure
* are:
*
*   int pr                       Half of the RANLUX p-value.
*   int ir                       Index of the random number to be
*                                updated next (0<=ir<12).
*   uint64_t (*state)[4]        Current state vectors and carry bits.
*
* The second index of the state array labels the 4 copies of the generator.
* After allocation, the state array has length 14, where the last element is
* unused, while the second-to-last is reserved for the carry bits. The data
* type uint64_t is defined in utils.h and is guaranteed to be an integer
* data type of size 8.
*
* The programs rlx_alloc_state(), rlx_init(), rlx_get_state(), rlx_set_state()
* are assumed to be called by the OpenMP master thread, but need not be called
* simultaneously on all MPI processes. All other programs are thread-safe.
*
*******************************************************************************/

#define RANLUX_COMMON_C

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include "utils.h"
#include "random.h"

static const uint64_t base=(uint64_t)(0x1000000000000);
static const uint64_t mask=(uint64_t)(0xffffffffffff);


void rlx_alloc_state(int n,rlx_state_t *s)
{
   int k;
   uint64_t (*state)[4];

   state=amalloc(n*14*sizeof(*state),6);
   error_loc(state==NULL,1,"rlx_alloc_state [ranlux_common.c]",
             "Unable to allocate state array");

   for (k=0;k<n;k++)
   {
      s[k].state=state;
      state+=14;
   }
}


void rlx_free_state(int n,rlx_state_t *s)
{
   if (n>0)
      afree(s[0].state);
}


void rlx_init(rlx_state_t *s,int seed,int flag)
{
   int i,k,l,ix,iy;
   int ibit,jbit,xbit[31];
   uint64_t (*state)[4];

   for (k=0;k<31;k++)
   {
      xbit[k]=seed&0x1;
      seed/=2;
   }

   ibit=0;
   jbit=18;
   state=(*s).state;

   for (i=0;i<4;i++)
   {
      for (k=0;k<24;k++)
      {
         ix=0;

         for (l=0;l<24;l++)
         {
            iy=xbit[ibit];
            ix=2*ix+iy;

            xbit[ibit]=(xbit[ibit]+xbit[jbit])%2;
            ibit=(ibit+1)%31;
            jbit=(jbit+1)%31;
         }

         if (((flag==0)&&((k%4)==i))||((flag!=0)&&((k%4)!=i)))
            ix=16777215-ix;

         if (k&0x1)
            state[k/2][i]+=((uint64_t)(ix)<<24);
         else
            state[k/2][i]=(uint64_t)(ix);
      }
   }

   (*s).state[12][0]=(uint64_t)(0);
   (*s).state[12][1]=(uint64_t)(0);
   (*s).state[12][2]=(uint64_t)(0);
   (*s).state[12][3]=(uint64_t)(0);

   (*s).state[13][0]=(uint64_t)(0);
   (*s).state[13][1]=(uint64_t)(0);
   (*s).state[13][2]=(uint64_t)(0);
   (*s).state[13][3]=(uint64_t)(0);

   (*s).ir=0;
}


void rlx_get_state(rlx_state_t *s,int *is)
{
   int k;
   uint64_t lmask,(*state)[4];

   lmask=(uint64_t)(0xffffff);
   state=(*s).state;

   for (k=0;k<12;k++)
   {
      is[0]=(int)((*state)[0]&lmask);
      is[1]=(int)((*state)[1]&lmask);
      is[2]=(int)((*state)[2]&lmask);
      is[3]=(int)((*state)[3]&lmask);

      is[4]=(int)((*state)[0]>>24);
      is[5]=(int)((*state)[1]>>24);
      is[6]=(int)((*state)[2]>>24);
      is[7]=(int)((*state)[3]>>24);

      is+=8;
      state+=1;
   }

   is[0]=(int)((*state)[0]);
   is[1]=(int)((*state)[1]);
   is[2]=(int)((*state)[2]);
   is[3]=(int)((*state)[3]);
}


void rlx_set_state(int *is,rlx_state_t *s)
{
   int ie,k,lbase;
   uint64_t (*state)[4];

   ie=0;
   lbase=0x1000000;
   state=(*s).state;

   for (k=0;k<96;k++)
      ie|=((is[k]<0)||(is[k]>=lbase));

   ie|=((is[96]<0)||(is[96]>1));
   ie|=((is[97]<0)||(is[97]>1));
   ie|=((is[98]<0)||(is[98]>1));
   ie|=((is[99]<0)||(is[99]>1));

   error_loc(ie,1,"rlx_set_state [ranlux_common.c]",
             "Input numbers are out of range");

   for (k=0;k<12;k++)
   {
      (*state)[0]=(uint64_t)(is[0]);
      (*state)[1]=(uint64_t)(is[1]);
      (*state)[2]=(uint64_t)(is[2]);
      (*state)[3]=(uint64_t)(is[3]);

      (*state)[0]+=((uint64_t)(is[4])<<24);
      (*state)[1]+=((uint64_t)(is[5])<<24);
      (*state)[2]+=((uint64_t)(is[6])<<24);
      (*state)[3]+=((uint64_t)(is[7])<<24);

      is+=8;
      state+=1;
   }

   (*state)[0]=(uint64_t)(is[0]);
   (*state)[1]=(uint64_t)(is[1]);
   (*state)[2]=(uint64_t)(is[2]);
   (*state)[3]=(uint64_t)(is[3]);
}


void rlx_update(rlx_state_t *s)
{
   int pr,ir,k;
   uint64_t d[4],(*pmin)[4],(*pmax)[4],(*pi)[4],(*pj)[4];

   pr=(*s).pr;
   ir=(*s).ir;

   pmin=(*s).state;
   pmax=pmin+12;
   pi=pmin+ir;
   if (ir>=5)
      pj=pi-5;
   else
      pj=pi+7;

   for (k=0;k<pr;k++)
   {
      d[0]=pj[0][0]-pi[0][0]-pmax[0][0];
      d[1]=pj[0][1]-pi[0][1]-pmax[0][1];
      d[2]=pj[0][2]-pi[0][2]-pmax[0][2];
      d[3]=pj[0][3]-pi[0][3]-pmax[0][3];
      pmax[0][0]=(d[0]<0);
      pmax[0][1]=(d[1]<0);
      pmax[0][2]=(d[2]<0);
      pmax[0][3]=(d[3]<0);
      pi[0][0]=(d[0]+base)&mask;
      pi[0][1]=(d[1]+base)&mask;
      pi[0][2]=(d[2]+base)&mask;
      pi[0][3]=(d[3]+base)&mask;

      pj+=1;
      if (pj==pmax)
         pj=pmin;

      pi+=1;
      if (pi==pmax)
         pi=pmin;
   }

   (*s).ir=(ir+pr)%12;
}


void rlx_converts(rlx_state_t *s,float *rs)
{
   int k;
   uint64_t lmask,(*state)[4];
   float onebit;

   onebit=(float)(ldexp(1.0,-24));
   lmask=(uint64_t)(0xffffff);
   state=(*s).state;

   for (k=0;k<12;k++)
   {
      rs[0]=(float)((*state)[0]&lmask)*onebit;
      rs[1]=(float)((*state)[1]&lmask)*onebit;
      rs[2]=(float)((*state)[2]&lmask)*onebit;
      rs[3]=(float)((*state)[3]&lmask)*onebit;

      rs[4]=(float)((*state)[0]>>24)*onebit;
      rs[5]=(float)((*state)[1]>>24)*onebit;
      rs[6]=(float)((*state)[2]>>24)*onebit;
      rs[7]=(float)((*state)[3]>>24)*onebit;

      rs+=8;
      state+=1;
   }
}


void rlx_convertd(rlx_state_t *s,double *rd)
{
   int k;
   uint64_t (*state)[4];
   double onebit;

   onebit=ldexp(1.0,-48);
   state=(*s).state;

   for (k=0;k<12;k++)
   {
      rd[0]=(double)((*state)[0])*onebit;
      rd[1]=(double)((*state)[1])*onebit;
      rd[2]=(double)((*state)[2])*onebit;
      rd[3]=(double)((*state)[3])*onebit;

      rd+=4;
      state+=1;
   }
}

