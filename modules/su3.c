
/*******************************************************************************
 *
 * File su3.c
 *
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef SU3_C
#define SU3_C

#include "su3.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "random.h"
#include "lattice.h"
#include <math.h>



static void random_su3_vector_dble(su3_vector_dble *v)
{
   double norm,fact,*r;
   vector_dble_t *w;

   w=(vector_dble_t*)(v);
   r=(*w).r;
   norm=0.0;

   while (norm<=0.1)
   {
      gauss_dble(r,6);
      norm=r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+
           r[3]*r[3]+r[4]*r[4]+r[5]*r[5];
   }

   fact=1.0/sqrt(norm);

   r[0]*=fact;
   r[1]*=fact;
   r[2]*=fact;
   r[3]*=fact;
   r[4]*=fact;
   r[5]*=fact;
}


void random_su3_dble(su3_dble *u)
{
   double norm,fact;
   su3_vector_dble *v;
   matrix_dble_t *m;

   m=(matrix_dble_t*)(u);
   v=(*m).v;

   random_su3_vector_dble(v);
   norm=0.0;

   while (norm<=0.1)
   {
      random_su3_vector_dble(v+1);
      _vector_cross_prod(v[2],v[0],v[1]);
      norm=_vector_prod_re(v[2],v[2]);
   }

   fact=1.0/sqrt(norm);

   v[2].c1.re*=fact;
   v[2].c1.im*=fact;
   v[2].c2.re*=fact;
   v[2].c2.im*=fact;
   v[2].c3.re*=fact;
   v[2].c3.im*=fact;

   _vector_cross_prod(v[1],v[2],v[0]);
}

#endif // SU3_C
