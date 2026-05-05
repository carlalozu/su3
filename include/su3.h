/*******************************************************************************
 *
 * File su3.h
 *
 * Everything is in double precision.
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef SU3_H
#define SU3_H

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "random.h"
#include "global.h"

typedef struct
{
    double re, im;
} complex_dble;

typedef struct
{
    complex_dble c1, c2, c3;
} su3_vec_c;

typedef struct
{
    double c1re, c1im;
    double c2re, c2im;
    double c3re, c3im;
} su3_vec_dble;

typedef struct {
    su3_vec_dble c1, c2, c3;
} su3_mat_dble;

typedef struct
{
    complex_dble c11, c12, c13, c21, c22, c23, c31, c32, c33;
} su3_dble;

typedef struct
{
   double q[2];
} qflt;

typedef struct
{
   complex_dble c1,c2,c3;
} su3_vector_dble;

typedef union
{
   su3_vector_dble v;
   double r[6];
} vector_dble_t;

typedef union
{
   su3_dble u;
   su3_vector_dble v[3];
} matrix_dble_t;

#ifdef __cplusplus
extern "C" {
#endif

extern void random_su3_dble(su3_dble *su3_mat);

#ifdef __cplusplus
}
#endif

#define _vector_prod_re(r,s) \
   (r).c1.re*(s).c1.re+(r).c1.im*(s).c1.im+ \
   (r).c2.re*(s).c2.re+(r).c2.im*(s).c2.im+ \
   (r).c3.re*(s).c3.re+(r).c3.im*(s).c3.im

#define _vector_cross_prod(v,w,z) \
   (v).c1.re= (w).c2.re*(z).c3.re-(w).c2.im*(z).c3.im  \
             -(w).c3.re*(z).c2.re+(w).c3.im*(z).c2.im; \
   (v).c1.im= (w).c3.re*(z).c2.im+(w).c3.im*(z).c2.re  \
             -(w).c2.re*(z).c3.im-(w).c2.im*(z).c3.re; \
   (v).c2.re= (w).c3.re*(z).c1.re-(w).c3.im*(z).c1.im  \
             -(w).c1.re*(z).c3.re+(w).c1.im*(z).c3.im; \
   (v).c2.im= (w).c1.re*(z).c3.im+(w).c1.im*(z).c3.re  \
             -(w).c3.re*(z).c1.im-(w).c3.im*(z).c1.re; \
   (v).c3.re= (w).c1.re*(z).c2.re-(w).c1.im*(z).c2.im  \
             -(w).c2.re*(z).c1.re+(w).c2.im*(z).c1.im; \
   (v).c3.im= (w).c2.re*(z).c1.im+(w).c2.im*(z).c1.re  \
             -(w).c1.re*(z).c2.im-(w).c1.im*(z).c2.re


#endif // SU3_H
