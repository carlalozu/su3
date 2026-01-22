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

typedef struct
{
    double re, im;
} complex;

typedef struct
{
    complex c1, c2, c3;
} su3_vec;

typedef struct
{
    complex c11, c12, c13, c21, c22, c23, c31, c32, c33;
} su3_mat;

// SU3 initialization
void unit_su3mat(su3_mat *su3_mat);
void random_su3mat(su3_mat *su3_mat);
void unit_su3vec(su3_vec *vec);

// Algebra
complex add(complex a, complex b);
void vec_add(su3_vec *res, su3_vec *a, su3_vec *b);
void su3matxsu3vec(su3_vec *res, su3_mat *mat, su3_vec *vec);
void su3matxsu3mat(su3_mat *res, su3_mat *u,su3_mat *v);
complex su3_trace(su3_mat *u);

#endif // SU3_H
