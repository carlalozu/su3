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

#pragma omp declare target
// SU3 initialization
void unit_su3mat(su3_mat *su3_mat);
void unit_su3vec(su3_vec *su3_vec);
void random_su3mat(su3_mat *su3_mat);
// Algebra
complex add(const complex a, const complex b);
void vec_add(su3_vec *res, const su3_vec *u, const su3_vec *v);
void su3matxsu3vec(su3_vec *res, const su3_mat *u, const su3_vec *v);
void su3matxsu3mat(su3_mat *res, const su3_mat *u, const su3_mat *v);
complex su3_trace(const su3_mat *u);
#pragma omp end declare target
#endif // SU3_H
