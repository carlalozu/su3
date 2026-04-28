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

#ifdef __CUDACC__
  #define LQCD_DEVICE    __device__
#else
  #define LQCD_DEVICE
#endif

typedef struct
{
    double re, im;
} complex;

typedef struct
{
    complex c1, c2, c3;
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
    complex c11, c12, c13, c21, c22, c23, c31, c32, c33;
} su3_mat_c;

// SU3 initialization
#pragma omp declare target
void unit_su3mat(su3_mat_c *su3_mat);
void random_su3mat(su3_mat_c *su3_mat, uint64_t *state);
void unit_su3vec(su3_vec_c *su3_vec);
#pragma omp end declare target

// Algebra
complex add(const complex a, const complex b);
void vec_add(su3_vec_c *res, const su3_vec_c *u, const su3_vec_c *v);

#pragma omp declare target
complex su3mat_trace(const su3_mat_c *u);
double su3matxsu3mat_retrace(const su3_mat_c *u, const su3_mat_c *v);
LQCD_DEVICE double su3matdxsu3matd_retrace(const su3_mat_dble *u, const su3_mat_dble *v);
void su3matxsu3vec(su3_vec_c *res, const su3_mat_c *u, const su3_vec_c *v);
void su3matdagxsu3vec(su3_vec_c*, const su3_mat_c*, const su3_vec_c*);
void su3matxsu3vec(su3_vec_c *res, const su3_mat_c *u, const su3_vec_c *v);
void su3matxsu3mat(su3_mat_c *res, const su3_mat_c *u, const su3_mat_c *v);
void su3matdagxsu3matdag(su3_mat_c *res, const su3_mat_c *u, const su3_mat_c *v);
#pragma omp end declare target

#endif // SU3_H
