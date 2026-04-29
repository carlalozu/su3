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

// SU3 initialization (bodies in su3.c)
#pragma omp declare target
void unit_su3mat(su3_mat_c *su3_mat);
void random_su3mat(su3_mat_c *su3_mat, uint64_t *state);
void unit_su3vec(su3_vec_c *su3_vec);
#pragma omp end declare target

#endif // SU3_H
