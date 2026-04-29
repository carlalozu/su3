
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

#pragma omp declare target
double local_rand(uint64_t *state) {
    // Standard LCG parameters (e.g., MMIX by Knuth)
    *state = 6364136223846793005ULL * (*state) + 1ULL;
    return (double)(*state >> 33) / 2147483647.0;
}
#pragma omp end declare target

void unit_su3mat(su3_mat_c *su3)
{
    _Static_assert(sizeof(su3_mat_c) == 18 * sizeof(double),
                   "su3 layout assumption broken");
    double *d = (double *)su3;
    for (int i = 0; i < 18; i++)
        d[i] = 1.0;
}

void random_su3mat(su3_mat_c *su3, uint64_t *state)
{
    _Static_assert(sizeof(su3_mat_c) == 18 * sizeof(double),
                   "su3 layout assumption broken");
    double *d = (double *)su3;
    for (int i = 0; i < 18; i++)
        d[i] = local_rand(state);
}

void unit_su3vec(su3_vec_c *vec)
{
    _Static_assert(sizeof(su3_vec_c) == 6 * sizeof(double),
                   "su3_vec_c layout assumption broken");
    double *d = (double *)vec;
    for (int i = 0; i < 6; i++)
        d[i] = 1.0;
}

#endif // SU3_C
