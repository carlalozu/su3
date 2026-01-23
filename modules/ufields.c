/*******************************************************************************
 *
 * File uflds.c
 *
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef UFLDS_C
#define UFLDS_C

#include "su3.h"
#include "su3v.h"
#include "global.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* SU(3) field AoS multiplication
 * Wrapper to perform
 * res_field = u_field * v_field
 * Where each field is an array of su3_mat of given size
 */
void usu3matxusu3mat(su3_mat *res, su3_mat *u_field, su3_mat *v_field, const size_t size)
{
#pragma omp for schedule(static)
    for (size_t i = 0; i < size; i++)
    {
        su3matxsu3mat(&res[i], &u_field[i], &v_field[i]);
    }
}

/* SU(3) matrix-vector field multiplication AoS
 * Wrapper to perform
 * res_field = u_field * v_field
 * Where each field is an array of su3_mat and su3_vec of given size
 */
void usu3matxusu3vec(su3_vec *res_field, su3_mat *u_field, su3_vec *v_field, const size_t size)
{
#pragma omp for schedule(static)
    for (size_t i = 0; i < size; i++)
    {
        su3matxsu3vec(&res_field[i], &u_field[i], &v_field[i]);
    }
}

void usu3mattrace(complex *res, const su3_mat *ufield, const size_t size)
{
#pragma omp for schedule(static)
    for (size_t i = 0; i < size; i++)
    {
        res[i] = su3_trace(&ufield[i]);
    }
}

/*
 * SU(3) matrix u times SU(3) vector SoA
 *
 * r.c1=(u*s).c1
 * r.c2=(u*s).c2
 * r.c3=(u*s).c3
 */
void fsu3matxsu3vec(su3_vec_field *res, const su3_mat_field *u, const su3_vec_field *v, const size_t begin, const size_t end)
{
    for (size_t i = begin; i < end; i++)
    {
        res->c1re[i] = u->c1.c1re[i] * v->c1re[i] - u->c1.c1im[i] * v->c1im[i] +
                       u->c1.c2re[i] * v->c2re[i] - u->c1.c2im[i] * v->c2im[i] +
                       u->c1.c3re[i] * v->c3re[i] - u->c1.c3im[i] * v->c3im[i];
        res->c1im[i] = u->c1.c1re[i] * v->c1im[i] + u->c1.c1im[i] * v->c1re[i] +
                       u->c1.c2re[i] * v->c2im[i] + u->c1.c2im[i] * v->c2re[i] +
                       u->c1.c3re[i] * v->c3im[i] + u->c1.c3im[i] * v->c3re[i];
        res->c2re[i] = u->c2.c1re[i] * v->c1re[i] - u->c2.c1im[i] * v->c1im[i] +
                       u->c2.c2re[i] * v->c2re[i] - u->c2.c2im[i] * v->c2im[i] +
                       u->c2.c3re[i] * v->c3re[i] - u->c2.c3im[i] * v->c3im[i];
        res->c2im[i] = u->c2.c1re[i] * v->c1im[i] + u->c2.c1im[i] * v->c1re[i] +
                       u->c2.c2re[i] * v->c2im[i] + u->c2.c2im[i] * v->c2re[i] +
                       u->c2.c3re[i] * v->c3im[i] + u->c2.c3im[i] * v->c3re[i];
        res->c3re[i] = u->c3.c1re[i] * v->c1re[i] - u->c3.c1im[i] * v->c1im[i] +
                       u->c3.c2re[i] * v->c2re[i] - u->c3.c2im[i] * v->c2im[i] +
                       u->c3.c3re[i] * v->c3re[i] - u->c3.c3im[i] * v->c3im[i];
        res->c3im[i] = u->c3.c1re[i] * v->c1im[i] + u->c3.c1im[i] * v->c1re[i] +
                       u->c3.c2re[i] * v->c2im[i] + u->c3.c2im[i] * v->c2re[i] +
                       u->c3.c3re[i] * v->c3im[i] + u->c3.c3im[i] * v->c3re[i];
    }
}

/* SU(3) matrix-matrix field multiplication SoA
 *
 * res_field = m_field * v_field
 * Where each field is a su3_mat_field of given size
 *
 *  Bandwidth
 *  from u = 18 load streams, 9 complex numbers
 *  from v = 18 load streams
 *  from res = 18 write streams
 *  total = 54 doubles = 54 x 8 bytes = 432 bytes
 *
 *  FLOPS
 *  per matrix element = 11 operations, 6 muls, 5 adds
 *  par complex matrix element = 22 operations
 *  per matrix = 18 * 11 = 198 FLOPS
 *
 *  Intensity
 *  198 / 432 ≈ 0.46 flops/byte
 */
void fsu3matxsu3mat(su3_mat_field *restrict res, const su3_mat_field *restrict u, const su3_mat_field *restrict v, const size_t begin, const size_t end)
{
    // if (res == u || res == v || u == v)
    // {
    //     fprintf(stderr,
    //             "Error in fsu3matxsu3mat: res aliases input field (res == u_field or res == v_field)\n");
    //     abort();
    // }

    for (size_t i = begin; i < end; i++)
    {
        res->c1.c1re[i] = u->c1.c1re[i] * v->c1.c1re[i] - u->c1.c1im[i] * v->c1.c1im[i] +
                          u->c1.c2re[i] * v->c1.c2re[i] - u->c1.c2im[i] * v->c1.c2im[i] +
                          u->c1.c3re[i] * v->c1.c3re[i] - u->c1.c3im[i] * v->c1.c3im[i];
        res->c1.c1im[i] = u->c1.c1re[i] * v->c1.c1im[i] + u->c1.c1im[i] * v->c1.c1re[i] +
                          u->c1.c2re[i] * v->c1.c2im[i] + u->c1.c2im[i] * v->c1.c2re[i] +
                          u->c1.c3re[i] * v->c1.c3im[i] + u->c1.c3im[i] * v->c1.c3re[i];
        res->c1.c2re[i] = u->c2.c1re[i] * v->c1.c1re[i] - u->c2.c1im[i] * v->c1.c1im[i] +
                          u->c2.c2re[i] * v->c1.c2re[i] - u->c2.c2im[i] * v->c1.c2im[i] +
                          u->c2.c3re[i] * v->c1.c3re[i] - u->c2.c3im[i] * v->c1.c3im[i];
        res->c1.c2im[i] = u->c2.c1re[i] * v->c1.c1im[i] + u->c2.c1im[i] * v->c1.c1re[i] +
                          u->c2.c2re[i] * v->c1.c2im[i] + u->c2.c2im[i] * v->c1.c2re[i] +
                          u->c2.c3re[i] * v->c1.c3im[i] + u->c2.c3im[i] * v->c1.c3re[i];
        res->c1.c3re[i] = u->c3.c1re[i] * v->c1.c1re[i] - u->c3.c1im[i] * v->c1.c1im[i] +
                          u->c3.c2re[i] * v->c1.c2re[i] - u->c3.c2im[i] * v->c1.c2im[i] +
                          u->c3.c3re[i] * v->c1.c3re[i] - u->c3.c3im[i] * v->c1.c3im[i];
        res->c1.c3im[i] = u->c3.c1re[i] * v->c1.c1im[i] + u->c3.c1im[i] * v->c1.c1re[i] +
                          u->c3.c2re[i] * v->c1.c2im[i] + u->c3.c2im[i] * v->c1.c2re[i] +
                          u->c3.c3re[i] * v->c1.c3im[i] + u->c3.c3im[i] * v->c1.c3re[i];

        res->c2.c1re[i] = u->c1.c1re[i] * v->c2.c1re[i] - u->c1.c1im[i] * v->c2.c1im[i] +
                          u->c1.c2re[i] * v->c2.c2re[i] - u->c1.c2im[i] * v->c2.c2im[i] +
                          u->c1.c3re[i] * v->c2.c3re[i] - u->c1.c3im[i] * v->c2.c3im[i];
        res->c2.c1im[i] = u->c1.c1re[i] * v->c2.c1im[i] + u->c1.c1im[i] * v->c2.c1re[i] +
                          u->c1.c2re[i] * v->c2.c2im[i] + u->c1.c2im[i] * v->c2.c2re[i] +
                          u->c1.c3re[i] * v->c2.c3im[i] + u->c1.c3im[i] * v->c2.c3re[i];
        res->c2.c2re[i] = u->c2.c1re[i] * v->c2.c1re[i] - u->c2.c1im[i] * v->c2.c1im[i] +
                          u->c2.c2re[i] * v->c2.c2re[i] - u->c2.c2im[i] * v->c2.c2im[i] +
                          u->c2.c3re[i] * v->c2.c3re[i] - u->c2.c3im[i] * v->c2.c3im[i];
        res->c2.c2im[i] = u->c2.c1re[i] * v->c2.c1im[i] + u->c2.c1im[i] * v->c2.c1re[i] +
                          u->c2.c2re[i] * v->c2.c2im[i] + u->c2.c2im[i] * v->c2.c2re[i] +
                          u->c2.c3re[i] * v->c2.c3im[i] + u->c2.c3im[i] * v->c2.c3re[i];
        res->c2.c3re[i] = u->c3.c1re[i] * v->c2.c1re[i] - u->c3.c1im[i] * v->c2.c1im[i] +
                          u->c3.c2re[i] * v->c2.c2re[i] - u->c3.c2im[i] * v->c2.c2im[i] +
                          u->c3.c3re[i] * v->c2.c3re[i] - u->c3.c3im[i] * v->c2.c3im[i];
        res->c2.c3im[i] = u->c3.c1re[i] * v->c2.c1im[i] + u->c3.c1im[i] * v->c2.c1re[i] +
                          u->c3.c2re[i] * v->c2.c2im[i] + u->c3.c2im[i] * v->c2.c2re[i] +
                          u->c3.c3re[i] * v->c2.c3im[i] + u->c3.c3im[i] * v->c2.c3re[i];

        res->c3.c1re[i] = u->c1.c1re[i] * v->c3.c1re[i] - u->c1.c1im[i] * v->c3.c1im[i] +
                          u->c1.c2re[i] * v->c3.c2re[i] - u->c1.c2im[i] * v->c3.c2im[i] +
                          u->c1.c3re[i] * v->c3.c3re[i] - u->c1.c3im[i] * v->c3.c3im[i];
        res->c3.c1im[i] = u->c1.c1re[i] * v->c3.c1im[i] + u->c1.c1im[i] * v->c3.c1re[i] +
                          u->c1.c2re[i] * v->c3.c2im[i] + u->c1.c2im[i] * v->c3.c2re[i] +
                          u->c1.c3re[i] * v->c3.c3im[i] + u->c1.c3im[i] * v->c3.c3re[i];
        res->c3.c2re[i] = u->c2.c1re[i] * v->c3.c1re[i] - u->c2.c1im[i] * v->c3.c1im[i] +
                          u->c2.c2re[i] * v->c3.c2re[i] - u->c2.c2im[i] * v->c3.c2im[i] +
                          u->c2.c3re[i] * v->c3.c3re[i] - u->c2.c3im[i] * v->c3.c3im[i];
        res->c3.c2im[i] = u->c2.c1re[i] * v->c3.c1im[i] + u->c2.c1im[i] * v->c3.c1re[i] +
                          u->c2.c2re[i] * v->c3.c2im[i] + u->c2.c2im[i] * v->c3.c2re[i] +
                          u->c2.c3re[i] * v->c3.c3im[i] + u->c2.c3im[i] * v->c3.c3re[i];
        res->c3.c3re[i] = u->c3.c1re[i] * v->c3.c1re[i] - u->c3.c1im[i] * v->c3.c1im[i] +
                          u->c3.c2re[i] * v->c3.c2re[i] - u->c3.c2im[i] * v->c3.c2im[i] +
                          u->c3.c3re[i] * v->c3.c3re[i] - u->c3.c3im[i] * v->c3.c3im[i];
        res->c3.c3im[i] = u->c3.c1re[i] * v->c3.c1im[i] + u->c3.c1im[i] * v->c3.c1re[i] +
                          u->c3.c2re[i] * v->c3.c2im[i] + u->c3.c2im[i] * v->c3.c2re[i] +
                          u->c3.c3re[i] * v->c3.c3im[i] + u->c3.c3im[i] * v->c3.c3re[i];
    }
}

void fsu3mattrace(complexv *res, const su3_mat_field *ufield, const size_t begin, const size_t end)
{
#pragma omp simd
    for (size_t i = begin; i < end; i++)
    {
        res->re[i] = ufield->c1.c1re[i] + ufield->c2.c2re[i] + ufield->c3.c3re[i];
        res->im[i] = ufield->c1.c1im[i] + ufield->c2.c2im[i] + ufield->c3.c3im[i];
    }
}

#endif // UFLDS_C
