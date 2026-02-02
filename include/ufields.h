/*******************************************************************************
 *
 * File uflds.h
 *
 * SoA operations
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef UFLDS_H
#define UFLDS_H

#include "su3.h"
#include "su3v.h"
#include "global.h"

/*
 * SU(3) matrix u times SU(3) vector SoA
 *
 * r.c1=(u*s).c1
 * r.c2=(u*s).c2
 * r.c3=(u*s).c3
 */
#pragma omp declare target
void fsu3matxsu3vec(su3_vec_field *restrict res, const su3_mat_field *restrict u, const su3_vec_field *restrict v, const size_t begin, const size_t end)
{
    if (res == v)
    {
        ERROR("Error in fsu3matxsu3vec: res aliases input field (res == v)\n");
    }

#pragma omp simd
    for (size_t i = begin; i < end; i++)
    {
        res->c1re[i] = u->c11re[i] * v->c1re[i] - u->c11im[i] * v->c1im[i] +
                       u->c12re[i] * v->c2re[i] - u->c12im[i] * v->c2im[i] +
                       u->c13re[i] * v->c3re[i] - u->c13im[i] * v->c3im[i];
        res->c1im[i] = u->c11re[i] * v->c1im[i] + u->c11im[i] * v->c1re[i] +
                       u->c12re[i] * v->c2im[i] + u->c12im[i] * v->c2re[i] +
                       u->c13re[i] * v->c3im[i] + u->c13im[i] * v->c3re[i];
        res->c2re[i] = u->c21re[i] * v->c1re[i] - u->c21im[i] * v->c1im[i] +
                       u->c22re[i] * v->c2re[i] - u->c22im[i] * v->c2im[i] +
                       u->c23re[i] * v->c3re[i] - u->c23im[i] * v->c3im[i];
        res->c2im[i] = u->c21re[i] * v->c1im[i] + u->c21im[i] * v->c1re[i] +
                       u->c22re[i] * v->c2im[i] + u->c22im[i] * v->c2re[i] +
                       u->c23re[i] * v->c3im[i] + u->c23im[i] * v->c3re[i];
        res->c3re[i] = u->c31re[i] * v->c1re[i] - u->c31im[i] * v->c1im[i] +
                       u->c32re[i] * v->c2re[i] - u->c32im[i] * v->c2im[i] +
                       u->c33re[i] * v->c3re[i] - u->c33im[i] * v->c3im[i];
        res->c3im[i] = u->c31re[i] * v->c1im[i] + u->c31im[i] * v->c1re[i] +
                       u->c32re[i] * v->c2im[i] + u->c32im[i] * v->c2re[i] +
                       u->c33re[i] * v->c3im[i] + u->c33im[i] * v->c3re[i];
    }
}
#pragma omp end declare target

/* SU(3) matrix-matrix field multiplication SoA
 *
 * res_field = m_field * v_field
 * Where each field is a su3_mat_field of given size
 *
 * Bandwidth
 * from u = 18 load streams, 9 complex numbers
 * from v = 18 load streams
 * from res = 18 write streams
 * total = 54 doubles = 54 x 8 bytes = 432 bytes
 *
 * FLOPS
 * per matrix element = 11 operations, 6 muls, 5 adds
 * par complex matrix element = 22 operations
 * total = 9 * 22 = 198 FLOPS
 *
 * Intensity
 * 198 / 432 ≈ 0.46 flops/byte
 * = Memory bound
 */
#pragma omp declare target
void fsu3matxsu3mat(su3_mat_field *restrict res, const su3_mat_field *restrict u, const su3_mat_field *restrict v, const size_t begin, const size_t end)
{
    if (res == u || res == v || u == v)
    {
        ERROR("Error in fsu3matxsu3mat: res aliases input field (res == u or res == v)\n");
    }
    #pragma omp simd
    for (size_t i = begin; i < end; i++)
    {
        res->c11re[i] = u->c11re[i] * v->c11re[i] - u->c11im[i] * v->c11im[i] +
                        u->c12re[i] * v->c12re[i] - u->c12im[i] * v->c12im[i] +
                        u->c13re[i] * v->c13re[i] - u->c13im[i] * v->c13im[i];
        res->c11im[i] = u->c11re[i] * v->c11im[i] + u->c11im[i] * v->c11re[i] +
                        u->c12re[i] * v->c12im[i] + u->c12im[i] * v->c12re[i] +
                        u->c13re[i] * v->c13im[i] + u->c13im[i] * v->c13re[i];
        res->c12re[i] = u->c21re[i] * v->c11re[i] - u->c21im[i] * v->c11im[i] +
                        u->c22re[i] * v->c12re[i] - u->c22im[i] * v->c12im[i] +
                        u->c23re[i] * v->c13re[i] - u->c23im[i] * v->c13im[i];
        res->c12im[i] = u->c21re[i] * v->c11im[i] + u->c21im[i] * v->c11re[i] +
                        u->c22re[i] * v->c12im[i] + u->c22im[i] * v->c12re[i] +
                        u->c23re[i] * v->c13im[i] + u->c23im[i] * v->c13re[i];
        res->c13re[i] = u->c31re[i] * v->c11re[i] - u->c31im[i] * v->c11im[i] +
                        u->c32re[i] * v->c12re[i] - u->c32im[i] * v->c12im[i] +
                        u->c33re[i] * v->c13re[i] - u->c33im[i] * v->c13im[i];
        res->c13im[i] = u->c31re[i] * v->c11im[i] + u->c31im[i] * v->c11re[i] +
                        u->c32re[i] * v->c12im[i] + u->c32im[i] * v->c12re[i] +
                        u->c33re[i] * v->c13im[i] + u->c33im[i] * v->c13re[i];

        res->c21re[i] = u->c11re[i] * v->c21re[i] - u->c11im[i] * v->c21im[i] +
                        u->c12re[i] * v->c22re[i] - u->c12im[i] * v->c22im[i] +
                        u->c13re[i] * v->c23re[i] - u->c13im[i] * v->c23im[i];
        res->c21im[i] = u->c11re[i] * v->c21im[i] + u->c11im[i] * v->c21re[i] +
                        u->c12re[i] * v->c22im[i] + u->c12im[i] * v->c22re[i] +
                        u->c13re[i] * v->c23im[i] + u->c13im[i] * v->c23re[i];
        res->c22re[i] = u->c21re[i] * v->c21re[i] - u->c21im[i] * v->c21im[i] +
                        u->c22re[i] * v->c22re[i] - u->c22im[i] * v->c22im[i] +
                        u->c23re[i] * v->c23re[i] - u->c23im[i] * v->c23im[i];
        res->c22im[i] = u->c21re[i] * v->c21im[i] + u->c21im[i] * v->c21re[i] +
                        u->c22re[i] * v->c22im[i] + u->c22im[i] * v->c22re[i] +
                        u->c23re[i] * v->c23im[i] + u->c23im[i] * v->c23re[i];
        res->c23re[i] = u->c31re[i] * v->c21re[i] - u->c31im[i] * v->c21im[i] +
                        u->c32re[i] * v->c22re[i] - u->c32im[i] * v->c22im[i] +
                        u->c33re[i] * v->c23re[i] - u->c33im[i] * v->c23im[i];
        res->c23im[i] = u->c31re[i] * v->c21im[i] + u->c31im[i] * v->c21re[i] +
                        u->c32re[i] * v->c22im[i] + u->c32im[i] * v->c22re[i] +
                        u->c33re[i] * v->c23im[i] + u->c33im[i] * v->c23re[i];

        res->c31re[i] = u->c11re[i] * v->c31re[i] - u->c11im[i] * v->c31im[i] +
                        u->c12re[i] * v->c32re[i] - u->c12im[i] * v->c32im[i] +
                        u->c13re[i] * v->c33re[i] - u->c13im[i] * v->c33im[i];
        res->c31im[i] = u->c11re[i] * v->c31im[i] + u->c11im[i] * v->c31re[i] +
                        u->c12re[i] * v->c32im[i] + u->c12im[i] * v->c32re[i] +
                        u->c13re[i] * v->c33im[i] + u->c13im[i] * v->c33re[i];
        res->c32re[i] = u->c21re[i] * v->c31re[i] - u->c21im[i] * v->c31im[i] +
                        u->c22re[i] * v->c32re[i] - u->c22im[i] * v->c32im[i] +
                        u->c23re[i] * v->c33re[i] - u->c23im[i] * v->c33im[i];
        res->c32im[i] = u->c21re[i] * v->c31im[i] + u->c21im[i] * v->c31re[i] +
                        u->c22re[i] * v->c32im[i] + u->c22im[i] * v->c32re[i] +
                        u->c23re[i] * v->c33im[i] + u->c23im[i] * v->c33re[i];
        res->c33re[i] = u->c31re[i] * v->c31re[i] - u->c31im[i] * v->c31im[i] +
                        u->c32re[i] * v->c32re[i] - u->c32im[i] * v->c32im[i] +
                        u->c33re[i] * v->c33re[i] - u->c33im[i] * v->c33im[i];
        res->c33im[i] = u->c31re[i] * v->c31im[i] + u->c31im[i] * v->c31re[i] +
                        u->c32re[i] * v->c32im[i] + u->c32im[i] * v->c32re[i] +
                        u->c33re[i] * v->c33im[i] + u->c33im[i] * v->c33re[i];
    }
}
#pragma omp end declare target

#pragma omp declare target
void fsu3mattrace(complexv *res, const su3_mat_field *ufield, const size_t begin, const size_t end)
{
#pragma omp simd
    for (size_t i = begin; i < end; i++)
    {
        res->re[i] = ufield->c11re[i] + ufield->c22re[i] + ufield->c33re[i];
        res->im[i] = ufield->c11im[i] + ufield->c22im[i] + ufield->c33im[i];
    }
}
#pragma omp end declare target
#endif // UFLDS_H
