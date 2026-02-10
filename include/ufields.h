/*******************************************************************************
 *
 * File uflds.h
 *
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef UFLDS_H
#define UFLDS_H

#include "su3.h"
#include "su3v.h"
#include "global.h"

// SoA operations
#pragma omp declare target
static inline __attribute__((always_inline)) void fsu3matxsu3vec(su3_vec_field *restrict res, const su3_mat_field *restrict u, const su3_vec_field *restrict v, const size_t i)
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
#pragma omp end declare target

/*
 * SU(3) matrix u^dagger times SU(3) vector s
 *
 * r.c1=(u^dagger*s).c1
 * r.c2=(u^dagger*s).c2
 * r.c3=(u^dagger*s).c3
 */
#pragma omp declare target
void fsu3matdagxsu3vec(su3_vec_field *r, const su3_mat_field *u, const su3_vec_field *s, const size_t i)
{
    r->c1re[i] = u->c1.c1re[i] * s->c1re[i] + u->c1.c1im[i] * s->c1im[i] +
                 u->c2.c1re[i] * s->c2re[i] + u->c2.c1im[i] * s->c2im[i] +
                 u->c3.c1re[i] * s->c3re[i] + u->c3.c1im[i] * s->c3im[i];
    r->c1im[i] = u->c1.c1re[i] * s->c1im[i] - u->c1.c1im[i] * s->c1re[i] +
                 u->c2.c1re[i] * s->c2im[i] - u->c2.c1im[i] * s->c2re[i] +
                 u->c3.c1re[i] * s->c3im[i] - u->c3.c1im[i] * s->c3re[i];
    r->c2re[i] = u->c1.c2re[i] * s->c1re[i] + u->c1.c2im[i] * s->c1im[i] +
                 u->c2.c2re[i] * s->c2re[i] + u->c2.c2im[i] * s->c2im[i] +
                 u->c3.c2re[i] * s->c3re[i] + u->c3.c2im[i] * s->c3im[i];
    r->c2im[i] = u->c1.c2re[i] * s->c1im[i] - u->c1.c2im[i] * s->c1re[i] +
                 u->c2.c2re[i] * s->c2im[i] - u->c2.c2im[i] * s->c2re[i] +
                 u->c3.c2re[i] * s->c3im[i] - u->c3.c2im[i] * s->c3re[i];
    r->c3re[i] = u->c1.c3re[i] * s->c1re[i] + u->c1.c3im[i] * s->c1im[i] +
                 u->c2.c3re[i] * s->c2re[i] + u->c2.c3im[i] * s->c2im[i] +
                 u->c3.c3re[i] * s->c3re[i] + u->c3.c3im[i] * s->c3im[i];
    r->c3im[i] = u->c1.c3re[i] * s->c1im[i] - u->c1.c3im[i] * s->c1re[i] +
                 u->c2.c3re[i] * s->c2im[i] - u->c2.c3im[i] * s->c2re[i] +
                 u->c3.c3re[i] * s->c3im[i] - u->c3.c3im[i] * s->c3re[i];
}
#pragma omp end declare target

#pragma omp declare target
static inline __attribute__((always_inline)) void fsu3mat_trace(complexv *res, const su3_mat_field *ufield, const size_t i)
{
    res->re[i] = ufield->c1.c1re[i] + ufield->c2.c2re[i] + ufield->c3.c3re[i];
    res->im[i] = ufield->c1.c1im[i] + ufield->c2.c2im[i] + ufield->c3.c3im[i];
}
#pragma omp end declare target

#pragma omp declare target
static inline __attribute__((always_inline)) void fsu3matxsu3mat_retrace(doublev *res, const su3_mat_field *u, const su3_mat_field *v, const size_t i)
{
    double tr_1 = 0.0;
    double tr_2 = 0.0;
    double tr_3 = 0.0;

    tr_1 += u->c1.c1re[i] * v->c1.c1re[i] - u->c1.c1im[i] * v->c1.c1im[i];
    tr_1 += u->c1.c2re[i] * v->c2.c1re[i] - u->c1.c2im[i] * v->c2.c1im[i];
    tr_1 += u->c1.c3re[i] * v->c3.c1re[i] - u->c1.c3im[i] * v->c3.c1im[i];

    tr_2 += u->c2.c1re[i] * v->c1.c2re[i] - u->c2.c1im[i] * v->c1.c2im[i];
    tr_2 += u->c2.c2re[i] * v->c2.c2re[i] - u->c2.c2im[i] * v->c2.c2im[i];
    tr_2 += u->c2.c3re[i] * v->c3.c2re[i] - u->c2.c3im[i] * v->c3.c2im[i];

    tr_3 += u->c3.c1re[i] * v->c1.c3re[i] - u->c3.c1im[i] * v->c1.c3im[i];
    tr_3 += u->c3.c2re[i] * v->c2.c3re[i] - u->c3.c2im[i] * v->c2.c3im[i];
    tr_3 += u->c3.c3re[i] * v->c3.c3re[i] - u->c3.c3im[i] * v->c3.c3im[i];

    res->base[i] = tr_1 + tr_2 + tr_3;
}
#pragma omp end declare target

#pragma omp declare target
static inline __attribute__((always_inline)) void fsu3matxsu3mat(su3_mat_field *restrict res, const su3_mat_field *restrict u, const su3_mat_field *restrict v, const size_t i)
{
    // if (res == u || res == v || u == v)
    // {
    //     fprintf(stderr,
    //             "Error in fsu3matxsu3mat: res aliases input field (res == u_field or res == v_field)\n");
    //     abort();
    // }
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
#pragma omp end declare target

/*
 * Computes w=u^dag*v^dag assuming that w is different from u and v.
 */
#pragma omp declare target
static inline __attribute__((always_inline))
void fsu3matdagxsu3matdag(su3_mat_field *restrict w, const su3_mat_field *restrict u, const su3_mat_field *restrict v, const size_t i)
{
    w->c1.c1re[i] = u->c1.c1re[i] * v->c1.c1re[i] + u->c1.c1im[i] * -v->c1.c1im[i] +
                    u->c2.c1re[i] * v->c1.c2re[i] + u->c2.c1im[i] * -v->c1.c2im[i] +
                    u->c3.c1re[i] * v->c1.c3re[i] + u->c3.c1im[i] * -v->c1.c3im[i];
    w->c1.c1im[i] = u->c1.c1re[i] * -v->c1.c1im[i] - u->c1.c1im[i] * v->c1.c1re[i] +
                    u->c2.c1re[i] * -v->c1.c2im[i] - u->c2.c1im[i] * v->c1.c2re[i] +
                    u->c3.c1re[i] * -v->c1.c3im[i] - u->c3.c1im[i] * v->c1.c3re[i];
    w->c2.c1re[i] = u->c1.c2re[i] * v->c1.c1re[i] + u->c1.c2im[i] * -v->c1.c1im[i] +
                    u->c2.c2re[i] * v->c1.c2re[i] + u->c2.c2im[i] * -v->c1.c2im[i] +
                    u->c3.c2re[i] * v->c1.c3re[i] + u->c3.c2im[i] * -v->c1.c3im[i];
    w->c2.c1im[i] = u->c1.c2re[i] * -v->c1.c1im[i] - u->c1.c2im[i] * v->c1.c1re[i] +
                    u->c2.c2re[i] * -v->c1.c2im[i] - u->c2.c2im[i] * v->c1.c2re[i] +
                    u->c3.c2re[i] * -v->c1.c3im[i] - u->c3.c2im[i] * v->c1.c3re[i];
    w->c3.c1re[i] = u->c1.c3re[i] * v->c1.c1re[i] + u->c1.c3im[i] * -v->c1.c1im[i] +
                    u->c2.c3re[i] * v->c1.c2re[i] + u->c2.c3im[i] * -v->c1.c2im[i] +
                    u->c3.c3re[i] * v->c1.c3re[i] + u->c3.c3im[i] * -v->c1.c3im[i];
    w->c3.c1im[i] = u->c1.c3re[i] * -v->c1.c1im[i] - u->c1.c3im[i] * v->c1.c1re[i] +
                    u->c2.c3re[i] * -v->c1.c2im[i] - u->c2.c3im[i] * v->c1.c2re[i] +
                    u->c3.c3re[i] * -v->c1.c3im[i] - u->c3.c3im[i] * v->c1.c3re[i];

    w->c1.c2re[i] = u->c1.c1re[i] * v->c2.c1re[i] + u->c1.c1im[i] * -v->c2.c1im[i] +
                    u->c2.c1re[i] * v->c2.c2re[i] + u->c2.c1im[i] * -v->c2.c2im[i] +
                    u->c3.c1re[i] * v->c2.c3re[i] + u->c3.c1im[i] * -v->c2.c3im[i];
    w->c1.c2im[i] = u->c1.c1re[i] * -v->c2.c1im[i] - u->c1.c1im[i] * v->c2.c1re[i] +
                    u->c2.c1re[i] * -v->c2.c2im[i] - u->c2.c1im[i] * v->c2.c2re[i] +
                    u->c3.c1re[i] * -v->c2.c3im[i] - u->c3.c1im[i] * v->c2.c3re[i];
    w->c2.c2re[i] = u->c1.c2re[i] * v->c2.c1re[i] + u->c1.c2im[i] * -v->c2.c1im[i] +
                    u->c2.c2re[i] * v->c2.c2re[i] + u->c2.c2im[i] * -v->c2.c2im[i] +
                    u->c3.c2re[i] * v->c2.c3re[i] + u->c3.c2im[i] * -v->c2.c3im[i];
    w->c2.c2im[i] = u->c1.c2re[i] * -v->c2.c1im[i] - u->c1.c2im[i] * v->c2.c1re[i] +
                    u->c2.c2re[i] * -v->c2.c2im[i] - u->c2.c2im[i] * v->c2.c2re[i] +
                    u->c3.c2re[i] * -v->c2.c3im[i] - u->c3.c2im[i] * v->c2.c3re[i];
    w->c3.c2re[i] = u->c1.c3re[i] * v->c2.c1re[i] + u->c1.c3im[i] * -v->c2.c1im[i] +
                    u->c2.c3re[i] * v->c2.c2re[i] + u->c2.c3im[i] * -v->c2.c2im[i] +
                    u->c3.c3re[i] * v->c2.c3re[i] + u->c3.c3im[i] * -v->c2.c3im[i];
    w->c3.c2im[i] = u->c1.c3re[i] * -v->c2.c1im[i] - u->c1.c3im[i] * v->c2.c1re[i] +
                    u->c2.c3re[i] * -v->c2.c2im[i] - u->c2.c3im[i] * v->c2.c2re[i] +
                    u->c3.c3re[i] * -v->c2.c3im[i] - u->c3.c3im[i] * v->c2.c3re[i];

    w->c1.c3re[i] = u->c1.c1re[i] * v->c3.c1re[i] + u->c1.c1im[i] * -v->c3.c1im[i] +
                    u->c2.c1re[i] * v->c3.c2re[i] + u->c2.c1im[i] * -v->c3.c2im[i] +
                    u->c3.c1re[i] * v->c3.c3re[i] + u->c3.c1im[i] * -v->c3.c3im[i];
    w->c1.c3im[i] = u->c1.c1re[i] * -v->c3.c1im[i] - u->c1.c1im[i] * v->c3.c1re[i] +
                    u->c2.c1re[i] * -v->c3.c2im[i] - u->c2.c1im[i] * v->c3.c2re[i] +
                    u->c3.c1re[i] * -v->c3.c3im[i] - u->c3.c1im[i] * v->c3.c3re[i];
    w->c2.c3re[i] = u->c1.c2re[i] * v->c3.c1re[i] + u->c1.c2im[i] * -v->c3.c1im[i] +
                    u->c2.c2re[i] * v->c3.c2re[i] + u->c2.c2im[i] * -v->c3.c2im[i] +
                    u->c3.c2re[i] * v->c3.c3re[i] + u->c3.c2im[i] * -v->c3.c3im[i];
    w->c2.c3im[i] = u->c1.c2re[i] * -v->c3.c1im[i] - u->c1.c2im[i] * v->c3.c1re[i] +
                    u->c2.c2re[i] * -v->c3.c2im[i] - u->c2.c2im[i] * v->c3.c2re[i] +
                    u->c3.c2re[i] * -v->c3.c3im[i] - u->c3.c2im[i] * v->c3.c3re[i];
    w->c3.c3re[i] = u->c1.c3re[i] * v->c3.c1re[i] + u->c1.c3im[i] * -v->c3.c1im[i] +
                    u->c2.c3re[i] * v->c3.c2re[i] + u->c2.c3im[i] * -v->c3.c2im[i] +
                    u->c3.c3re[i] * v->c3.c3re[i] + u->c3.c3im[i] * -v->c3.c3im[i];
    w->c3.c3im[i] = u->c1.c3re[i] * -v->c3.c1im[i] - u->c1.c3im[i] * v->c3.c1re[i] +
                    u->c2.c3re[i] * -v->c3.c2im[i] - u->c2.c3im[i] * v->c3.c2re[i] +
                    u->c3.c3re[i] * -v->c3.c3im[i] - u->c3.c3im[i] * v->c3.c3re[i];
}
#pragma omp end declare target

#endif // UFLDS_H
