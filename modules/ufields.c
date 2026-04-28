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

/*
 * SU(3) matrix u times SU(3) vector SoA
 *
 * r.c1=(u*s).c1
 * r.c2=(u*s).c2
 * r.c3=(u*s).c3
 */

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

// SoA operations


#ifdef __CUDACC__
  #define LQCD_DEVICE    __device__
#else
  #define LQCD_DEVICE
#endif

LQCD_DEVICE void fsu3matxsu3vec(su3_vec_dble *res,const su3_mat_field *u,const su3_vec_field *v,const size_t i)
{
    res->c1re = u->c1.c1re[i] * v->c1re[i] - u->c1.c1im[i] * v->c1im[i] +
                u->c1.c2re[i] * v->c2re[i] - u->c1.c2im[i] * v->c2im[i] +
                u->c1.c3re[i] * v->c3re[i] - u->c1.c3im[i] * v->c3im[i];
    res->c1im = u->c1.c1re[i] * v->c1im[i] + u->c1.c1im[i] * v->c1re[i] +
                u->c1.c2re[i] * v->c2im[i] + u->c1.c2im[i] * v->c2re[i] +
                u->c1.c3re[i] * v->c3im[i] + u->c1.c3im[i] * v->c3re[i];
    res->c2re = u->c2.c1re[i] * v->c1re[i] - u->c2.c1im[i] * v->c1im[i] +
                u->c2.c2re[i] * v->c2re[i] - u->c2.c2im[i] * v->c2im[i] +
                u->c2.c3re[i] * v->c3re[i] - u->c2.c3im[i] * v->c3im[i];
    res->c2im = u->c2.c1re[i] * v->c1im[i] + u->c2.c1im[i] * v->c1re[i] +
                u->c2.c2re[i] * v->c2im[i] + u->c2.c2im[i] * v->c2re[i] +
                u->c2.c3re[i] * v->c3im[i] + u->c2.c3im[i] * v->c3re[i];
    res->c3re = u->c3.c1re[i] * v->c1re[i] - u->c3.c1im[i] * v->c1im[i] +
                u->c3.c2re[i] * v->c2re[i] - u->c3.c2im[i] * v->c2im[i] +
                u->c3.c3re[i] * v->c3re[i] - u->c3.c3im[i] * v->c3im[i];
    res->c3im = u->c3.c1re[i] * v->c1im[i] + u->c3.c1im[i] * v->c1re[i] +
                u->c3.c2re[i] * v->c2im[i] + u->c3.c2im[i] * v->c2re[i] +
                u->c3.c3re[i] * v->c3im[i] + u->c3.c3im[i] * v->c3re[i];
}


/*
 * SU(3) matrix u^dagger times SU(3) vector s
 *
 * r.c1=(u^dagger*s).c1
 * r.c2=(u^dagger*s).c2
 * r.c3=(u^dagger*s).c3
 */

LQCD_DEVICE void fsu3matdagxsu3vec(su3_vec_dble *r, const su3_mat_field *u, const su3_vec_field *s, const size_t i)
{
    r->c1re = u->c1.c1re[i] * s->c1re[i] + u->c1.c1im[i] * s->c1im[i] +
              u->c2.c1re[i] * s->c2re[i] + u->c2.c1im[i] * s->c2im[i] +
              u->c3.c1re[i] * s->c3re[i] + u->c3.c1im[i] * s->c3im[i];
    r->c1im = u->c1.c1re[i] * s->c1im[i] - u->c1.c1im[i] * s->c1re[i] +
              u->c2.c1re[i] * s->c2im[i] - u->c2.c1im[i] * s->c2re[i] +
              u->c3.c1re[i] * s->c3im[i] - u->c3.c1im[i] * s->c3re[i];
    r->c2re = u->c1.c2re[i] * s->c1re[i] + u->c1.c2im[i] * s->c1im[i] +
              u->c2.c2re[i] * s->c2re[i] + u->c2.c2im[i] * s->c2im[i] +
              u->c3.c2re[i] * s->c3re[i] + u->c3.c2im[i] * s->c3im[i];
    r->c2im = u->c1.c2re[i] * s->c1im[i] - u->c1.c2im[i] * s->c1re[i] +
              u->c2.c2re[i] * s->c2im[i] - u->c2.c2im[i] * s->c2re[i] +
              u->c3.c2re[i] * s->c3im[i] - u->c3.c2im[i] * s->c3re[i];
    r->c3re = u->c1.c3re[i] * s->c1re[i] + u->c1.c3im[i] * s->c1im[i] +
              u->c2.c3re[i] * s->c2re[i] + u->c2.c3im[i] * s->c2im[i] +
              u->c3.c3re[i] * s->c3re[i] + u->c3.c3im[i] * s->c3im[i];
    r->c3im = u->c1.c3re[i] * s->c1im[i] - u->c1.c3im[i] * s->c1re[i] +
              u->c2.c3re[i] * s->c2im[i] - u->c2.c3im[i] * s->c2re[i] +
              u->c3.c3re[i] * s->c3im[i] - u->c3.c3im[i] * s->c3re[i];
}


LQCD_DEVICE void fsu3matxsu3mat(su3_mat_dble *res, const su3_mat_field *u, const su3_mat_field *v, const size_t i)
{
    res->c1.c1re = u->c1.c1re[i] * v->c1.c1re[i] - u->c1.c1im[i] * v->c1.c1im[i] +
                   u->c1.c2re[i] * v->c1.c2re[i] - u->c1.c2im[i] * v->c1.c2im[i] +
                   u->c1.c3re[i] * v->c1.c3re[i] - u->c1.c3im[i] * v->c1.c3im[i];
    res->c1.c1im = u->c1.c1re[i] * v->c1.c1im[i] + u->c1.c1im[i] * v->c1.c1re[i] +
                   u->c1.c2re[i] * v->c1.c2im[i] + u->c1.c2im[i] * v->c1.c2re[i] +
                   u->c1.c3re[i] * v->c1.c3im[i] + u->c1.c3im[i] * v->c1.c3re[i];
    res->c1.c2re = u->c2.c1re[i] * v->c1.c1re[i] - u->c2.c1im[i] * v->c1.c1im[i] +
                   u->c2.c2re[i] * v->c1.c2re[i] - u->c2.c2im[i] * v->c1.c2im[i] +
                   u->c2.c3re[i] * v->c1.c3re[i] - u->c2.c3im[i] * v->c1.c3im[i];
    res->c1.c2im = u->c2.c1re[i] * v->c1.c1im[i] + u->c2.c1im[i] * v->c1.c1re[i] +
                   u->c2.c2re[i] * v->c1.c2im[i] + u->c2.c2im[i] * v->c1.c2re[i] +
                   u->c2.c3re[i] * v->c1.c3im[i] + u->c2.c3im[i] * v->c1.c3re[i];
    res->c1.c3re = u->c3.c1re[i] * v->c1.c1re[i] - u->c3.c1im[i] * v->c1.c1im[i] +
                   u->c3.c2re[i] * v->c1.c2re[i] - u->c3.c2im[i] * v->c1.c2im[i] +
                   u->c3.c3re[i] * v->c1.c3re[i] - u->c3.c3im[i] * v->c1.c3im[i];
    res->c1.c3im = u->c3.c1re[i] * v->c1.c1im[i] + u->c3.c1im[i] * v->c1.c1re[i] +
                   u->c3.c2re[i] * v->c1.c2im[i] + u->c3.c2im[i] * v->c1.c2re[i] +
                   u->c3.c3re[i] * v->c1.c3im[i] + u->c3.c3im[i] * v->c1.c3re[i];

    res->c2.c1re = u->c1.c1re[i] * v->c2.c1re[i] - u->c1.c1im[i] * v->c2.c1im[i] +
                   u->c1.c2re[i] * v->c2.c2re[i] - u->c1.c2im[i] * v->c2.c2im[i] +
                   u->c1.c3re[i] * v->c2.c3re[i] - u->c1.c3im[i] * v->c2.c3im[i];
    res->c2.c1im = u->c1.c1re[i] * v->c2.c1im[i] + u->c1.c1im[i] * v->c2.c1re[i] +
                   u->c1.c2re[i] * v->c2.c2im[i] + u->c1.c2im[i] * v->c2.c2re[i] +
                   u->c1.c3re[i] * v->c2.c3im[i] + u->c1.c3im[i] * v->c2.c3re[i];
    res->c2.c2re = u->c2.c1re[i] * v->c2.c1re[i] - u->c2.c1im[i] * v->c2.c1im[i] +
                   u->c2.c2re[i] * v->c2.c2re[i] - u->c2.c2im[i] * v->c2.c2im[i] +
                   u->c2.c3re[i] * v->c2.c3re[i] - u->c2.c3im[i] * v->c2.c3im[i];
    res->c2.c2im = u->c2.c1re[i] * v->c2.c1im[i] + u->c2.c1im[i] * v->c2.c1re[i] +
                   u->c2.c2re[i] * v->c2.c2im[i] + u->c2.c2im[i] * v->c2.c2re[i] +
                   u->c2.c3re[i] * v->c2.c3im[i] + u->c2.c3im[i] * v->c2.c3re[i];
    res->c2.c3re = u->c3.c1re[i] * v->c2.c1re[i] - u->c3.c1im[i] * v->c2.c1im[i] +
                   u->c3.c2re[i] * v->c2.c2re[i] - u->c3.c2im[i] * v->c2.c2im[i] +
                   u->c3.c3re[i] * v->c2.c3re[i] - u->c3.c3im[i] * v->c2.c3im[i];
    res->c2.c3im = u->c3.c1re[i] * v->c2.c1im[i] + u->c3.c1im[i] * v->c2.c1re[i] +
                   u->c3.c2re[i] * v->c2.c2im[i] + u->c3.c2im[i] * v->c2.c2re[i] +
                   u->c3.c3re[i] * v->c2.c3im[i] + u->c3.c3im[i] * v->c2.c3re[i];

    res->c3.c1re = u->c1.c1re[i] * v->c3.c1re[i] - u->c1.c1im[i] * v->c3.c1im[i] +
                   u->c1.c2re[i] * v->c3.c2re[i] - u->c1.c2im[i] * v->c3.c2im[i] +
                   u->c1.c3re[i] * v->c3.c3re[i] - u->c1.c3im[i] * v->c3.c3im[i];
    res->c3.c1im = u->c1.c1re[i] * v->c3.c1im[i] + u->c1.c1im[i] * v->c3.c1re[i] +
                   u->c1.c2re[i] * v->c3.c2im[i] + u->c1.c2im[i] * v->c3.c2re[i] +
                   u->c1.c3re[i] * v->c3.c3im[i] + u->c1.c3im[i] * v->c3.c3re[i];
    res->c3.c2re = u->c2.c1re[i] * v->c3.c1re[i] - u->c2.c1im[i] * v->c3.c1im[i] +
                   u->c2.c2re[i] * v->c3.c2re[i] - u->c2.c2im[i] * v->c3.c2im[i] +
                   u->c2.c3re[i] * v->c3.c3re[i] - u->c2.c3im[i] * v->c3.c3im[i];
    res->c3.c2im = u->c2.c1re[i] * v->c3.c1im[i] + u->c2.c1im[i] * v->c3.c1re[i] +
                   u->c2.c2re[i] * v->c3.c2im[i] + u->c2.c2im[i] * v->c3.c2re[i] +
                   u->c2.c3re[i] * v->c3.c3im[i] + u->c2.c3im[i] * v->c3.c3re[i];
    res->c3.c3re = u->c3.c1re[i] * v->c3.c1re[i] - u->c3.c1im[i] * v->c3.c1im[i] +
                   u->c3.c2re[i] * v->c3.c2re[i] - u->c3.c2im[i] * v->c3.c2im[i] +
                   u->c3.c3re[i] * v->c3.c3re[i] - u->c3.c3im[i] * v->c3.c3im[i];
    res->c3.c3im = u->c3.c1re[i] * v->c3.c1im[i] + u->c3.c1im[i] * v->c3.c1re[i] +
                   u->c3.c2re[i] * v->c3.c2im[i] + u->c3.c2im[i] * v->c3.c2re[i] +
                   u->c3.c3re[i] * v->c3.c3im[i] + u->c3.c3im[i] * v->c3.c3re[i];
}


/*
 * Computes w=u^dag*v^dag assuming that w is different from u and v.
 */

LQCD_DEVICE void fsu3matdagxsu3matdag(su3_mat_dble *w, const su3_mat_field *u, const su3_mat_field *v, const size_t i)
{
    w->c1.c1re = u->c1.c1re[i] * v->c1.c1re[i] + u->c1.c1im[i] * -v->c1.c1im[i] +
                 u->c2.c1re[i] * v->c1.c2re[i] + u->c2.c1im[i] * -v->c1.c2im[i] +
                 u->c3.c1re[i] * v->c1.c3re[i] + u->c3.c1im[i] * -v->c1.c3im[i];
    w->c1.c1im = u->c1.c1re[i] * -v->c1.c1im[i] - u->c1.c1im[i] * v->c1.c1re[i] +
                 u->c2.c1re[i] * -v->c1.c2im[i] - u->c2.c1im[i] * v->c1.c2re[i] +
                 u->c3.c1re[i] * -v->c1.c3im[i] - u->c3.c1im[i] * v->c1.c3re[i];
    w->c2.c1re = u->c1.c2re[i] * v->c1.c1re[i] + u->c1.c2im[i] * -v->c1.c1im[i] +
                 u->c2.c2re[i] * v->c1.c2re[i] + u->c2.c2im[i] * -v->c1.c2im[i] +
                 u->c3.c2re[i] * v->c1.c3re[i] + u->c3.c2im[i] * -v->c1.c3im[i];
    w->c2.c1im = u->c1.c2re[i] * -v->c1.c1im[i] - u->c1.c2im[i] * v->c1.c1re[i] +
                 u->c2.c2re[i] * -v->c1.c2im[i] - u->c2.c2im[i] * v->c1.c2re[i] +
                 u->c3.c2re[i] * -v->c1.c3im[i] - u->c3.c2im[i] * v->c1.c3re[i];
    w->c3.c1re = u->c1.c3re[i] * v->c1.c1re[i] + u->c1.c3im[i] * -v->c1.c1im[i] +
                 u->c2.c3re[i] * v->c1.c2re[i] + u->c2.c3im[i] * -v->c1.c2im[i] +
                 u->c3.c3re[i] * v->c1.c3re[i] + u->c3.c3im[i] * -v->c1.c3im[i];
    w->c3.c1im = u->c1.c3re[i] * -v->c1.c1im[i] - u->c1.c3im[i] * v->c1.c1re[i] +
                 u->c2.c3re[i] * -v->c1.c2im[i] - u->c2.c3im[i] * v->c1.c2re[i] +
                 u->c3.c3re[i] * -v->c1.c3im[i] - u->c3.c3im[i] * v->c1.c3re[i];

    w->c1.c2re = u->c1.c1re[i] * v->c2.c1re[i] + u->c1.c1im[i] * -v->c2.c1im[i] +
                 u->c2.c1re[i] * v->c2.c2re[i] + u->c2.c1im[i] * -v->c2.c2im[i] +
                 u->c3.c1re[i] * v->c2.c3re[i] + u->c3.c1im[i] * -v->c2.c3im[i];
    w->c1.c2im = u->c1.c1re[i] * -v->c2.c1im[i] - u->c1.c1im[i] * v->c2.c1re[i] +
                 u->c2.c1re[i] * -v->c2.c2im[i] - u->c2.c1im[i] * v->c2.c2re[i] +
                 u->c3.c1re[i] * -v->c2.c3im[i] - u->c3.c1im[i] * v->c2.c3re[i];
    w->c2.c2re = u->c1.c2re[i] * v->c2.c1re[i] + u->c1.c2im[i] * -v->c2.c1im[i] +
                 u->c2.c2re[i] * v->c2.c2re[i] + u->c2.c2im[i] * -v->c2.c2im[i] +
                 u->c3.c2re[i] * v->c2.c3re[i] + u->c3.c2im[i] * -v->c2.c3im[i];
    w->c2.c2im = u->c1.c2re[i] * -v->c2.c1im[i] - u->c1.c2im[i] * v->c2.c1re[i] +
                 u->c2.c2re[i] * -v->c2.c2im[i] - u->c2.c2im[i] * v->c2.c2re[i] +
                 u->c3.c2re[i] * -v->c2.c3im[i] - u->c3.c2im[i] * v->c2.c3re[i];
    w->c3.c2re = u->c1.c3re[i] * v->c2.c1re[i] + u->c1.c3im[i] * -v->c2.c1im[i] +
                 u->c2.c3re[i] * v->c2.c2re[i] + u->c2.c3im[i] * -v->c2.c2im[i] +
                 u->c3.c3re[i] * v->c2.c3re[i] + u->c3.c3im[i] * -v->c2.c3im[i];
    w->c3.c2im = u->c1.c3re[i] * -v->c2.c1im[i] - u->c1.c3im[i] * v->c2.c1re[i] +
                 u->c2.c3re[i] * -v->c2.c2im[i] - u->c2.c3im[i] * v->c2.c2re[i] +
                 u->c3.c3re[i] * -v->c2.c3im[i] - u->c3.c3im[i] * v->c2.c3re[i];

    w->c1.c3re = u->c1.c1re[i] * v->c3.c1re[i] + u->c1.c1im[i] * -v->c3.c1im[i] +
                 u->c2.c1re[i] * v->c3.c2re[i] + u->c2.c1im[i] * -v->c3.c2im[i] +
                 u->c3.c1re[i] * v->c3.c3re[i] + u->c3.c1im[i] * -v->c3.c3im[i];
    w->c1.c3im = u->c1.c1re[i] * -v->c3.c1im[i] - u->c1.c1im[i] * v->c3.c1re[i] +
                 u->c2.c1re[i] * -v->c3.c2im[i] - u->c2.c1im[i] * v->c3.c2re[i] +
                 u->c3.c1re[i] * -v->c3.c3im[i] - u->c3.c1im[i] * v->c3.c3re[i];
    w->c2.c3re = u->c1.c2re[i] * v->c3.c1re[i] + u->c1.c2im[i] * -v->c3.c1im[i] +
                 u->c2.c2re[i] * v->c3.c2re[i] + u->c2.c2im[i] * -v->c3.c2im[i] +
                 u->c3.c2re[i] * v->c3.c3re[i] + u->c3.c2im[i] * -v->c3.c3im[i];
    w->c2.c3im = u->c1.c2re[i] * -v->c3.c1im[i] - u->c1.c2im[i] * v->c3.c1re[i] +
                 u->c2.c2re[i] * -v->c3.c2im[i] - u->c2.c2im[i] * v->c3.c2re[i] +
                 u->c3.c2re[i] * -v->c3.c3im[i] - u->c3.c2im[i] * v->c3.c3re[i];
    w->c3.c3re = u->c1.c3re[i] * v->c3.c1re[i] + u->c1.c3im[i] * -v->c3.c1im[i] +
                 u->c2.c3re[i] * v->c3.c2re[i] + u->c2.c3im[i] * -v->c3.c2im[i] +
                 u->c3.c3re[i] * v->c3.c3re[i] + u->c3.c3im[i] * -v->c3.c3im[i];
    w->c3.c3im = u->c1.c3re[i] * -v->c3.c1im[i] - u->c1.c3im[i] * v->c3.c1re[i] +
                 u->c2.c3re[i] * -v->c3.c2im[i] - u->c2.c3im[i] * v->c3.c2re[i] +
                 u->c3.c3re[i] * -v->c3.c3im[i] - u->c3.c3im[i] * v->c3.c3re[i];
}

double plaq_dble(su3_mat_c *u_field, su3_mat_c *v_field, su3_mat_c *w_field, su3_mat_c *x_field){

    su3_mat_c temp_a;
    su3_mat_c temp_b;
    su3matxsu3mat(&temp_a, &u_field, &v_field);
    su3matdagxsu3matdag(&temp_b, &w_field, &x_field);
    return su3matxsu3mat_retrace(&temp_a, &temp_b);
}



double plaq_dblev(su3_mat_field *u_fieldv, su3_mat_field *v_fieldv, su3_mat_field *w_fieldv, su3_mat_field *x_fieldv, size_t i){

    su3_mat_c temp_a;
    su3_mat_c temp_b;
    fsu3matxsu3mat(&temp_a, u_fieldv, v_fieldv, i);
    fsu3matdagxsu3matdag(&temp_b, w_fieldv, x_fieldv, i);
    return su3matdxsu3matd_retrace(&temp_a, &temp_b);
}

#endif // UFLDS_C
