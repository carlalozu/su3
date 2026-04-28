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
#ifndef __CUDACC__
#include "global.h"
#endif

#ifdef __CUDACC__
  #define LQCD_DEVICE    __device__
#else
  #define LQCD_DEVICE
#endif

// SoA operations
LQCD_OMP_BEGIN
LQCD_DEVICE static inline void fsu3matxsu3vec(
    su3_vec_dble *restrict res,
    const su3_mat_field *restrict u,
    const su3_vec_field *restrict v,
    const size_t i)
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
LQCD_OMP_END


LQCD_OMP_BEGIN
LQCD_DEVICE static inline double su3matdxsu3matd_retrace(const su3_mat_dble *u, const su3_mat_dble *v)
{
    double tr_1 = u->c1.c1re * v->c1.c1re - u->c1.c1im * v->c1.c1im 
                + u->c1.c2re * v->c2.c1re - u->c1.c2im * v->c2.c1im 
                + u->c1.c3re * v->c3.c1re - u->c1.c3im * v->c3.c1im;
    double tr_2 = u->c2.c1re * v->c1.c2re - u->c2.c1im * v->c1.c2im 
                + u->c2.c2re * v->c2.c2re - u->c2.c2im * v->c2.c2im 
                + u->c2.c3re * v->c3.c2re - u->c2.c3im * v->c3.c2im;
    double tr_3 = u->c3.c1re * v->c1.c3re - u->c3.c1im * v->c1.c3im 
                + u->c3.c2re * v->c2.c3re - u->c3.c2im * v->c2.c3im 
                + u->c3.c3re * v->c3.c3re - u->c3.c3im * v->c3.c3im;
    return tr_1 + tr_2 + tr_3;
}
LQCD_OMP_END

LQCD_OMP_BEGIN
LQCD_DEVICE static inline void fsu3matxsu3mat(
    su3_mat_dble *restrict res,
    const su3_mat_field *restrict u,
    const su3_mat_field *restrict v,
    const size_t i)
{
    // if (res == u || res == v || u == v)
    // {
    //     fprintf(stderr,
    //             "Error in fsu3matxsu3mat: res aliases input field (res == u_field or res == v_field)\n");
    //     abort();
    // }
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
LQCD_OMP_END

/*
 * Computes w=u^dag*v^dag assuming that w is different from u and v.
 */
LQCD_OMP_BEGIN
LQCD_DEVICE static inline void fsu3matdagxsu3matdag(
    su3_mat_dble *restrict w,
    const su3_mat_field *restrict u,
    const su3_mat_field *restrict v,
    const size_t i)
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
LQCD_OMP_END

#endif // UFLDS_H
