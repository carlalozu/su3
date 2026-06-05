
/*******************************************************************************
 *
 * File uflds.h
 *
 * Copyright (C) 2011, 2012, 2013 Martin Luescher
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef UFLDS_H
#define UFLDS_H

#ifndef SU3_H
#include "su3.h"
#endif

#ifndef SU3V_H
#include "su3v.h"
#endif

#include "profiler.h"
#include "global.h"

/* PLAQ_SUM_C */
extern double plaq_sum_dble(int icom);
extern double plaq_sum_dble_gpu(int icom);
extern double plaq_sum_dblev(su3_mat_field *u_fld, int icom);

/* UFLDS_C */
#ifdef __cplusplus
extern "C"
{
#endif
  extern su3_dble *udfld(void);
  extern void random_ud(void);
  extern void random_udv(su3_mat_field *su3mf);
#ifdef __cplusplus
}
#endif
extern prof_section compute;

PRAGMA_OMP_BEGIN
DEVICE_KEYWORD double su3matdxsu3matd_retrace(const su3_mat_dble *u, const su3_mat_dble *v)
{
  double tr_1 = u->c1.c1re * v->c1.c1re - u->c1.c1im * v->c1.c1im + u->c1.c2re * v->c2.c1re - u->c1.c2im * v->c2.c1im + u->c1.c3re * v->c3.c1re - u->c1.c3im * v->c3.c1im;
  double tr_2 = u->c2.c1re * v->c1.c2re - u->c2.c1im * v->c1.c2im + u->c2.c2re * v->c2.c2re - u->c2.c2im * v->c2.c2im + u->c2.c3re * v->c3.c2re - u->c2.c3im * v->c3.c2im;
  double tr_3 = u->c3.c1re * v->c1.c3re - u->c3.c1im * v->c1.c3im + u->c3.c2re * v->c2.c3re - u->c3.c2im * v->c2.c3im + u->c3.c3re * v->c3.c3re - u->c3.c3im * v->c3.c3im;
  return tr_1 + tr_2 + tr_3;
}
PRAGMA_OMP_END

PRAGMA_OMP_BEGIN
DEVICE_KEYWORD void _fsu3matconj_inverse_multiply(const su3_mat_field *u, const su3_vec_field *v, su3_vector_dble *res, int ip0, int ip1)
{
  res->c1.re = u->c1.c1re[ip0] * v->c1re[ip1] - u->c1.c1im[ip0] * v->c1im[ip1] +
               u->c2.c1re[ip0] * v->c2re[ip1] - u->c2.c1im[ip0] * v->c2im[ip1] +
               u->c3.c1re[ip0] * v->c3re[ip1] - u->c3.c1im[ip0] * v->c3im[ip1];
  res->c1.im = -u->c1.c1re[ip0] * v->c1im[ip1] - u->c1.c1im[ip0] * v->c1re[ip1] +
               -u->c2.c1re[ip0] * v->c2im[ip1] - u->c2.c1im[ip0] * v->c2re[ip1] +
               -u->c3.c1re[ip0] * v->c3im[ip1] - u->c3.c1im[ip0] * v->c3re[ip1];
  res->c2.re = u->c1.c2re[ip0] * v->c1re[ip1] - u->c1.c2im[ip0] * v->c1im[ip1] +
               u->c2.c2re[ip0] * v->c2re[ip1] - u->c2.c2im[ip0] * v->c2im[ip1] +
               u->c3.c2re[ip0] * v->c3re[ip1] - u->c3.c2im[ip0] * v->c3im[ip1];
  res->c2.im = -u->c1.c2re[ip0] * v->c1im[ip1] - u->c1.c2im[ip0] * v->c1re[ip1] +
               -u->c2.c2re[ip0] * v->c2im[ip1] - u->c2.c2im[ip0] * v->c2re[ip1] +
               -u->c3.c2re[ip0] * v->c3im[ip1] - u->c3.c2im[ip0] * v->c3re[ip1];
  res->c3.re = u->c1.c3re[ip0] * v->c1re[ip1] - u->c1.c3im[ip0] * v->c1im[ip1] +
               u->c2.c3re[ip0] * v->c2re[ip1] - u->c2.c3im[ip0] * v->c2im[ip1] +
               u->c3.c3re[ip0] * v->c3re[ip1] - u->c3.c3im[ip0] * v->c3im[ip1];
  res->c3.im = -u->c1.c3re[ip0] * v->c1im[ip1] - u->c1.c3im[ip0] * v->c1re[ip1] +
               -u->c2.c3re[ip0] * v->c2im[ip1] - u->c2.c3im[ip0] * v->c2re[ip1] +
               -u->c3.c3re[ip0] * v->c3im[ip1] - u->c3.c3im[ip0] * v->c3re[ip1];
}
PRAGMA_OMP_END
/*
 * Computes w=u^dag*v^dag assuming that w is different from u and v.
 */
PRAGMA_OMP_BEGIN
DEVICE_KEYWORD void fsu3matdagxsu3matdag(
    su3_dble *res, const su3_mat_field *u, int ip0, int ip1)
{
  su3_vec_field psi;
  su3_vector_dble chi;

  psi = (*u).c1;
  _fsu3matconj_inverse_multiply(u, &psi, &chi, ip0, ip1);
  (*res).c11 = chi.c1;
  (*res).c21 = chi.c2;
  (*res).c31 = chi.c3;

  psi = (*u).c2;
  _fsu3matconj_inverse_multiply(u, &psi, &chi, ip0, ip1);
  (*res).c12 = chi.c1;
  (*res).c22 = chi.c2;
  (*res).c32 = chi.c3;

  psi = (*u).c3;
  _fsu3matconj_inverse_multiply(u, &psi, &chi, ip0, ip1);
  (*res).c13 = chi.c1;
  (*res).c23 = chi.c2;
  (*res).c33 = chi.c3;
}
PRAGMA_OMP_END

// SoA operations
PRAGMA_OMP_BEGIN
DEVICE_KEYWORD void fsu3matxsu3vec(const su3_mat_field *u, const su3_vec_field *v, su3_vector_dble *res, int ip0, int ip1)
{
  res->c1.re = u->c1.c1re[ip0] * v->c1re[ip1] - u->c1.c1im[ip0] * v->c1im[ip1] +
               u->c1.c2re[ip0] * v->c2re[ip1] - u->c1.c2im[ip0] * v->c2im[ip1] +
               u->c1.c3re[ip0] * v->c3re[ip1] - u->c1.c3im[ip0] * v->c3im[ip1];
  res->c1.im = u->c1.c1re[ip0] * v->c1im[ip1] + u->c1.c1im[ip0] * v->c1re[ip1] +
               u->c1.c2re[ip0] * v->c2im[ip1] + u->c1.c2im[ip0] * v->c2re[ip1] +
               u->c1.c3re[ip0] * v->c3im[ip1] + u->c1.c3im[ip0] * v->c3re[ip1];
  res->c2.re = u->c2.c1re[ip0] * v->c1re[ip1] - u->c2.c1im[ip0] * v->c1im[ip1] +
               u->c2.c2re[ip0] * v->c2re[ip1] - u->c2.c2im[ip0] * v->c2im[ip1] +
               u->c2.c3re[ip0] * v->c3re[ip1] - u->c2.c3im[ip0] * v->c3im[ip1];
  res->c2.im = u->c2.c1re[ip0] * v->c1im[ip1] + u->c2.c1im[ip0] * v->c1re[ip1] +
               u->c2.c2re[ip0] * v->c2im[ip1] + u->c2.c2im[ip0] * v->c2re[ip1] +
               u->c2.c3re[ip0] * v->c3im[ip1] + u->c2.c3im[ip0] * v->c3re[ip1];
  res->c3.re = u->c3.c1re[ip0] * v->c1re[ip1] - u->c3.c1im[ip0] * v->c1im[ip1] +
               u->c3.c2re[ip0] * v->c2re[ip1] - u->c3.c2im[ip0] * v->c2im[ip1] +
               u->c3.c3re[ip0] * v->c3re[ip1] - u->c3.c3im[ip0] * v->c3im[ip1];
  res->c3.im = u->c3.c1re[ip0] * v->c1im[ip1] + u->c3.c1im[ip0] * v->c1re[ip1] +
               u->c3.c2re[ip0] * v->c2im[ip1] + u->c3.c2im[ip0] * v->c2re[ip1] +
               u->c3.c3re[ip0] * v->c3im[ip1] + u->c3.c3im[ip0] * v->c3re[ip1];
}
PRAGMA_OMP_END

PRAGMA_OMP_BEGIN
DEVICE_KEYWORD void fsu3matxsu3mat(
    su3_dble *res, const su3_mat_field *u, int ip0, int ip1)
{
  // printf("n: %i, ix: %i, ip: (%i, %i) \n", n, i, ip2, ip3);

  su3_vec_field psi;
  su3_vector_dble chi;

  psi.c1re = (*u).c1.c1re;
  psi.c1im = (*u).c1.c1im;
  psi.c2re = (*u).c2.c1re;
  psi.c2im = (*u).c2.c1im;
  psi.c3re = (*u).c3.c1re;
  psi.c3im = (*u).c3.c1im;
  fsu3matxsu3vec(u, &psi, &chi, ip0, ip1);
  (*res).c11 = chi.c1;
  (*res).c21 = chi.c2;
  (*res).c31 = chi.c3;

  psi.c1re = (*u).c1.c2re;
  psi.c1im = (*u).c1.c2im;
  psi.c2re = (*u).c2.c2re;
  psi.c2im = (*u).c2.c2im;
  psi.c3re = (*u).c3.c2re;
  psi.c3im = (*u).c3.c2im;
  fsu3matxsu3vec(u, &psi, &chi, ip0, ip1);
  (*res).c12 = chi.c1;
  (*res).c22 = chi.c2;
  (*res).c32 = chi.c3;

  psi.c1re = (*u).c1.c3re;
  psi.c1im = (*u).c1.c3im;
  psi.c2re = (*u).c2.c3re;
  psi.c2im = (*u).c2.c3im;
  psi.c3re = (*u).c3.c3re;
  psi.c3im = (*u).c3.c3im;
  fsu3matxsu3vec(u, &psi, &chi, ip0, ip1);
  (*res).c13 = chi.c1;
  (*res).c23 = chi.c2;
  (*res).c33 = chi.c3;
}
PRAGMA_OMP_END

#endif // UFLDS_H
