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
#pragma omp declare target
LQCD_DEVICE void fsu3matxsu3vec(su3_vec_dble *res,const su3_mat_field *u,const su3_vec_field *v,const size_t i);
LQCD_DEVICE void fsu3matdagxsu3vec(su3_vec_dble *r, const su3_mat_field *u, const su3_vec_field *s, const size_t i);
LQCD_DEVICE void fsu3matxsu3mat(su3_mat_dble *res, const su3_mat_field *u, const su3_mat_field *v, const size_t i);
LQCD_DEVICE void fsu3matdagxsu3matdag(su3_mat_dble *w, const su3_mat_field *u, const su3_mat_field *v, const size_t i);
#pragma omp end declare target


double plaq_dble(su3_mat_c *u_field, su3_mat_c *v_field, su3_mat_c *w_field, su3_mat_c *x_field);
double plaq_dblev(su3_mat_field *u_fieldv, su3_mat_field *v_fieldv, su3_mat_field *w_fieldv, su3_mat_field *x_fieldv, size_t i);

#endif // UFLDS_H
