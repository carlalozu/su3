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

// AoS operations
void usu3matxusu3vec(su3_vec *res_field, su3_mat *u_field, su3_vec *v_field, const size_t size);
void usu3matxusu3mat(su3_mat *res, su3_mat *u_field, su3_mat *v_field, const size_t size);
void usu3mattrace(complex *res, su3_mat *u_field, const size_t size);

// SoA operations
void fsu3matxsu3vec(su3_vec_field *res, const su3_mat_field *m_field, const su3_vec_field *v_field, const size_t begin, const size_t end);
void fsu3matxsu3mat(su3_mat_field *restrict res, const su3_mat_field *restrict u, const su3_mat_field *restrict v, const size_t begin, const size_t end);
void fsu3mattrace(complexv *res, const su3_mat_field *ufield, const size_t begin, const size_t end);

#endif // UFLDS_H
