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
void usu3matxusu3vec(su3_vec *res_field, su3_mat *u_field, su3_vec *v_field, size_t size);
void usu3matxusu3mat(su3_mat *res, su3_mat *u_field, su3_mat *v_field, size_t size);

// SoA operations
void fsu3matxsu3vec(su3_vec_field *res, su3_mat_field *m_field, su3_vec_field *v_field, size_t size);
void fsu3matxsu3mat(su3_mat_field *res, su3_mat_field *m_field, su3_mat_field *v_field, size_t size);

#endif // UFLDS_H
