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
void fsu3matxsu3vec(su3_vec_field *restrict res, const su3_mat_field *restrict m_field, const su3_vec_field *restrict v_field, const size_t begin, const size_t end);
void fsu3matxsu3mat(su3_mat_field *restrict res, const su3_mat_field *restrict u, const su3_mat_field *restrict v, const size_t begin, const size_t end);
void fsu3mattrace(complexv *res, const su3_mat_field *ufield, const size_t begin, const size_t end);
#pragma omp end declare target
#endif // UFLDS_H
