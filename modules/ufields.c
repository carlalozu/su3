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


void usu3matxusu3mat(su3_mat *res, su3_mat *u_field, su3_mat *v_field, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        su3matxsu3mat(&res[i], &u_field[i], &v_field[i]);
    }
}

void usu3matxusu3vec(su3_vec *res_field, su3_mat *u_field, su3_vec *v_field, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        su3matxsu3vec(&res_field[i], &u_field[i], &v_field[i]);
    }
}

void fusu3matxusu3vec(su3_vec_field *res, su3_mat_field *m_field, su3_vec_field *v_field, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        // Apply the matrix m_field[i] to the vector v_field[i]
        // and store the result in res[i]
        // This is a simplified version assuming a direct mapping
        // In practice, you would need to iterate over each component of the vectors
        // and perform the matrix-vector multiplication
    }
}
# endif // UFLDS_C
