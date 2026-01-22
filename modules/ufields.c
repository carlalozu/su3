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

/* SU(3) field AoS multiplication
 * Wrapper to perform
 * res_field = u_field * v_field
 * Where each field is an array of su3_mat of given size
 */
void usu3matxusu3mat(su3_mat *res, su3_mat *u_field, su3_mat *v_field, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        su3matxsu3mat(&res[i], &u_field[i], &v_field[i]);
    }
}

/* SU(3) matrix-vector field multiplication AoS
 * Wrapper to perform
 * res_field = u_field * v_field
 * Where each field is an array of su3_mat and su3_vec of given size
 */
void usu3matxusu3vec(su3_vec *res_field, su3_mat *u_field, su3_vec *v_field, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        su3matxsu3vec(&res_field[i], &u_field[i], &v_field[i]);
    }
}

/*
 * SU(3) matrix u times SU(3) vector SoA
 *
 * r.c1=(u*s).c1
 * r.c2=(u*s).c2
 * r.c3=(u*s).c3
 */
void fsu3matxsu3vec(su3_vec_field *res, su3_mat_field *u, su3_vec_field *v, size_t size)
{
    for (size_t i = 0; i < size; i++)
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
}

/* SU(3) matrix-matrix field multiplication SoA
 * 
 * res_field = m_field * v_field
 * Where each field is a su3_mat_field of given size
 */
void fsu3matxsu3mat(su3_mat_field *res, su3_mat_field *u_field, su3_mat_field *v_field, size_t size)
{
    // call fsu3matxsu3vec three times, once for each column
    fsu3matxsu3vec(&res->c1, u_field, &v_field->c1, size);
    fsu3matxsu3vec(&res->c2, u_field, &v_field->c2, size);
    fsu3matxsu3vec(&res->c3, u_field, &v_field->c3, size);
}

#endif // UFLDS_C
