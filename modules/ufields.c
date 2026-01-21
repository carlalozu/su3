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


void usu3xusu3(su3_cdble *res, su3_cdble *u_field, su3_cdble *v_field, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        su3xsu3(&res[i], &u_field[i], &v_field[i]);
    }
}

void usu3xusu3vec(su3_vector_cdble *res_field, su3_cdble *u_field, su3_vector_cdble *v_field, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        su3xsu3vec(&res_field[i], &u_field[i], &v_field[i]);
    }
}

void usu3xusu3v(su3_field_dble *res, su3_field_dble *u_field, su3_field_dble *v_field, size_t size)
{
    // TODO: implement this function
}
# endif // UFLDS_C
