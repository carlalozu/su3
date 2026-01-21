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
#include "global.h"
#include <stdio.h>
#include <stdlib.h>


int alloc_ufield(su3_cdble **field, size_t size)
{
    // confirm size of su3_cdble is 18 * sizeof(double)
    _Static_assert(sizeof(su3_cdble) == 18 * sizeof(double),
               "su3_cdble layout assumption broken");
    for (size_t i = 0; i < size; i++) {
        field[i] = malloc(sizeof(su3_cdble));
        if (field[i] == NULL) {
            return -1;
        }
    }
    if (*field == NULL) {
        return -1;
    }
    return 0;
}

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

# endif // UFLDS_C
