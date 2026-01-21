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

int alloc_su3_cdble(su3_cdble **su3)
{
    // confirm size of su3_cdble is 18 * sizeof(double)
    _Static_assert(sizeof(su3_cdble) == 18 * sizeof(double),
               "su3_cdble layout assumption broken");
    *su3 = malloc(sizeof **su3);

    if (*su3 == NULL) {
        return -1;
    }
    return 0;
}

void unit_su3_cdble(su3_cdble *su3, int value){

    _Static_assert(sizeof(su3_cdble) == 18 * sizeof(double),
    "su3_cdble layout assumption broken");
    memset(su3, 0, sizeof *su3);
    su3->c11.re = (double)value;
    su3->c22.re = (double)value;
    su3->c33.re = (double)value;
}

int alloc_ufield(su3_cdble **field, int size)
{
    // confirm size of su3_cdble is 18 * sizeof(double)
    _Static_assert(sizeof(su3_cdble) == 18 * sizeof(double),
               "su3_cdble layout assumption broken");
    for (int i = 0; i < size; i++) {
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

# endif // UFLDS_C
