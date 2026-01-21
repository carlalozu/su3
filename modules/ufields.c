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
