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
    double *d = (double *)(void *)su3;
    for (int i = 0; i < 9; i++) {
        d[i*2] = value;
        d[i*2+1] = 0.0;
    }
}
# endif // UFLDS_C
