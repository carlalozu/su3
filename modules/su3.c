
/*******************************************************************************
*
* File su3prod.c
*
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef SU3PROD_C
#define SU3PROD_C

#include "su3.h"

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

complex_dble add(complex_dble a, complex_dble b)
{
    return (complex_dble){a.re + b.re, a.im + b.im};
}

#endif // SU3PROD_C