
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

complex_dble add(complex_dble a, complex_dble b)
{
    return (complex_dble){a.re + b.re, a.im + b.im};
}

#endif // SU3PROD_C