
#ifndef SU3_H
#define SU3_H

#include <stdlib.h>
#include <string.h>

typedef struct
{
    double re, im;
} complex_dble;

typedef struct
{
    complex_dble c11, c12, c13, c21, c22, c23, c31, c32, c33;
} su3_cdble;

complex_dble add(complex_dble a, complex_dble b);
int alloc_su3_cdble(su3_cdble **su3);
void unit_su3_cdble(su3_cdble *su3, int value);


#endif // SU3_H
