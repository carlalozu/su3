
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

// typedef struct
// {
//     double *c11r, *c12r, *c13r, *c21r, *c22r, *c23r, *c31r, *c32r, *c33r;
//     double *c11i, *c12i, *c13i, *c21i, *c22i, *c23i, *c31i, *c32i, *c33i;
// } su3_dble_field;

complex_dble add(complex_dble a, complex_dble b);

#endif // SU3_H
