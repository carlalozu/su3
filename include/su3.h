#ifndef SU3_H
#define SU3_H

typedef struct
{
    double re, im;
} complex_dble;

typedef struct
{
    complex_dble c11, c12, c13, c21, c22, c23, c31, c32, c33;
} su3_dble;

typedef struct
{
    complex_dble *c11, *c12, *c13, *c21, *c22, *c23, *c31, *c32, *c33;
} su3_dble_field;

#endif // SU3_H