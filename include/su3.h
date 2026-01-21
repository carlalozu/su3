
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
   complex_dble c1,c2,c3;
} su3_vector_cdble;

typedef struct
{
    complex_dble c11, c12, c13, c21, c22, c23, c31, c32, c33;
} su3_cdble;


// SU3 initialization and allocation
int alloc_su3_cdble(su3_cdble **su3);
void unit_su3_cdble(su3_cdble *su3, int value);

int alloc_su3_vector_cdble(su3_vector_cdble **vec);
void unit_su3_vector_cdble(su3_vector_cdble *vec, int value);

// Algebra
complex_dble add(complex_dble a, complex_dble b);
void vec_add(su3_vector_cdble *res, su3_vector_cdble *a, su3_vector_cdble *b);
void mat_vec_mult(su3_vector_cdble *res, su3_cdble *mat, su3_vector_cdble *vec);


#endif // SU3_H
