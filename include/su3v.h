
#ifndef SU3V_H
#define SU3V_H

#include <stdlib.h>
#include <string.h>
#include "global.h"

typedef struct
{
    // size_t volume;
    double re[VOLUME];
    double im[VOLUME];
} complexv;

/* SU3 vector
* Each component is a pointer to an array of doubles
* Real and imaginary parts are stored separately
*/
typedef struct
{
    double c1re[VOLUME], c1im[VOLUME];
    double c2re[VOLUME], c2im[VOLUME];
    double c3re[VOLUME], c3im[VOLUME];
} su3_vec_field;

/* SU3 matrix 
* Each element is a su3_vec_vdble
* Represents a column of the SU3 matrix
*/
typedef struct
{
    double c11re[VOLUME], c11im[VOLUME];
    double c12re[VOLUME], c12im[VOLUME];
    double c13re[VOLUME], c13im[VOLUME];

    double c21re[VOLUME], c21im[VOLUME];
    double c22re[VOLUME], c22im[VOLUME];
    double c23re[VOLUME], c23im[VOLUME];

    double c31re[VOLUME], c31im[VOLUME];
    double c32re[VOLUME], c32im[VOLUME];
    double c33re[VOLUME], c33im[VOLUME];

} su3_mat_field;


#pragma omp declare target
// void complexv_init(complexv *x, size_t volume);
// void complexv_free(complexv *x);
// void su3_vec_field_init(su3_vec_field *u, size_t volume);
// void su3_vec_field_free(su3_vec_field *u);
// void su3_mat_field_init(su3_mat_field *u, size_t volume);
// void su3_mat_field_free(su3_mat_field *u);
void unit_su3mat_field(su3_mat_field *su3_mat);
void random_su3mat_field(su3_mat_field *su3_mat);
void random_su3vec_field(su3_vec_field *su3_vec);
#pragma omp end declare target

#endif // SU3V_H
