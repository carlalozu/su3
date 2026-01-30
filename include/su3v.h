
#ifndef SU3V_H
#define SU3V_H

#include <stdlib.h>
#include <string.h>
#include "global.h"

typedef struct
{
    // size_t volume;
    double re[VOLUME_TRD];
    double im[VOLUME_TRD];
} complexv;

/* SU3 vector
* Each component is a pointer to an array of doubles
* Real and imaginary parts are stored separately
*/
typedef struct
{
    double c1re[VOLUME_TRD], c1im[VOLUME_TRD];
    double c2re[VOLUME_TRD], c2im[VOLUME_TRD];
    double c3re[VOLUME_TRD], c3im[VOLUME_TRD];
} su3_vec_field;

/* SU3 matrix 
* Each element is a su3_vec_vdble
* Represents a column of the SU3 matrix
*/
typedef struct
{
    double c11re[VOLUME_TRD], c11im[VOLUME_TRD];
    double c12re[VOLUME_TRD], c12im[VOLUME_TRD];
    double c13re[VOLUME_TRD], c13im[VOLUME_TRD];

    double c21re[VOLUME_TRD], c21im[VOLUME_TRD];
    double c22re[VOLUME_TRD], c22im[VOLUME_TRD];
    double c23re[VOLUME_TRD], c23im[VOLUME_TRD];

    double c31re[VOLUME_TRD], c31im[VOLUME_TRD];
    double c32re[VOLUME_TRD], c32im[VOLUME_TRD];
    double c33re[VOLUME_TRD], c33im[VOLUME_TRD];

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
