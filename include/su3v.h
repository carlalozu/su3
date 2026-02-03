
#ifndef SU3V_H
#define SU3V_H

#include <stdlib.h>
#include <string.h>
#include "global.h"

typedef struct
{
    size_t volume;
    double *re;
    double *im;
} complexv;

/* SU3 vector
* Each component is a pointer to an array of doubles
* Real and imaginary parts are stored separately
*/
typedef struct
{
    size_t volume;
    double *base;
    double *c1re, *c1im;
    double *c2re, *c2im;
    double *c3re, *c3im;
} su3_vec_field;

/* SU3 matrix 
* Each element is a su3_vec_vdble
* Represents a column of the SU3 matrix
*/
typedef struct
{
    su3_vec_field c1, c2, c3;

} su3_mat_field;

void complexv_init(complexv *x, size_t volume);
void complexv_free(complexv *x);

void su3_vec_field_init(su3_vec_field *v, size_t volume);
void su3_vec_field_free(su3_vec_field *v);

void su3_mat_field_init(su3_mat_field *m, size_t volume);
void su3_mat_field_free(su3_mat_field *m);

#pragma omp declare target
void su3_vec_field_map_pointers(su3_vec_field *v);
void su3_mat_field_map_pointers(su3_mat_field *m);
#pragma omp end declare target

void random_su3mat_field(su3_mat_field *su3_mat);
void random_su3vec_field(su3_vec_field *su3_vec);
void unit_su3mat_field(su3_mat_field *su3_mat);

#endif // SU3V_H
