
#ifndef SU3V_C
#define SU3V_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "su3v.h"
#include "global.h"


void random_su3vec_field(su3_vec_field *su3vf)
{
    // size_t n = 6 * su3vf->volume;
    // double *d = su3vf->base;
    double *d = (double *)su3vf;

    for (size_t i = 0; i < 6*VOLUME; i++)
        d[i] = (double)rand() / (double)RAND_MAX;
}

void unit_su3vec_field(su3_vec_field *su3vf)
{
    // size_t n = 6 * su3vf->volume;
    // double *d = su3vf->base;
    double *d = (double *)su3vf;

    for (size_t i = 0; i < 6*VOLUME; i++)
        d[i] = 1.0;
}

void random_su3mat_field(su3_mat_field *su3mf)
{
    double *d = (double *)su3mf;

    for (size_t i = 0; i < 3*6*VOLUME; i++)
        d[i] = (double)rand() / (double)RAND_MAX;
}

void unit_su3mat_field(su3_mat_field *su3mf)
{
    double *d = (double *)su3mf;

    for (size_t i = 0; i < 3*6*VOLUME; i++)
        d[i] = 1.0;
}

// void complexv_init(complexv *x, size_t volume)
// {
//     x->volume = volume;
//     x->re = (double*)malloc(volume * sizeof(double));
//     x->im = (double*)malloc(volume * sizeof(double));
//     if (!x->re || !x->im) {
//         free(x->re);
//         free(x->im);
//         x->re = x->im = NULL;
//         x->volume = 0;
//         ERROR("Erorr allocating complexv");
//         abort();
//     }
// }
// 
// void complexv_free(complexv *x)
// {
//     free(x->re);
//     free(x->im);
//     x->re = x->im = NULL;
//     x->volume = 0;
// }
// 
// void su3_vec_field_init(su3_vec_field *v, size_t volume)
// {
//     v->volume = volume;
// 
//     v->base = (double*)malloc((size_t)6 * volume * sizeof(double));
//     if (!v->base) {
//         v->volume = 0;
//         ERROR("Erorr allocating su3_vec_field");
//         abort();
//     }
// 
//     v->c1re = v->base + 0*volume;
//     v->c1im = v->base + 1*volume;
//     v->c2re = v->base + 2*volume;
//     v->c2im = v->base + 3*volume;
//     v->c3re = v->base + 4*volume;
//     v->c3im = v->base + 5*volume;
// }
// 
// void su3_vec_field_free(su3_vec_field *v)
// {
//     free(v->base);
//     v->base = NULL;
//     v->c1re = v->c1im = v->c2re = v->c2im = v->c3re = v->c3im = NULL;
//     v->volume = 0;
// }


// void su3_mat_field_init(su3_mat_field *m, size_t volume)
// {
//     su3_vec_field_init(&m->c1, volume);
//     su3_vec_field_init(&m->c2, volume);
//     su3_vec_field_init(&m->c3, volume);
// }

// void su3_mat_field_free(su3_mat_field *m)
// {
//     su3_vec_field_free(&m->c3);
//     su3_vec_field_free(&m->c2);
//     su3_vec_field_free(&m->c1);
// }

#endif // SU3V_C
