
#ifndef SU3V_C
#define SU3V_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "su3v.h"
#include "global.h"


void random_su3vec_field(su3_vec_field *su3vf, const size_t volume)
{
    double *d = (double *)su3vf;
    for (size_t i = 0; i < 6*volume; i++)
        d[i] = (double)rand() / RAND_MAX;
}

void random_su3mat_field(su3_mat_field *su3mf, const size_t volume)
{
    double *d = (double *)su3mf;
    for (size_t i = 0; i < 18*volume; i++)
        d[i] = (double)rand() / RAND_MAX;
}

void unit_su3mat_field(su3_mat_field *su3mf, const size_t volume)
{
    double *d = (double *)su3mf;
    #pragma omp for schedule(static)
    for (size_t i = 0; i < 18*volume; i++)
        d[i] = 1.0;
}

void complexv_init(complexv *x, size_t volume)
{
    x->volume = volume;
    x->re = (double*)malloc(volume * sizeof(double));
    x->im = (double*)malloc(volume * sizeof(double));
    if (!x->re || !x->im) {
        free(x->re);
        free(x->im);
        x->re = x->im = NULL;
        x->volume = 0;
        fprintf(stderr, "Erorr allocating complexv");
        abort();
    }
}

void complexv_free(complexv *x)
{
    free(x->re);
    free(x->im);
    x->re = x->im = NULL;
    x->volume = 0;
}

#endif // SU3V_C