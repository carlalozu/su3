
#ifndef SU3V_C
#define SU3V_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "su3v.h"
#include "global.h"


void random_su3vec_field(su3_vec_field *su3vf)
{
    size_t n = 6 * su3vf->volume;
    double *d = su3vf->base;

    for (size_t i = 0; i < n; i++)
        d[i] = (double)rand() / (double)RAND_MAX;
}

void unit_su3vec_field(su3_vec_field *su3vf)
{
    size_t n = 6 * su3vf->volume;
    double *d = su3vf->base;

    for (size_t i = 0; i < n; i++)
        d[i] = 1.0;
}

void random_su3mat_field(su3_mat_field *su3mf)
{
    random_su3vec_field(&su3mf->c1);
    random_su3vec_field(&su3mf->c2);
    random_su3vec_field(&su3mf->c3);
}

void unit_su3mat_field(su3_mat_field *su3mf)
{
    unit_su3vec_field(&su3mf->c1);
    unit_su3vec_field(&su3mf->c2);
    unit_su3vec_field(&su3mf->c3);
}

void complexv_init(complexv *x, size_t volume)
{
    size_t size = 2 * volume * sizeof(double);
    // Round up to nearest 8
    size_t padded_volume = (volume + 7) & ~7; 
    x->volume = padded_volume;
    x->base = (double*)aligned_alloc(ALIGN, size);
    if (!x->base) {
        x->volume = 0;
        fprintf(stderr, "Erorr allocating complexv");
        abort();
    }
    x->re = x->base + 0*padded_volume;
    x->im = x->base + 1*padded_volume;
}

void complexv_free(complexv *x)
{
    free(x->re);
    free(x->im);
    x->re = x->im = NULL;
    x->volume = 0;
}

void su3_vec_field_init(su3_vec_field *v, size_t volume)
{
    // Round up to nearest 8
    size_t padded_volume = (volume + 7) & ~7; 
    v->volume = padded_volume;
    size_t size = 6 * padded_volume * sizeof(double);
    v->base = (double*)aligned_alloc(ALIGN, size);
    if (!v->base) {
        v->volume = 0;
        fprintf(stderr, "Erorr allocating su3_vec_field");
        abort();
    }

    v->c1re = v->base + 0*padded_volume;
    v->c1im = v->base + 1*padded_volume;
    v->c2re = v->base + 2*padded_volume;
    v->c2im = v->base + 3*padded_volume;
    v->c3re = v->base + 4*padded_volume;
    v->c3im = v->base + 5*padded_volume;
}

void su3_vec_field_free(su3_vec_field *v)
{
    free(v->base);
    v->base = NULL;
    v->c1re = v->c1im = v->c2re = v->c2im = v->c3re = v->c3im = NULL;
    v->volume = 0;
}


void su3_mat_field_init(su3_mat_field *m, size_t volume)
{
    su3_vec_field_init(&m->c1, volume);
    su3_vec_field_init(&m->c2, volume);
    su3_vec_field_init(&m->c3, volume);
}

void su3_mat_field_free(su3_mat_field *m)
{
    su3_vec_field_free(&m->c3);
    su3_vec_field_free(&m->c2);
    su3_vec_field_free(&m->c1);
}

#pragma omp declare target
void complex_field_map_pointers(complexv *v)
{
    size_t volume = v->volume;
    v->re = v->base + 0*volume;
    v->im = v->base + 1*volume;
}
#pragma omp end declare target

#pragma omp declare target
void su3_vec_field_map_pointers(su3_vec_field *v)
{
    size_t volume = v->volume;
    v->c1re = v->base + 0*volume;
    v->c1im = v->base + 1*volume;
    v->c2re = v->base + 2*volume;
    v->c2im = v->base + 3*volume;
    v->c3re = v->base + 4*volume;
    v->c3im = v->base + 5*volume;
}
#pragma omp end declare target

#pragma omp declare target
void su3_mat_field_map_pointers(su3_mat_field *m)
{
    size_t volume = m->c1.volume;
    m->c1.c1re = m->c1.base + 0*volume;
    m->c1.c1im = m->c1.base + 1*volume;
    m->c1.c2re = m->c1.base + 2*volume;
    m->c1.c2im = m->c1.base + 3*volume;
    m->c1.c3re = m->c1.base + 4*volume;
    m->c1.c3im = m->c1.base + 5*volume;

    m->c2.c1re = m->c2.base + 0*volume;
    m->c2.c1im = m->c2.base + 1*volume;
    m->c2.c2re = m->c2.base + 2*volume;
    m->c2.c2im = m->c2.base + 3*volume;
    m->c2.c3re = m->c2.base + 4*volume;
    m->c2.c3im = m->c2.base + 5*volume;

    m->c3.c1re = m->c3.base + 0*volume;
    m->c3.c1im = m->c3.base + 1*volume;
    m->c3.c2re = m->c3.base + 2*volume;
    m->c3.c2im = m->c3.base + 3*volume;
    m->c3.c3re = m->c3.base + 4*volume;
    m->c3.c3im = m->c3.base + 5*volume;
}
#pragma omp end declare target

#endif // SU3V_C
