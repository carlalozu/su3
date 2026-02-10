
/*******************************************************************************
 *
 * File su3prod.c
 *
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef SU3PROD_C
#define SU3PROD_C

#include "su3.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void unit_su3mat(su3_mat *su3)
{
    _Static_assert(sizeof(su3_mat) == 18 * sizeof(float),
                   "su3 layout assumption broken");
    float *d = (float *)su3;
    for (int i = 0; i < 18; i++)
        d[i] = 1.0;
}

void random_su3mat(su3_mat *su3)
{
    _Static_assert(sizeof(su3_mat) == 18 * sizeof(float),
                   "su3 layout assumption broken");
    float *d = (float *)su3;
    for (int i = 0; i < 18; i++)
        d[i] = (float)rand() / RAND_MAX;
}

void unit_su3vec(su3_vec *vec)
{
    _Static_assert(sizeof(su3_vec) == 6 * sizeof(float),
                   "su3_vec layout assumption broken");
    float *d = (float *)vec;
    for (int i = 0; i < 6; i++)
        d[i] = 1.0;
}

complex add(const complex a, const complex b)
{
    return (complex){a.re + b.re, a.im + b.im};
}

/*
 * SU(3) vector addition
 *
 * r.c1=s1.c1+s2.c1
 * r.c2=s1.c2+s2.c2
 * r.c3=s1.c3+s2.c3
 */
void vec_add(su3_vec *res, const su3_vec *s1, const su3_vec *s2)
{
    res->c1.re = s1->c1.re + s2->c1.re;
    res->c1.im = s1->c1.im + s2->c1.im;
    res->c2.re = s1->c2.re + s2->c2.re;
    res->c2.im = s1->c2.im + s2->c2.im;
    res->c3.re = s1->c3.re + s2->c3.re;
    res->c3.im = s1->c3.im + s2->c3.im;
}

/*
 * SU(3) matrix u times SU(3) vector s
 *
 * r.c1=(u*s).c1
 * r.c2=(u*s).c2
 * r.c3=(u*s).c3
 */
void su3matxsu3vec(su3_vec *res, const su3_mat *u, const su3_vec *s)
{
    res->c1.re = u->c11.re * s->c1.re - u->c11.im * s->c1.im +
                 u->c12.re * s->c2.re - u->c12.im * s->c2.im +
                 u->c13.re * s->c3.re - u->c13.im * s->c3.im;
    res->c1.im = u->c11.re * s->c1.im + u->c11.im * s->c1.re +
                 u->c12.re * s->c2.im + u->c12.im * s->c2.re +
                 u->c13.re * s->c3.im + u->c13.im * s->c3.re;
    res->c2.re = u->c21.re * s->c1.re - u->c21.im * s->c1.im +
                 u->c22.re * s->c2.re - u->c22.im * s->c2.im +
                 u->c23.re * s->c3.re - u->c23.im * s->c3.im;
    res->c2.im = u->c21.re * s->c1.im + u->c21.im * s->c1.re +
                 u->c22.re * s->c2.im + u->c22.im * s->c2.re +
                 u->c23.re * s->c3.im + u->c23.im * s->c3.re;
    res->c3.re = u->c31.re * s->c1.re - u->c31.im * s->c1.im +
                 u->c32.re * s->c2.re - u->c32.im * s->c2.im +
                 u->c33.re * s->c3.re - u->c33.im * s->c3.im;
    res->c3.im = u->c31.re * s->c1.im + u->c31.im * s->c1.re +
                 u->c32.re * s->c2.im + u->c32.im * s->c2.re +
                 u->c33.re * s->c3.im + u->c33.im * s->c3.re;
}

/*
 * SU(3) matrix u^dagger times SU(3) vector s
 *
 * r.c1=(u^dagger*s).c1
 * r.c2=(u^dagger*s).c2
 * r.c3=(u^dagger*s).c3
 */
void su3matdagxsu3vec(su3_vec *r, const su3_mat *u, const su3_vec *s)
{
    (*r).c1.re = (*u).c11.re * (*s).c1.re + (*u).c11.im * (*s).c1.im +
                 (*u).c21.re * (*s).c2.re + (*u).c21.im * (*s).c2.im +
                 (*u).c31.re * (*s).c3.re + (*u).c31.im * (*s).c3.im;
    (*r).c1.im = (*u).c11.re * (*s).c1.im - (*u).c11.im * (*s).c1.re +
                 (*u).c21.re * (*s).c2.im - (*u).c21.im * (*s).c2.re +
                 (*u).c31.re * (*s).c3.im - (*u).c31.im * (*s).c3.re;
    (*r).c2.re = (*u).c12.re * (*s).c1.re + (*u).c12.im * (*s).c1.im +
                 (*u).c22.re * (*s).c2.re + (*u).c22.im * (*s).c2.im +
                 (*u).c32.re * (*s).c3.re + (*u).c32.im * (*s).c3.im;
    (*r).c2.im = (*u).c12.re * (*s).c1.im - (*u).c12.im * (*s).c1.re +
                 (*u).c22.re * (*s).c2.im - (*u).c22.im * (*s).c2.re +
                 (*u).c32.re * (*s).c3.im - (*u).c32.im * (*s).c3.re;
    (*r).c3.re = (*u).c13.re * (*s).c1.re + (*u).c13.im * (*s).c1.im +
                 (*u).c23.re * (*s).c2.re + (*u).c23.im * (*s).c2.im +
                 (*u).c33.re * (*s).c3.re + (*u).c33.im * (*s).c3.im;
    (*r).c3.im = (*u).c13.re * (*s).c1.im - (*u).c13.im * (*s).c1.re +
                 (*u).c23.re * (*s).c2.im - (*u).c23.im * (*s).c2.re +
                 (*u).c33.re * (*s).c3.im - (*u).c33.im * (*s).c3.re;
}

/*
 * SU(3) matrix multiplication
 *
 * res = u * v
 */
void su3matxsu3mat(su3_mat *res, const su3_mat *u, const su3_mat *v)
{
    // --- Column 1 ---
    res->c11.re = u->c11.re * v->c11.re - u->c11.im * v->c11.im +
                  u->c12.re * v->c21.re - u->c12.im * v->c21.im +
                  u->c13.re * v->c31.re - u->c13.im * v->c31.im;
    res->c11.im = u->c11.re * v->c11.im + u->c11.im * v->c11.re +
                  u->c12.re * v->c21.im + u->c12.im * v->c21.re +
                  u->c13.re * v->c31.im + u->c13.im * v->c31.re;

    res->c21.re = u->c21.re * v->c11.re - u->c21.im * v->c11.im +
                  u->c22.re * v->c21.re - u->c22.im * v->c21.im +
                  u->c23.re * v->c31.re - u->c23.im * v->c31.im;
    res->c21.im = u->c21.re * v->c11.im + u->c21.im * v->c11.re +
                  u->c22.re * v->c21.im + u->c22.im * v->c21.re +
                  u->c23.re * v->c31.im + u->c23.im * v->c31.re;

    res->c31.re = u->c31.re * v->c11.re - u->c31.im * v->c11.im +
                  u->c32.re * v->c21.re - u->c32.im * v->c21.im +
                  u->c33.re * v->c31.re - u->c33.im * v->c31.im;
    res->c31.im = u->c31.re * v->c11.im + u->c31.im * v->c11.re +
                  u->c32.re * v->c21.im + u->c32.im * v->c21.re +
                  u->c33.re * v->c31.im + u->c33.im * v->c31.re;

    // --- Column 2 ---
    res->c12.re = u->c11.re * v->c12.re - u->c11.im * v->c12.im +
                  u->c12.re * v->c22.re - u->c12.im * v->c22.im +
                  u->c13.re * v->c32.re - u->c13.im * v->c32.im;
    res->c12.im = u->c11.re * v->c12.im + u->c11.im * v->c12.re +
                  u->c12.re * v->c22.im + u->c12.im * v->c22.re +
                  u->c13.re * v->c32.im + u->c13.im * v->c32.re;

    res->c22.re = u->c21.re * v->c12.re - u->c21.im * v->c12.im +
                  u->c22.re * v->c22.re - u->c22.im * v->c22.im +
                  u->c23.re * v->c32.re - u->c23.im * v->c32.im;
    res->c22.im = u->c21.re * v->c12.im + u->c21.im * v->c12.re +
                  u->c22.re * v->c22.im + u->c22.im * v->c22.re +
                  u->c23.re * v->c32.im + u->c23.im * v->c32.re;

    res->c32.re = u->c31.re * v->c12.re - u->c31.im * v->c12.im +
                  u->c32.re * v->c22.re - u->c32.im * v->c22.im +
                  u->c33.re * v->c32.re - u->c33.im * v->c32.im;
    res->c32.im = u->c31.re * v->c12.im + u->c31.im * v->c12.re +
                  u->c32.re * v->c22.im + u->c32.im * v->c22.re +
                  u->c33.re * v->c32.im + u->c33.im * v->c32.re;

    // --- Column 3 ---
    res->c13.re = u->c11.re * v->c13.re - u->c11.im * v->c13.im +
                  u->c12.re * v->c23.re - u->c12.im * v->c23.im +
                  u->c13.re * v->c33.re - u->c13.im * v->c33.im;
    res->c13.im = u->c11.re * v->c13.im + u->c11.im * v->c13.re +
                  u->c12.re * v->c23.im + u->c12.im * v->c23.re +
                  u->c13.re * v->c33.im + u->c13.im * v->c33.re;

    res->c23.re = u->c21.re * v->c13.re - u->c21.im * v->c13.im +
                  u->c22.re * v->c23.re - u->c22.im * v->c23.im +
                  u->c23.re * v->c33.re - u->c23.im * v->c33.im;
    res->c23.im = u->c21.re * v->c13.im + u->c21.im * v->c13.re +
                  u->c22.re * v->c23.im + u->c22.im * v->c23.re +
                  u->c23.re * v->c33.im + u->c23.im * v->c33.re;

    res->c33.re = u->c31.re * v->c13.re - u->c31.im * v->c13.im +
                  u->c32.re * v->c23.re - u->c32.im * v->c23.im +
                  u->c33.re * v->c33.re - u->c33.im * v->c33.im;
    res->c33.im = u->c31.re * v->c13.im + u->c31.im * v->c13.re +
                  u->c32.re * v->c23.im + u->c32.im * v->c23.re +
                  u->c33.re * v->c33.im + u->c33.im * v->c33.re;
}

/*
 * Computes w=u^dag*v^dag assuming that w is different from u and v.
 */
void su3matdagxsu3matdag(su3_mat *w, const su3_mat *u, const su3_mat *v)
{
    w->c11.re = u->c11.re * v->c11.re + u->c11.im * -v->c11.im +
                u->c21.re * v->c12.re + u->c21.im * -v->c12.im +
                u->c31.re * v->c13.re + u->c31.im * -v->c13.im;
    w->c11.im = u->c11.re * -v->c11.im - u->c11.im * v->c11.re +
                u->c21.re * -v->c12.im - u->c21.im * v->c12.re +
                u->c31.re * -v->c13.im - u->c31.im * v->c13.re;
    w->c21.re = u->c12.re * v->c11.re + u->c12.im * -v->c11.im +
                u->c22.re * v->c12.re + u->c22.im * -v->c12.im +
                u->c32.re * v->c13.re + u->c32.im * -v->c13.im;
    w->c21.im = u->c12.re * -v->c11.im - u->c12.im * v->c11.re +
                u->c22.re * -v->c12.im - u->c22.im * v->c12.re +
                u->c32.re * -v->c13.im - u->c32.im * v->c13.re;
    w->c31.re = u->c13.re * v->c11.re + u->c13.im * -v->c11.im +
                u->c23.re * v->c12.re + u->c23.im * -v->c12.im +
                u->c33.re * v->c13.re + u->c33.im * -v->c13.im;
    w->c31.im = u->c13.re * -v->c11.im - u->c13.im * v->c11.re +
                u->c23.re * -v->c12.im - u->c23.im * v->c12.re +
                u->c33.re * -v->c13.im - u->c33.im * v->c13.re;

    w->c12.re = u->c11.re * v->c21.re + u->c11.im * -v->c21.im +
                u->c21.re * v->c22.re + u->c21.im * -v->c22.im +
                u->c31.re * v->c23.re + u->c31.im * -v->c23.im;
    w->c12.im = u->c11.re * -v->c21.im - u->c11.im * v->c21.re +
                u->c21.re * -v->c22.im - u->c21.im * v->c22.re +
                u->c31.re * -v->c23.im - u->c31.im * v->c23.re;
    w->c22.re = u->c12.re * v->c21.re + u->c12.im * -v->c21.im +
                u->c22.re * v->c22.re + u->c22.im * -v->c22.im +
                u->c32.re * v->c23.re + u->c32.im * -v->c23.im;
    w->c22.im = u->c12.re * -v->c21.im - u->c12.im * v->c21.re +
                u->c22.re * -v->c22.im - u->c22.im * v->c22.re +
                u->c32.re * -v->c23.im - u->c32.im * v->c23.re;
    w->c32.re = u->c13.re * v->c21.re + u->c13.im * -v->c21.im +
                u->c23.re * v->c22.re + u->c23.im * -v->c22.im +
                u->c33.re * v->c23.re + u->c33.im * -v->c23.im;
    w->c32.im = u->c13.re * -v->c21.im - u->c13.im * v->c21.re +
                u->c23.re * -v->c22.im - u->c23.im * v->c22.re +
                u->c33.re * -v->c23.im - u->c33.im * v->c23.re;

    w->c13.re = u->c11.re * v->c31.re + u->c11.im * -v->c31.im +
                u->c21.re * v->c32.re + u->c21.im * -v->c32.im +
                u->c31.re * v->c33.re + u->c31.im * -v->c33.im;
    w->c13.im = u->c11.re * -v->c31.im - u->c11.im * v->c31.re +
                u->c21.re * -v->c32.im - u->c21.im * v->c32.re +
                u->c31.re * -v->c33.im - u->c31.im * v->c33.re;
    w->c23.re = u->c12.re * v->c31.re + u->c12.im * -v->c31.im +
                u->c22.re * v->c32.re + u->c22.im * -v->c32.im +
                u->c32.re * v->c33.re + u->c32.im * -v->c33.im;
    w->c23.im = u->c12.re * -v->c31.im - u->c12.im * v->c31.re +
                u->c22.re * -v->c32.im - u->c22.im * v->c32.re +
                u->c32.re * -v->c33.im - u->c32.im * v->c33.re;
    w->c33.re = u->c13.re * v->c31.re + u->c13.im * -v->c31.im +
                u->c23.re * v->c32.re + u->c23.im * -v->c32.im +
                u->c33.re * v->c33.re + u->c33.im * -v->c33.im;
    w->c33.im = u->c13.re * -v->c31.im - u->c13.im * v->c31.re +
                u->c23.re * -v->c32.im - u->c23.im * v->c32.re +
                u->c33.re * -v->c33.im - u->c33.im * v->c33.re;
}

/* SU(3) trace
 *
 * tr = trace(u)
 */
complex su3mat_trace(const su3_mat *u)
{
    complex tr;
    tr.re = u->c11.re + u->c22.re + u->c33.re;
    tr.im = u->c11.im + u->c22.im + u->c33.im;
    return tr;
}

/* SU(3)xSU(3) real trace
 *
 * tr = retrace(uv)
 */
float su3matxsu3mat_retrace(const su3_mat *u, const su3_mat *v)
{
    float tr_1 = 0.0;
    float tr_2 = 0.0;
    float tr_3 = 0.0;

    tr_1 += (*u).c11.re * (*v).c11.re - (*u).c11.im * (*v).c11.im;
    tr_1 += (*u).c12.re * (*v).c21.re - (*u).c12.im * (*v).c21.im;
    tr_1 += (*u).c13.re * (*v).c31.re - (*u).c13.im * (*v).c31.im;

    tr_2 += (*u).c21.re * (*v).c12.re - (*u).c21.im * (*v).c12.im;
    tr_2 += (*u).c22.re * (*v).c22.re - (*u).c22.im * (*v).c22.im;
    tr_2 += (*u).c23.re * (*v).c32.re - (*u).c23.im * (*v).c32.im;

    tr_3 += (*u).c31.re * (*v).c13.re - (*u).c31.im * (*v).c13.im;
    tr_3 += (*u).c32.re * (*v).c23.re - (*u).c32.im * (*v).c23.im;
    tr_3 += (*u).c33.re * (*v).c33.re - (*u).c33.im * (*v).c33.im;

    return tr_1 + tr_2 + tr_3;
}

#endif // SU3PROD_C
