
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
#include "su3v.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void unit_su3mat(su3_mat *su3)
{
    _Static_assert(sizeof(su3_mat) == 18 * sizeof(double),
                   "su3 layout assumption broken");
    double *d = (double *)su3;
    for (int i = 0; i < 18; i++)
        d[i] = 1.0;
}

void random_su3mat(su3_mat *su3)
{
    _Static_assert(sizeof(su3_mat) == 18 * sizeof(double),
                   "su3 layout assumption broken");
    double *d = (double *)su3;
    for (int i = 0; i < 18; i++)
        d[i] = (double)rand() / RAND_MAX;
}

void unit_su3vec(su3_vec *vec)
{
    _Static_assert(sizeof(su3_vec) == 6 * sizeof(double),
                   "su3_vec layout assumption broken");
    double *d = (double *)vec;
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
 * SU(3) matrix multiplication
 *
 * res = u * v
 */
void su3matxsu3mat(su3_mat *res, const su3_mat *u, const su3_mat *v)
{
    su3_vec psi, chi;

    psi.c1 = v->c11;
    psi.c2 = v->c21;
    psi.c3 = v->c31;
    su3matxsu3vec(&chi, u, &psi);
    res->c11 = chi.c1;
    res->c21 = chi.c2;
    res->c31 = chi.c3;

    psi.c1 = v->c12;
    psi.c2 = v->c22;
    psi.c3 = v->c32;
    su3matxsu3vec(&chi, u, &psi);
    res->c12 = chi.c1;
    res->c22 = chi.c2;
    res->c32 = chi.c3;

    psi.c1 = v->c13;
    psi.c2 = v->c23;
    psi.c3 = v->c33;
    su3matxsu3vec(&chi, u, &psi);
    res->c13 = chi.c1;
    res->c23 = chi.c2;
    res->c33 = chi.c3;
}

/* SU(3) trace
 *
 * tr = trace(u)
 */
inline complex su3_trace(const su3_mat *u)
{
    complex tr;
    tr.re = u->c11.re + u->c22.re + u->c33.re;
    tr.im = u->c11.im + u->c22.im + u->c33.im;
    return tr;
}
#endif // SU3PROD_C
