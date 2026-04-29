
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

#include "su3prod.h"

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
void vec_add(su3_vec_c *res, const su3_vec_c *s1, const su3_vec_c *s2)
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
void su3matxsu3vec(su3_vec_c *res, const su3_mat_c *u, const su3_vec_c *s)
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
void su3matdagxsu3vec(su3_vec_c *r, const su3_mat_c *u, const su3_vec_c *s)
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

/* SU(3) trace
 *
 * tr = trace(u)
 */
complex su3mat_trace(const su3_mat_c *u)
{
    complex tr;
    tr.re = u->c11.re + u->c22.re + u->c33.re;
    tr.im = u->c11.im + u->c22.im + u->c33.im;
    return tr;
}

#endif // SU3PROD_C
