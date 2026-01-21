
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

int alloc_su3_cdble(su3_cdble **su3)
{
    // confirm size of su3_cdble is 18 * sizeof(double)
    _Static_assert(sizeof(su3_cdble) == 18 * sizeof(double),
                   "su3_cdble layout assumption broken");
    *su3 = malloc(sizeof **su3);

    if (*su3 == NULL)
    {
        return -1;
    }
    return 0;
}

void unit_su3_cdble(su3_cdble *su3, int value)
{

    _Static_assert(sizeof(su3_cdble) == 18 * sizeof(double),
                   "su3_cdble layout assumption broken");
    memset(su3, 0, sizeof *su3);
    su3->c11.re = (double)value;
    su3->c22.re = (double)value;
    su3->c33.re = (double)value;
}

complex_dble add(complex_dble a, complex_dble b)
{
    return (complex_dble){a.re + b.re, a.im + b.im};
}

/*
 * r.c1=s1.c1+s2.c1
 * r.c2=s1.c2+s2.c2
 * r.c3=s1.c3+s2.c3
 */
void vec_add(su3_vector_cdble *r, su3_vector_cdble *s1, su3_vector_cdble *s2)
{
    r->c1.re = s1->c1.re + s2->c1.re;
    r->c1.im = s1->c1.im + s2->c1.im;
    r->c2.re = s1->c2.re + s2->c2.re;
    r->c2.im = s1->c2.im + s2->c2.im;
    r->c3.re = s1->c3.re + s2->c3.re;
    r->c3.im = s1->c3.im + s2->c3.im;
}

/*
 * SU(3) matrix u times SU(3) vector s
 *
 * r.c1=(u*s).c1
 * r.c2=(u*s).c2
 * r.c3=(u*s).c3
 */
void mat_vec_mult(su3_vector_cdble *res, su3_cdble *u, su3_vector_cdble *s)
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

#endif // SU3PROD_C