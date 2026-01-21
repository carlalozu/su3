
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

int alloc_su3_vector_cdble(su3_vector_cdble **su3_vec)
{
    // confirm size of su3_cdble is 18 * sizeof(double)
    _Static_assert(sizeof(su3_vector_cdble) == 6 * sizeof(double),
                   "su3_vector_cdble layout assumption broken");
    *su3_vec = malloc(sizeof **su3_vec);

    if (*su3_vec == NULL)
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

void unit_su3_vector_cdble(su3_vector_cdble *vec, int value)
{
    
    _Static_assert(sizeof(su3_vector_cdble) == 6 * sizeof(double),
                   "su3_vector_cdble layout assumption broken");
    memset(vec, 0, sizeof *vec);
    vec->c1.re = (double)value;
    vec->c2.re = (double)value;
    vec->c3.re = (double)value;
}


complex_dble add(complex_dble a, complex_dble b)
{
    return (complex_dble){a.re + b.re, a.im + b.im};
}

/*
 * SU(3) vector addition
 *
 * r.c1=s1.c1+s2.c1
 * r.c2=s1.c2+s2.c2
 * r.c3=s1.c3+s2.c3
 */
void vec_add(su3_vector_cdble *res, su3_vector_cdble *s1, su3_vector_cdble *s2)
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
void su3xsu3vec(su3_vector_cdble *res, su3_cdble *u, su3_vector_cdble *s)
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

void su3xsu3(su3_cdble *res, su3_cdble *u,su3_cdble *v)
{
   su3_vector_cdble psi,chi;

   psi.c1=(*v).c11;
   psi.c2=(*v).c21;
   psi.c3=(*v).c31;
   su3xsu3vec(&chi,u,&psi);
   (*res).c11=chi.c1;
   (*res).c21=chi.c2;
   (*res).c31=chi.c3;

   psi.c1=(*v).c12;
   psi.c2=(*v).c22;
   psi.c3=(*v).c32;
   su3xsu3vec(&chi,u,&psi);
   (*res).c12=chi.c1;
   (*res).c22=chi.c2;
   (*res).c32=chi.c3;

   psi.c1=(*v).c13;
   psi.c2=(*v).c23;
   psi.c3=(*v).c33;
   su3xsu3vec(&chi,u,&psi);
   (*res).c13=chi.c1;
   (*res).c23=chi.c2;
   (*res).c33=chi.c3;
}

#endif // SU3PROD_C