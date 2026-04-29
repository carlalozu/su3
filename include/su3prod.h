
#ifndef SU3PROD_H
#define SU3PROD_H

#include "su3.h"

#if defined(KOKKOS_CORE_HPP)
  #define DEVICE_KEYWORD    KOKKOS_INLINE_FUNCTION
  #define PRAGMA_OMP_BEGIN
  #define PRAGMA_OMP_END
#elif defined(__CUDACC__)
  #define DEVICE_KEYWORD    __device__ static inline
  #define PRAGMA_OMP_BEGIN
  #define PRAGMA_OMP_END
#else
  #define DEVICE_KEYWORD static inline
  #define PRAGMA_OMP_BEGIN _Pragma("omp declare target")
  #define PRAGMA_OMP_END   _Pragma("omp end declare target")
#endif

#ifdef __cplusplus
extern "C" {
#endif

complex add(const complex a, const complex b);
void vec_add(su3_vec_c *res, const su3_vec_c *u, const su3_vec_c *v);

PRAGMA_OMP_BEGIN
complex su3mat_trace(const su3_mat_c *u);
void su3matxsu3vec(su3_vec_c *res, const su3_mat_c *u, const su3_vec_c *v);
void su3matdagxsu3vec(su3_vec_c *r, const su3_mat_c *u, const su3_vec_c *s);
PRAGMA_OMP_END

#ifdef __cplusplus
}
#endif


// ---------------------------------------------------------------------------
// Matrix operations — inline on all backends
// ---------------------------------------------------------------------------

PRAGMA_OMP_BEGIN
DEVICE_KEYWORD double su3matxsu3mat_retrace(const su3_mat_c *u, const su3_mat_c *v)
{
    double tr_1 = u->c11.re * v->c11.re - u->c11.im * v->c11.im
                + u->c12.re * v->c21.re - u->c12.im * v->c21.im
                + u->c13.re * v->c31.re - u->c13.im * v->c31.im;
    double tr_2 = u->c21.re * v->c12.re - u->c21.im * v->c12.im
                + u->c22.re * v->c22.re - u->c22.im * v->c22.im
                + u->c23.re * v->c32.re - u->c23.im * v->c32.im;
    double tr_3 = u->c31.re * v->c13.re - u->c31.im * v->c13.im
                + u->c32.re * v->c23.re - u->c32.im * v->c23.im
                + u->c33.re * v->c33.re - u->c33.im * v->c33.im;
    return tr_1 + tr_2 + tr_3;
}
PRAGMA_OMP_END

PRAGMA_OMP_BEGIN
DEVICE_KEYWORD void su3matxsu3mat(su3_mat_c *res, const su3_mat_c *u, const su3_mat_c *v)
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
PRAGMA_OMP_END

PRAGMA_OMP_BEGIN
DEVICE_KEYWORD void su3matdagxsu3matdag(su3_mat_c *w, const su3_mat_c *u, const su3_mat_c *v)
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
PRAGMA_OMP_END

#endif // SU3PROD_H
