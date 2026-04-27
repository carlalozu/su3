#ifndef SU3V_CUDA_CUH
#define SU3V_CUDA_CUH

#include <stdio.h>
#include <stdlib.h>
#include "su3v.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Single-site SU(3) vector stored entirely in registers.
// Mirrors su3_vec_field at one index i.
typedef struct {
    double c1re, c1im;
    double c2re, c2im;
    double c3re, c3im;
} su3_vec_reg;

// Single-site SU(3) matrix stored entirely in registers.
// Mirrors su3_mat_field at one index i.
typedef struct {
    su3_vec_reg c1, c2, c3;
} su3_mat_reg;

// ---------------------------------------------------------------------------
// res = u[i] * v[i]  (register output, SoA inputs)
// ---------------------------------------------------------------------------
__device__ static inline void fsu3matxsu3mat_reg(
    su3_mat_reg *__restrict__ res,
    const su3_mat_field *__restrict__ u,
    const su3_mat_field *__restrict__ v,
    size_t i)
{
    // column c1
    res->c1.c1re = u->c1.c1re[i] * v->c1.c1re[i] - u->c1.c1im[i] * v->c1.c1im[i]
                 + u->c1.c2re[i] * v->c1.c2re[i] - u->c1.c2im[i] * v->c1.c2im[i]
                 + u->c1.c3re[i] * v->c1.c3re[i] - u->c1.c3im[i] * v->c1.c3im[i];
    res->c1.c1im = u->c1.c1re[i] * v->c1.c1im[i] + u->c1.c1im[i] * v->c1.c1re[i]
                 + u->c1.c2re[i] * v->c1.c2im[i] + u->c1.c2im[i] * v->c1.c2re[i]
                 + u->c1.c3re[i] * v->c1.c3im[i] + u->c1.c3im[i] * v->c1.c3re[i];
    res->c1.c2re = u->c2.c1re[i] * v->c1.c1re[i] - u->c2.c1im[i] * v->c1.c1im[i]
                 + u->c2.c2re[i] * v->c1.c2re[i] - u->c2.c2im[i] * v->c1.c2im[i]
                 + u->c2.c3re[i] * v->c1.c3re[i] - u->c2.c3im[i] * v->c1.c3im[i];
    res->c1.c2im = u->c2.c1re[i] * v->c1.c1im[i] + u->c2.c1im[i] * v->c1.c1re[i]
                 + u->c2.c2re[i] * v->c1.c2im[i] + u->c2.c2im[i] * v->c1.c2re[i]
                 + u->c2.c3re[i] * v->c1.c3im[i] + u->c2.c3im[i] * v->c1.c3re[i];
    res->c1.c3re = u->c3.c1re[i] * v->c1.c1re[i] - u->c3.c1im[i] * v->c1.c1im[i]
                 + u->c3.c2re[i] * v->c1.c2re[i] - u->c3.c2im[i] * v->c1.c2im[i]
                 + u->c3.c3re[i] * v->c1.c3re[i] - u->c3.c3im[i] * v->c1.c3im[i];
    res->c1.c3im = u->c3.c1re[i] * v->c1.c1im[i] + u->c3.c1im[i] * v->c1.c1re[i]
                 + u->c3.c2re[i] * v->c1.c2im[i] + u->c3.c2im[i] * v->c1.c2re[i]
                 + u->c3.c3re[i] * v->c1.c3im[i] + u->c3.c3im[i] * v->c1.c3re[i];

    // column c2
    res->c2.c1re = u->c1.c1re[i] * v->c2.c1re[i] - u->c1.c1im[i] * v->c2.c1im[i]
                 + u->c1.c2re[i] * v->c2.c2re[i] - u->c1.c2im[i] * v->c2.c2im[i]
                 + u->c1.c3re[i] * v->c2.c3re[i] - u->c1.c3im[i] * v->c2.c3im[i];
    res->c2.c1im = u->c1.c1re[i] * v->c2.c1im[i] + u->c1.c1im[i] * v->c2.c1re[i]
                 + u->c1.c2re[i] * v->c2.c2im[i] + u->c1.c2im[i] * v->c2.c2re[i]
                 + u->c1.c3re[i] * v->c2.c3im[i] + u->c1.c3im[i] * v->c2.c3re[i];
    res->c2.c2re = u->c2.c1re[i] * v->c2.c1re[i] - u->c2.c1im[i] * v->c2.c1im[i]
                 + u->c2.c2re[i] * v->c2.c2re[i] - u->c2.c2im[i] * v->c2.c2im[i]
                 + u->c2.c3re[i] * v->c2.c3re[i] - u->c2.c3im[i] * v->c2.c3im[i];
    res->c2.c2im = u->c2.c1re[i] * v->c2.c1im[i] + u->c2.c1im[i] * v->c2.c1re[i]
                 + u->c2.c2re[i] * v->c2.c2im[i] + u->c2.c2im[i] * v->c2.c2re[i]
                 + u->c2.c3re[i] * v->c2.c3im[i] + u->c2.c3im[i] * v->c2.c3re[i];
    res->c2.c3re = u->c3.c1re[i] * v->c2.c1re[i] - u->c3.c1im[i] * v->c2.c1im[i]
                 + u->c3.c2re[i] * v->c2.c2re[i] - u->c3.c2im[i] * v->c2.c2im[i]
                 + u->c3.c3re[i] * v->c2.c3re[i] - u->c3.c3im[i] * v->c2.c3im[i];
    res->c2.c3im = u->c3.c1re[i] * v->c2.c1im[i] + u->c3.c1im[i] * v->c2.c1re[i]
                 + u->c3.c2re[i] * v->c2.c2im[i] + u->c3.c2im[i] * v->c2.c2re[i]
                 + u->c3.c3re[i] * v->c2.c3im[i] + u->c3.c3im[i] * v->c2.c3re[i];

    // column c3
    res->c3.c1re = u->c1.c1re[i] * v->c3.c1re[i] - u->c1.c1im[i] * v->c3.c1im[i]
                 + u->c1.c2re[i] * v->c3.c2re[i] - u->c1.c2im[i] * v->c3.c2im[i]
                 + u->c1.c3re[i] * v->c3.c3re[i] - u->c1.c3im[i] * v->c3.c3im[i];
    res->c3.c1im = u->c1.c1re[i] * v->c3.c1im[i] + u->c1.c1im[i] * v->c3.c1re[i]
                 + u->c1.c2re[i] * v->c3.c2im[i] + u->c1.c2im[i] * v->c3.c2re[i]
                 + u->c1.c3re[i] * v->c3.c3im[i] + u->c1.c3im[i] * v->c3.c3re[i];
    res->c3.c2re = u->c2.c1re[i] * v->c3.c1re[i] - u->c2.c1im[i] * v->c3.c1im[i]
                 + u->c2.c2re[i] * v->c3.c2re[i] - u->c2.c2im[i] * v->c3.c2im[i]
                 + u->c2.c3re[i] * v->c3.c3re[i] - u->c2.c3im[i] * v->c3.c3im[i];
    res->c3.c2im = u->c2.c1re[i] * v->c3.c1im[i] + u->c2.c1im[i] * v->c3.c1re[i]
                 + u->c2.c2re[i] * v->c3.c2im[i] + u->c2.c2im[i] * v->c3.c2re[i]
                 + u->c2.c3re[i] * v->c3.c3im[i] + u->c2.c3im[i] * v->c3.c3re[i];
    res->c3.c3re = u->c3.c1re[i] * v->c3.c1re[i] - u->c3.c1im[i] * v->c3.c1im[i]
                 + u->c3.c2re[i] * v->c3.c2re[i] - u->c3.c2im[i] * v->c3.c2im[i]
                 + u->c3.c3re[i] * v->c3.c3re[i] - u->c3.c3im[i] * v->c3.c3im[i];
    res->c3.c3im = u->c3.c1re[i] * v->c3.c1im[i] + u->c3.c1im[i] * v->c3.c1re[i]
                 + u->c3.c2re[i] * v->c3.c2im[i] + u->c3.c2im[i] * v->c3.c2re[i]
                 + u->c3.c3re[i] * v->c3.c3im[i] + u->c3.c3im[i] * v->c3.c3re[i];
}

// ---------------------------------------------------------------------------
// res = u†[i] * v†[i]  (register output, SoA inputs)
// ---------------------------------------------------------------------------
__device__ static inline void fsu3matdagxsu3matdag_reg(
    su3_mat_reg *__restrict__ res,
    const su3_mat_field *__restrict__ u,
    const su3_mat_field *__restrict__ v,
    size_t i)
{
    // column c1
    res->c1.c1re =  u->c1.c1re[i] * v->c1.c1re[i] - u->c1.c1im[i] * v->c1.c1im[i]
                 +  u->c2.c1re[i] * v->c1.c2re[i] - u->c2.c1im[i] * v->c1.c2im[i]
                 +  u->c3.c1re[i] * v->c1.c3re[i] - u->c3.c1im[i] * v->c1.c3im[i];
    res->c1.c1im = -u->c1.c1re[i] * v->c1.c1im[i] - u->c1.c1im[i] * v->c1.c1re[i]
                  -u->c2.c1re[i] * v->c1.c2im[i] - u->c2.c1im[i] * v->c1.c2re[i]
                  -u->c3.c1re[i] * v->c1.c3im[i] - u->c3.c1im[i] * v->c1.c3re[i];
    res->c2.c1re =  u->c1.c2re[i] * v->c1.c1re[i] - u->c1.c2im[i] * v->c1.c1im[i]
                 +  u->c2.c2re[i] * v->c1.c2re[i] - u->c2.c2im[i] * v->c1.c2im[i]
                 +  u->c3.c2re[i] * v->c1.c3re[i] - u->c3.c2im[i] * v->c1.c3im[i];
    res->c2.c1im = -u->c1.c2re[i] * v->c1.c1im[i] - u->c1.c2im[i] * v->c1.c1re[i]
                  -u->c2.c2re[i] * v->c1.c2im[i] - u->c2.c2im[i] * v->c1.c2re[i]
                  -u->c3.c2re[i] * v->c1.c3im[i] - u->c3.c2im[i] * v->c1.c3re[i];
    res->c3.c1re =  u->c1.c3re[i] * v->c1.c1re[i] - u->c1.c3im[i] * v->c1.c1im[i]
                 +  u->c2.c3re[i] * v->c1.c2re[i] - u->c2.c3im[i] * v->c1.c2im[i]
                 +  u->c3.c3re[i] * v->c1.c3re[i] - u->c3.c3im[i] * v->c1.c3im[i];
    res->c3.c1im = -u->c1.c3re[i] * v->c1.c1im[i] - u->c1.c3im[i] * v->c1.c1re[i]
                  -u->c2.c3re[i] * v->c1.c2im[i] - u->c2.c3im[i] * v->c1.c2re[i]
                  -u->c3.c3re[i] * v->c1.c3im[i] - u->c3.c3im[i] * v->c1.c3re[i];

    // column c2
    res->c1.c2re =  u->c1.c1re[i] * v->c2.c1re[i] - u->c1.c1im[i] * v->c2.c1im[i]
                 +  u->c2.c1re[i] * v->c2.c2re[i] - u->c2.c1im[i] * v->c2.c2im[i]
                 +  u->c3.c1re[i] * v->c2.c3re[i] - u->c3.c1im[i] * v->c2.c3im[i];
    res->c1.c2im = -u->c1.c1re[i] * v->c2.c1im[i] - u->c1.c1im[i] * v->c2.c1re[i]
                  -u->c2.c1re[i] * v->c2.c2im[i] - u->c2.c1im[i] * v->c2.c2re[i]
                  -u->c3.c1re[i] * v->c2.c3im[i] - u->c3.c1im[i] * v->c2.c3re[i];
    res->c2.c2re =  u->c1.c2re[i] * v->c2.c1re[i] - u->c1.c2im[i] * v->c2.c1im[i]
                 +  u->c2.c2re[i] * v->c2.c2re[i] - u->c2.c2im[i] * v->c2.c2im[i]
                 +  u->c3.c2re[i] * v->c2.c3re[i] - u->c3.c2im[i] * v->c2.c3im[i];
    res->c2.c2im = -u->c1.c2re[i] * v->c2.c1im[i] - u->c1.c2im[i] * v->c2.c1re[i]
                  -u->c2.c2re[i] * v->c2.c2im[i] - u->c2.c2im[i] * v->c2.c2re[i]
                  -u->c3.c2re[i] * v->c2.c3im[i] - u->c3.c2im[i] * v->c2.c3re[i];
    res->c3.c2re =  u->c1.c3re[i] * v->c2.c1re[i] - u->c1.c3im[i] * v->c2.c1im[i]
                 +  u->c2.c3re[i] * v->c2.c2re[i] - u->c2.c3im[i] * v->c2.c2im[i]
                 +  u->c3.c3re[i] * v->c2.c3re[i] - u->c3.c3im[i] * v->c2.c3im[i];
    res->c3.c2im = -u->c1.c3re[i] * v->c2.c1im[i] - u->c1.c3im[i] * v->c2.c1re[i]
                  -u->c2.c3re[i] * v->c2.c2im[i] - u->c2.c3im[i] * v->c2.c2re[i]
                  -u->c3.c3re[i] * v->c2.c3im[i] - u->c3.c3im[i] * v->c2.c3re[i];

    // column c3
    res->c1.c3re =  u->c1.c1re[i] * v->c3.c1re[i] - u->c1.c1im[i] * v->c3.c1im[i]
                 +  u->c2.c1re[i] * v->c3.c2re[i] - u->c2.c1im[i] * v->c3.c2im[i]
                 +  u->c3.c1re[i] * v->c3.c3re[i] - u->c3.c1im[i] * v->c3.c3im[i];
    res->c1.c3im = -u->c1.c1re[i] * v->c3.c1im[i] - u->c1.c1im[i] * v->c3.c1re[i]
                  -u->c2.c1re[i] * v->c3.c2im[i] - u->c2.c1im[i] * v->c3.c2re[i]
                  -u->c3.c1re[i] * v->c3.c3im[i] - u->c3.c1im[i] * v->c3.c3re[i];
    res->c2.c3re =  u->c1.c2re[i] * v->c3.c1re[i] - u->c1.c2im[i] * v->c3.c1im[i]
                 +  u->c2.c2re[i] * v->c3.c2re[i] - u->c2.c2im[i] * v->c3.c2im[i]
                 +  u->c3.c2re[i] * v->c3.c3re[i] - u->c3.c2im[i] * v->c3.c3im[i];
    res->c2.c3im = -u->c1.c2re[i] * v->c3.c1im[i] - u->c1.c2im[i] * v->c3.c1re[i]
                  -u->c2.c2re[i] * v->c3.c2im[i] - u->c2.c2im[i] * v->c3.c2re[i]
                  -u->c3.c2re[i] * v->c3.c3im[i] - u->c3.c2im[i] * v->c3.c3re[i];
    res->c3.c3re =  u->c1.c3re[i] * v->c3.c1re[i] - u->c1.c3im[i] * v->c3.c1im[i]
                 +  u->c2.c3re[i] * v->c3.c2re[i] - u->c2.c3im[i] * v->c3.c2im[i]
                 +  u->c3.c3re[i] * v->c3.c3re[i] - u->c3.c3im[i] * v->c3.c3im[i];
    res->c3.c3im = -u->c1.c3re[i] * v->c3.c1im[i] - u->c1.c3im[i] * v->c3.c1re[i]
                  -u->c2.c3re[i] * v->c3.c2im[i] - u->c2.c3im[i] * v->c3.c2re[i]
                  -u->c3.c3re[i] * v->c3.c3im[i] - u->c3.c3im[i] * v->c3.c3re[i];
}

// ---------------------------------------------------------------------------
// Re Tr(u * v)  (both inputs in registers, scalar output)
// ---------------------------------------------------------------------------
__device__ static inline double fsu3matxsu3mat_retrace_reg(
    const su3_mat_reg *u, const su3_mat_reg *v)
{
    double tr_1 = u->c1.c1re * v->c1.c1re - u->c1.c1im * v->c1.c1im
                + u->c1.c2re * v->c2.c1re - u->c1.c2im * v->c2.c1im
                + u->c1.c3re * v->c3.c1re - u->c1.c3im * v->c3.c1im;
    double tr_2 = u->c2.c1re * v->c1.c2re - u->c2.c1im * v->c1.c2im
                + u->c2.c2re * v->c2.c2re - u->c2.c2im * v->c2.c2im
                + u->c2.c3re * v->c3.c2re - u->c2.c3im * v->c3.c2im;
    double tr_3 = u->c3.c1re * v->c1.c3re - u->c3.c1im * v->c1.c3im
                + u->c3.c2re * v->c2.c3re - u->c3.c2im * v->c2.c3im
                + u->c3.c3re * v->c3.c3re - u->c3.c3im * v->c3.c3im;
    return tr_1 + tr_2 + tr_3;
}

// ---------------------------------------------------------------------------
// Host-side memory management (implementations in su3v_cuda.cu)
// ---------------------------------------------------------------------------
void su3_mat_field_cuda_alloc(su3_mat_field *d, size_t volume);
void su3_mat_field_cuda_free(su3_mat_field *d);
void su3_mat_field_cuda_upload(su3_mat_field *d, const su3_mat_field *h);
void su3_mat_field_cuda_download(su3_mat_field *h, const su3_mat_field *d);

void doublev_cuda_alloc(doublev *d, size_t volume);
void doublev_cuda_free(doublev *d);
void doublev_cuda_download(doublev *h, const doublev *d);

// Fused kernel launcher
void launch_fsu3_combined(
    doublev *d_res,
    const su3_mat_field *d_u, const su3_mat_field *d_v,
    const su3_mat_field *d_w, const su3_mat_field *d_x,
    size_t volume, int threads_per_block);

void launch_flush_cache(double *d_buf, size_t n);

#endif // SU3V_CUDA_CUH
