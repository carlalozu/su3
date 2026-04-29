#ifndef SU3V_CUDA_CUH
#define SU3V_CUDA_CUH

#include <stdio.h>
#include <stdlib.h>
#include "su3.h"
#include "su3v.h"
#include "ufields.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)


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
void launch_plaq_dble(
    doublev *d_res,
    const su3_mat_field *d_u, const su3_mat_field *d_v,
    const su3_mat_field *d_w, const su3_mat_field *d_x,
    size_t volume, int threads_per_block);

void launch_flush_cache(double *d_buf, size_t n);

#endif // SU3V_CUDA_CUH
