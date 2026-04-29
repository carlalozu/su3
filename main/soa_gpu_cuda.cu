#include <stdio.h>
#include <stdlib.h>
#include "global.h"
#include "su3v.h"
#include "su3v_cuda.cuh"

static const size_t FLUSH_NELEMS = 15728640UL;

__global__ static void flush_cache_kernel(double *buf, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] += 1.0;
}

__global__ static void plaq_dblev(
    double *res,
    const su3_mat_field d_u, const su3_mat_field d_v,
    const su3_mat_field d_w, const su3_mat_field d_x,
    size_t volume)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= volume) return;

    su3_mat_c u, v, w, x, tmp;
    fsu3matxsu3mat(&tmp, &u, &v);
    fsu3matdagxsu3matdag(&tmp, &tmp, &w);
    res[i] = su3matdxsu3matd_retrace(&tmp, &x);
}

int main(int argc, char *argv[])
{
    int reps = 100;
    int idx  = 0;
    if (argc > 1) reps = atoi(argv[1]);
    if (argc > 2) idx  = atoi(argv[2]);

    printf("SoA CUDA kernel benchmark\n");
    printf("Volume:      %d\n", VOLUME);
    printf("Repetitions: %d\n", reps);

    // -----------------------------------------------------------------------
    // Host fields
    // -----------------------------------------------------------------------
    su3_mat_field h_u, h_v, h_w, h_x;
    doublev       h_res;

    su3_mat_field_init(&h_u, VOLUME);
    su3_mat_field_init(&h_v, VOLUME);
    su3_mat_field_init(&h_w, VOLUME);
    su3_mat_field_init(&h_x, VOLUME);
    doublev_init(&h_res, VOLUME);

    random_su3mat_field(&h_u);
    random_su3mat_field(&h_v);
    random_su3mat_field(&h_w);
    random_su3mat_field(&h_x);

    // -----------------------------------------------------------------------
    // Device fields
    // -----------------------------------------------------------------------
    su3_mat_field d_u, d_v, d_w, d_x;
    doublev       d_res;

    su3_mat_field_cuda_alloc(&d_u, VOLUME);
    su3_mat_field_cuda_alloc(&d_v, VOLUME);
    su3_mat_field_cuda_alloc(&d_w, VOLUME);
    su3_mat_field_cuda_alloc(&d_x, VOLUME);
    doublev_cuda_alloc(&d_res, VOLUME);

    su3_mat_field_cuda_upload(&d_u, &h_u);
    su3_mat_field_cuda_upload(&d_v, &h_v);
    su3_mat_field_cuda_upload(&d_w, &h_w);
    su3_mat_field_cuda_upload(&d_x, &h_x);

    // Flush buffer
    double *d_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flush, FLUSH_NELEMS * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_flush, 0, FLUSH_NELEMS * sizeof(double)));

    const int THREADS = 256;

    // -----------------------------------------------------------------------
    // Warm-up
    // -----------------------------------------------------------------------
    int blocks = ((int)VOLUME + THREADS - 1) / THREADS;
    for (int r = 0; r < 3; r++) {
        plaq_dblev<<<blocks, THREADS>>>(d_res.base, d_u, d_v, d_w, d_x, VOLUME);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // -----------------------------------------------------------------------
    // Benchmark
    // -----------------------------------------------------------------------
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    double total_ms = 0.0;
    int flush_blocks = ((int)FLUSH_NELEMS + THREADS - 1) / THREADS;

    for (int r = 0; r < reps; r++) {
        flush_cache_kernel<<<flush_blocks, THREADS>>>(d_flush, FLUSH_NELEMS);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(ev_start));
        plaq_dblev<<<blocks, THREADS>>>(d_res.base, d_u, d_v, d_w, d_x, VOLUME);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_ms += ms;
    }

    double avg_ms = total_ms / reps;
    double avg_s  = avg_ms * 1e-3;

    // Arithmetic intensity: 2*(198+198+36) FLOP per site (mat*mat + dagdag + retrace)
    // Memory: 4 input matrices * 18 complex doubles = 4*18*2*8 = 1152 B/site
    //         + 1 output double = 8 B/site → 1160 B/site
    double gflops   = (double)VOLUME * 432.0 / avg_s * 1e-9;
    double gbytes   = (double)VOLUME * 1160.0;

    printf("\nResults\n");
    printf("  total  = %.6f s  (%d reps)\n", total_ms * 1e-3, reps);
    printf("  avg    = %.6f s  (%.3f ms)\n", avg_s, avg_ms);
    printf("  GFLOP/s = %.2f\n", gflops);
    printf("  GB     = %.2f\n", gbytes);

    // -----------------------------------------------------------------------
    // Verify one element
    // -----------------------------------------------------------------------
    doublev_cuda_download(&h_res, &d_res);
    if (idx >= 0 && (size_t)idx < (size_t)VOLUME)
        printf("  res[%d] = %.10f\n", idx, h_res.base[idx]);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_flush));

    su3_mat_field_cuda_free(&d_u);
    su3_mat_field_cuda_free(&d_v);
    su3_mat_field_cuda_free(&d_w);
    su3_mat_field_cuda_free(&d_x);
    doublev_cuda_free(&d_res);

    su3_mat_field_free(&h_u);
    su3_mat_field_free(&h_v);
    su3_mat_field_free(&h_w);
    su3_mat_field_free(&h_x);
    free(h_res.base);

    return 0;
}
