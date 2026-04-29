#include <stdio.h>
#include <stdlib.h>
#include "global.h"
#include "su3.h"
#include "su3v_cuda.cuh"

static const size_t FLUSH_NELEMS = 15728640UL;

__global__ static void plaq_dble(
    double *res,
    const su3_mat_c *d_u, const su3_mat_c *d_v,
    const su3_mat_c *d_w, const su3_mat_c *d_x,
    size_t volume)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= volume) return;

    su3_mat_c tmp_a, temp_b;
    su3matxsu3mat(&tmp_a, &d_u[i], &d_v[i]);
    su3matdagxsu3matdag(&temp_b, &d_w[i], &d_x[i]);
    res[i] = su3matxsu3mat_retrace(&tmp_a, &temp_b);
}

int main(int argc, char *argv[])
{
    int reps = 100;
    int idx  = 0;
    if (argc > 1) reps = atoi(argv[1]);
    if (argc > 2) idx  = atoi(argv[2]);

    printf("AoS CUDA kernel benchmark\n");
    printf("Volume:      %d\n", VOLUME);
    printf("Repetitions: %d\n", reps);

    // -----------------------------------------------------------------------
    // Host fields
    // -----------------------------------------------------------------------
    su3_mat_c *h_u   = (su3_mat_c *)malloc(VOLUME * sizeof(su3_mat_c));
    su3_mat_c *h_v   = (su3_mat_c *)malloc(VOLUME * sizeof(su3_mat_c));
    su3_mat_c *h_w   = (su3_mat_c *)malloc(VOLUME * sizeof(su3_mat_c));
    su3_mat_c *h_x   = (su3_mat_c *)malloc(VOLUME * sizeof(su3_mat_c));
    double    *h_res = (double    *)malloc(VOLUME * sizeof(double));

    for (size_t i = 0; i < VOLUME; i++) {
        uint64_t state = 12345ULL + i;
        random_su3mat(&h_u[i], &state);
        random_su3mat(&h_v[i], &state);
        random_su3mat(&h_w[i], &state);
        random_su3mat(&h_x[i], &state);
    }

    // -----------------------------------------------------------------------
    // Device fields
    // -----------------------------------------------------------------------
    su3_mat_c *d_u, *d_v, *d_w, *d_x;
    double    *d_res;

    CUDA_CHECK(cudaMalloc(&d_u,   VOLUME * sizeof(su3_mat_c)));
    CUDA_CHECK(cudaMalloc(&d_v,   VOLUME * sizeof(su3_mat_c)));
    CUDA_CHECK(cudaMalloc(&d_w,   VOLUME * sizeof(su3_mat_c)));
    CUDA_CHECK(cudaMalloc(&d_x,   VOLUME * sizeof(su3_mat_c)));
    CUDA_CHECK(cudaMalloc(&d_res, VOLUME * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_u, h_u, VOLUME * sizeof(su3_mat_c), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, VOLUME * sizeof(su3_mat_c), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, VOLUME * sizeof(su3_mat_c), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, VOLUME * sizeof(su3_mat_c), cudaMemcpyHostToDevice));

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
        plaq_dble<<<blocks, THREADS>>>(d_res, d_u, d_v, d_w, d_x, VOLUME);
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
        plaq_dble<<<blocks, THREADS>>>(d_res, d_u, d_v, d_w, d_x, VOLUME);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_ms += ms;
    }

    double avg_ms = total_ms / reps;
    double avg_s  = avg_ms * 1e-3;
    double gflops = (double)VOLUME * 432.0 / avg_s * 1e-9;
    double gbytes = (double)VOLUME * 1160.0;

    printf("\nResults\n");
    printf("  total  = %.6f s  (%d reps)\n", total_ms * 1e-3, reps);
    printf("  avg    = %.6f s  (%.3f ms)\n", avg_s, avg_ms);
    printf("  GFLOP/s = %.2f\n", gflops);
    printf("  GB     = %.2f\n", gbytes);

    // -----------------------------------------------------------------------
    // Verify one element
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_res, d_res, VOLUME * sizeof(double), cudaMemcpyDeviceToHost));
    if (idx >= 0 && (size_t)idx < (size_t)VOLUME)
        printf("  res[%d] = %.10f\n", idx, h_res[idx]);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_flush));

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_res));

    free(h_u); free(h_v); free(h_w); free(h_x); free(h_res);

    return 0;
}
