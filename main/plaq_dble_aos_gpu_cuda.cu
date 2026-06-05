#include <stdio.h>
#include <stdlib.h>
#include "global.h"
#include "su3.h"
#include "su3v_cuda.cuh"
#include "lattice.h"

static const size_t FLUSH_NELEMS = 15728640UL;

__global__ static void plaq_dble(
    double *res,
    su3_dble *d_fld,
    size_t volume)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= volume) return;

    su3_dble tmp_a, temp_b;
    su3xsu3      (&tmp_a,  &d_fld[0*volume+i], &d_fld[1*volume+i]);
    su3dagxsu3dag(&temp_b, &d_fld[2*volume+i], &d_fld[3*volume+i]);
    res[i] = cm3x3_retr(&tmp_a, &temp_b);
}

int main(int argc, char *argv[])
{
    int reps = 100;
    int idx  = 0;
    if (argc > 1) reps = atoi(argv[1]);
    if (argc > 2) idx  = atoi(argv[2]);


    printf("\nplaq_dble AoS CUDA kernel benchmark\n");
    printf("------------------------------------------\n");
    printf("Volume:      %d\n", VOLUME);
    printf("Repetitions: %d\n", reps);
    printf("Data structure: AoS\n");
    printf("Lattice geometry: %ix%ix%ix%i\n", L0,L1,L2,L3);
    printf("Local lattice geometry: %ix%ix%ix%i\n\n", L0_TRD,L1_TRD,L2_TRD,L3_TRD);

    // -----------------------------------------------------------------------
    // Host fields
    // -----------------------------------------------------------------------
    su3_dble  *h_fld = udfld();
    double    *h_res = (double  *)malloc(VOLUME * sizeof(double));

    start_ranlux(0, 12345);
    geometry();
    random_ud();

    // -----------------------------------------------------------------------
    // Device fields
    // -----------------------------------------------------------------------
    su3_dble *d_fld;
    double    *d_res;

    CUDA_CHECK(cudaMalloc(&d_fld, 4*VOLUME * sizeof(su3_dble)));
    CUDA_CHECK(cudaMalloc(&d_res,   VOLUME * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_fld, h_fld, 4*VOLUME * sizeof(su3_dble), cudaMemcpyHostToDevice));

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
        plaq_dble<<<blocks, THREADS>>>(d_res, d_fld, VOLUME);
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
        plaq_dble<<<blocks, THREADS>>>(d_res, d_fld, VOLUME);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_ms += ms;
    }

    double avg_ms = total_ms / reps;
    double avg_s  = avg_ms * 1e-3;
    double gflops = (double)VOLUME * 432.0 / avg_s * 1e-9;

    printf("\nResults\n");
    printf("Local gauge field size (KB): %d\n",
           (int)(72 * VOLUME * sizeof(double) / 1024));
    printf("  total  = %.6f s  (%d reps)\n", total_ms * 1e-3, reps);
    printf("  avg    = %.6f s  (%.3f ms)\n", avg_s, avg_ms);
    printf("  GFLOP/s = %.2f\n", gflops);
    printf("Time per lattice point (sec): %.9f\n", avg_s / (double)VOLUME);

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

    CUDA_CHECK(cudaFree(d_fld));
    CUDA_CHECK(cudaFree(d_res));

    free(h_res);

    return 0;
}
