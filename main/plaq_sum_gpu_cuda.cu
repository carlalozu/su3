#include <stdio.h>
#include <stdlib.h>
#include "global.h"
#include "su3.h"
#include "su3prod.h"
#include "su3v_cuda.cuh"

static const size_t FLUSH_NELEMS = 15728640UL;

/* Each thread handles one lattice site and accumulates the contributions
 * from all 6 (mu,nu) plaquette orientations.  With periodic BC (bc=3) every
 * plaquette contributes without weight, matching local_plaq_sum_dble(iw=0).
 *
 * d_iupT is a flat [4*VOLUME] array: d_iupT[mu*volume + ix] is the forward
 * neighbor of site ix in direction mu, identical to iupT[mu][ix].
 *
 * The site-local partial sum is accumulated into *total via atomicAdd.
 */
__global__ static void plaq_sum_kernel(
    double         *total,
    const su3_dble *udb,
    const int      *d_iupT,
    int             volume)
{
    int ix = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (ix >= volume) return;

    double pa = 0.0;

    for (int mu = 0; mu < 4; mu++) {
        for (int nu = mu + 1; nu < 4; nu++) {
            /* plaq_uidx inlined:
             *   ip[0] = offset(ix,     mu) = mu*volume + ix
             *   ip[1] = offset(iup_mu, nu) = nu*volume + iupT[mu][ix]
             *   ip[2] = offset(ix,     nu) = nu*volume + ix
             *   ip[3] = offset(iup_nu, mu) = mu*volume + iupT[nu][ix]   */
            int iup_mu = d_iupT[mu * volume + ix];
            int iup_nu = d_iupT[nu * volume + ix];

            int ip0 = mu * volume + ix;
            int ip1 = nu * volume + iup_mu;
            int ip2 = nu * volume + ix;
            int ip3 = mu * volume + iup_nu;

            su3_dble wd1, wd2;
            su3xsu3     (&wd1, (su3_dble *)udb + ip0, (su3_dble *)udb + ip1);
            su3dagxsu3dag(&wd2, (su3_dble *)udb + ip3, (su3_dble *)udb + ip2);
            pa += cm3x3_retr(&wd1, &wd2);
        }
    }

    atomicAdd(total, pa);
}

/* Build the flat iupT[4*VOLUME] neighbor table on the host.
 * Sites are lexicographically ordered: ix = y3 + y2*L3 + y1*L2*L3 + y0*L1*L2*L3.
 * Periodic boundary conditions are applied in all directions. */
static void build_iupT(int *iupT_flat)
{
    for (int ix = 0; ix < VOLUME; ix++) {
        int tmp = ix;
        int y3  = tmp % L3; tmp /= L3;
        int y2  = tmp % L2; tmp /= L2;
        int y1  = tmp % L1; tmp /= L1;
        int y0  = tmp;

        iupT_flat[0 * VOLUME + ix] = y3 +  y2       *L3 +  y1       *L2*L3 + ((y0+1)%L0)*L1*L2*L3;
        iupT_flat[1 * VOLUME + ix] = y3 +  y2       *L3 + ((y1+1)%L1)*L2*L3 +  y0       *L1*L2*L3;
        iupT_flat[2 * VOLUME + ix] = y3 + ((y2+1)%L2)*L3 +  y1       *L2*L3 +  y0       *L1*L2*L3;
        iupT_flat[3 * VOLUME + ix] = ((y3+1)%L3) + y2*L3 + y1*L2*L3 + y0*L1*L2*L3;
    }
}

int main(int argc, char *argv[])
{
    int reps = 100;
    if (argc > 1) reps = atoi(argv[1]);

    printf("Plaquette sum CUDA kernel benchmark\n");
    printf("------------------------------------\n");
    printf("Volume:      %d\n", VOLUME);
    printf("Repetitions: %d\n\n", reps);

    // -----------------------------------------------------------------------
    // Host: geometry and gauge field
    // -----------------------------------------------------------------------
    int *h_iupT = (int *)malloc(4 * VOLUME * sizeof(int));
    build_iupT(h_iupT);

    su3_dble *h_udb = (su3_dble *)malloc(4 * VOLUME * sizeof(su3_dble));
    rlxd_init(1, 1, 1, 1);
    for (int i = 0; i < 4 * VOLUME; i++)
        random_su3_dble(&h_udb[i]);

    // -----------------------------------------------------------------------
    // Device: allocate and upload
    // -----------------------------------------------------------------------
    su3_dble *d_udb  = nullptr;
    int      *d_iupT = nullptr;
    double   *d_total = nullptr;

    CUDA_CHECK(cudaMalloc(&d_udb,   4 * VOLUME * sizeof(su3_dble)));
    CUDA_CHECK(cudaMalloc(&d_iupT,  4 * VOLUME * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_udb,  h_udb,  4 * VOLUME * sizeof(su3_dble), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iupT, h_iupT, 4 * VOLUME * sizeof(int),      cudaMemcpyHostToDevice));

    // Flush buffer
    double *d_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flush, FLUSH_NELEMS * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_flush, 0, FLUSH_NELEMS * sizeof(double)));

    const int THREADS = 256;
    int blocks       = (VOLUME       + THREADS - 1) / THREADS;
    int flush_blocks = ((int)FLUSH_NELEMS + THREADS - 1) / THREADS;

    // -----------------------------------------------------------------------
    // Warm-up
    // -----------------------------------------------------------------------
    for (int r = 0; r < 3; r++) {
        CUDA_CHECK(cudaMemset(d_total, 0, sizeof(double)));
        plaq_sum_kernel<<<blocks, THREADS>>>(d_total, d_udb, d_iupT, VOLUME);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // -----------------------------------------------------------------------
    // Benchmark
    // -----------------------------------------------------------------------
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    double total_ms = 0.0;
    double last_sum = 0.0;

    for (int r = 0; r < reps; r++) {
        flush_cache_kernel<<<flush_blocks, THREADS>>>(d_flush, FLUSH_NELEMS);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_total, 0, sizeof(double)));
        CUDA_CHECK(cudaEventRecord(ev_start));
        plaq_sum_kernel<<<blocks, THREADS>>>(d_total, d_udb, d_iupT, VOLUME);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_ms += ms;
    }

    CUDA_CHECK(cudaMemcpy(&last_sum, d_total, sizeof(double), cudaMemcpyDeviceToHost));

    double avg_ms  = total_ms / reps;
    double avg_s   = avg_ms * 1e-3;
    long long flops = 432LL * 6 * VOLUME;
    double gflops  = (double)flops / avg_s * 1e-9;

    printf("Local gauge field size (KB): %d\n",
           (int)(72 * VOLUME * sizeof(double) / 1024));
    printf("Volume: %d\n", VOLUME);
    printf("Number of repetitions: %d\n", reps);
    printf("Average time for plaq_sum (ms):  %.6f\n", avg_ms);
    printf("Average time for plaq_sum (sec): %.9f\n", avg_s);
    printf("Flops per call: %lld\n", flops);
    printf("Total performance (GFlop/s): %.2f\n", gflops);
    printf("Time per lattice point (sec): %.9f\n", avg_s / (double)VOLUME);
    printf("Plaquette sum: %.10f\n\n", last_sum);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_flush));
    CUDA_CHECK(cudaFree(d_total));
    CUDA_CHECK(cudaFree(d_iupT));
    CUDA_CHECK(cudaFree(d_udb));

    free(h_udb);
    free(h_iupT);

    return 0;
}
