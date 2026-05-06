
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "global.h"
#include "lattice.h"
#include "uflds.h"
#include "profiler.h"
#include "random.h"

prof_section compute = {.name = "compute"};

static const size_t FLUSH_NELEMS = 15728640UL;

int main(int argc, char *argv[])
{
    int reps = 100;
    if (argc > 1) reps = atoi(argv[1]);

    printf("Plaquette sum OpenMP GPU offload benchmark\n");
    printf("------------------------------------------\n");
    printf("Volume:      %d\n", VOLUME);
    printf("Repetitions: %d\n\n", reps);
    printf("OpenMP threads: %d\n\n", NTHREAD);
    printf("Data structure: AoS\n\n");
    printf("Lattice geometry: %ix%ix%ix%i\n\n", L0,L1,L2,L3);
    printf("Local Lattice geometry: %ix%ix%ix%i\n\n", L0_TRD,L1_TRD,L2_TRD,L3_TRD);

    /* geometry() builds ipt/iup/iupT and maps them to the device.
     * random_ud() allocates the gauge field via udfld(), fills it with
     * random SU(3) matrices on the host, and uploads it to the device. */
    start_ranlux(0, 12345);
    geometry();
    random_ud();

    /* Cache-flush buffer */
    double *flush_buf = (double *)malloc(FLUSH_NELEMS * sizeof(double));
    #pragma omp target enter data map(alloc: flush_buf[0:FLUSH_NELEMS])

    /* -----------------------------------------------------------------------
     * Warm-up
     * ----------------------------------------------------------------------- */
    for (int r = 0; r < 3; r++)
        plaq_sum_dble(1);

    prof_reset(&compute);

    /* -----------------------------------------------------------------------
     * Benchmark — mirrors openQCD devel/uflds/time.c structure:
     *   flush cache before each call, accumulate wall time.
     * plaq_sum_dble internally calls prof_begin/prof_end on `compute`.
     * ----------------------------------------------------------------------- */
    double total_s = 0.0;
    double last_sum = 0.0;

    for (int r = 0; r < reps; r++) {
        /* Flush L2/L3 cache */
        #pragma omp target teams distribute parallel for
        for (size_t j = 0; j < FLUSH_NELEMS; j++)
            flush_buf[j] += 1.0;

        double t0 = omp_get_wtime();
        last_sum = plaq_sum_dble(1);
        total_s += omp_get_wtime() - t0;
    }

    double avg_s  = total_s / (double)reps;
    long long flops = 432LL * 6 * VOLUME; /* 432 flop/plaquette × 6 planes × V */
    double gflops = (double)flops / avg_s * 1e-9;

    printf("Local gauge field size (KB): %d\n",
           (int)(72 * VOLUME * sizeof(double) / 1024));
    printf("Volume: %d\n", VOLUME);
    printf("Number of repetitions: %d\n", reps);
    printf("Average time for plaq_sum_dble (sec): %.9f\n", avg_s);
    printf("Flops per call: %lld\n", flops);
    printf("Total performance (GFlop/s): %.2f\n", gflops);
    printf("Time per lattice point (sec): %.9f\n", avg_s / (double)VOLUME);
    printf("Plaquette sum: %.10f\n\n", last_sum);

    prof_report(&compute);

    /* -----------------------------------------------------------------------
     * Cleanup (gauge field lifetime managed by udfld/uflds.c)
     * ----------------------------------------------------------------------- */
    #pragma omp target exit data map(release: flush_buf[0:FLUSH_NELEMS])
    free(flush_buf);

    return 0;
}
