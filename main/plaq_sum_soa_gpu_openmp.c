
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "global.h"
#include "lattice.h"
#include "uflds.h"
#include "su3v_openmp.h"
#include "profiler.h"
#include "random.h"

prof_section compute = {.name = "compute"};

static const size_t FLUSH_NELEMS = 15728640UL;

int main(int argc, char *argv[])
{
    int reps = 100;
    if (argc > 1) reps = atoi(argv[1]);
    omp_set_num_threads(NTHREAD);

    printf("\nPlaquette sum OpenMP GPU offload benchmark\n");
    printf("------------------------------------------\n");
    printf("Volume:      %d\n", VOLUME);
    printf("Repetitions: %d\n", reps);
    printf("OpenMP threads: %d\n", NTHREAD);
    printf("Data structure: SoA\n");
    printf("Lattice geometry: %ix%ix%ix%i\n", L0,L1,L2,L3);
    printf("Local lattice geometry: %ix%ix%ix%i\n\n", L0_TRD,L1_TRD,L2_TRD,L3_TRD);

    /* geometry() builds ipt/iup/iupT and maps them to the device.
     * random_udv() fills the SoA gauge field with random SU(3)
     * matrices; enter_su3_mat_field() uploads it to the device. */
    start_ranlux(0, 12345);
    geometry();

    su3_mat_field u_fld;
    su3_mat_field_init(&u_fld, 4*VOLUME);
    random_udv(&u_fld);
    enter_su3_mat_field(&u_fld);

    /* Cache-flush buffer */
    double *flush_buf = (double *)malloc(FLUSH_NELEMS * sizeof(double));
    #pragma omp target enter data map(alloc: flush_buf[0:FLUSH_NELEMS])

    /* -----------------------------------------------------------------------
     * Warm-up
     * ----------------------------------------------------------------------- */
    for (int r = 0; r < 3; r++)
        plaq_sum_dblev(&u_fld, 1);

    prof_reset(&compute);

    /* -----------------------------------------------------------------------
     * Benchmark — flush cache before each call, accumulate wall time.
     * plaq_sum_dblev internally calls prof_begin/prof_end on `compute`.
     * ----------------------------------------------------------------------- */
    double total_s = 0.0;
    double last_sum = 0.0;

    for (int r = 0; r < reps; r++) {
        /* Flush L2/L3 cache */
        #pragma omp target teams distribute parallel for
        for (size_t j = 0; j < FLUSH_NELEMS; j++)
            flush_buf[j] += 1.0;

        double t0 = omp_get_wtime();
        last_sum = plaq_sum_dblev(&u_fld, 1);
        total_s += omp_get_wtime() - t0;
    }

    double avg_s  = total_s / (double)reps;
    long long flops = 432LL * 6 * VOLUME; /* 432 flop/plaquette × 6 planes × V */
    double gflops = (double)flops / avg_s * 1e-9;

    printf("\nResults\n");
    printf("Local gauge field size (KB): %d\n",
           (int)(72 * VOLUME * sizeof(double) / 1024));
    printf("Average time for plaq_sum_dblev (sec): %.9f\n", avg_s);
    printf("Flops per call: %lld\n", flops);
    printf("Total performance (GFlop/s): %.2f\n", gflops);
    printf("Time per lattice point (sec): %.9f\n", avg_s / (double)VOLUME);
    printf("Plaquette sum: %.10f\n\n", last_sum);

    prof_report(&compute);

    /* -----------------------------------------------------------------------
     * Cleanup
     * ----------------------------------------------------------------------- */
    #pragma omp target exit data map(release: flush_buf[0:FLUSH_NELEMS])
    free(flush_buf);

    su3_mat_field_free(&u_fld);

    return 0;
}
