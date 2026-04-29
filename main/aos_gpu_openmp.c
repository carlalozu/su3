#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "global.h"
#include "su3v.h"
#include "su3v_openmp.h"
#include "ufields.h"

static const size_t FLUSH_NELEMS = 15728640UL;

int main(int argc, char *argv[])
{
    int reps = 100;
    int idx  = 0;
    if (argc > 1) reps = atoi(argv[1]);
    if (argc > 2) idx  = atoi(argv[2]);

    printf("AoS OpenMP offload benchmark\n");
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

    // -----------------------------------------------------------------------
    // Map data to device
    // -----------------------------------------------------------------------
    #pragma omp target enter data map(alloc: h_u[0:VOLUME], h_v[0:VOLUME], h_w[0:VOLUME], h_x[0:VOLUME])
    #pragma omp target enter data map(alloc: h_res[0:VOLUME])

    #pragma omp target teams distribute parallel for
    for (size_t i = 0; i < VOLUME; i++) {
        uint64_t thread_state = 12345ULL + i;
        random_su3mat(&h_u[i], &thread_state);
        random_su3mat(&h_v[i], &thread_state);
        random_su3mat(&h_w[i], &thread_state);
        random_su3mat(&h_x[i], &thread_state);
    }

    double *flush_buf = (double *)malloc(FLUSH_NELEMS * sizeof(double));
    #pragma omp target enter data map(alloc: flush_buf[0:FLUSH_NELEMS])

    // -----------------------------------------------------------------------
    // Warm-up
    // -----------------------------------------------------------------------
    for (int r = 0; r < 3; r++) {
        #pragma omp target teams distribute parallel for
        for (size_t i = 0; i < VOLUME; i++) {
            su3_mat_c temp, res;
            su3matxsu3mat      (&temp, &h_u[i], &h_v[i]);
            su3matdagxsu3matdag(&res,  &h_w[i], &h_x[i]);
            h_res[i] = su3matxsu3mat_retrace(&temp, &res);
        }
    }

    // -----------------------------------------------------------------------
    // Benchmark
    // -----------------------------------------------------------------------
    double total_s = 0.0;

    for (int r = 0; r < reps; r++) {
        #pragma omp target teams distribute parallel for
        for (size_t j = 0; j < FLUSH_NELEMS; j++)
            flush_buf[j] += 1.0;

        double t0 = omp_get_wtime();
        #pragma omp target teams distribute parallel for
        for (size_t i = 0; i < VOLUME; i++) {
            su3_mat_c temp, res;
            su3matxsu3mat      (&temp, &h_u[i], &h_v[i]);
            su3matdagxsu3matdag(&res,  &h_w[i], &h_x[i]);
            h_res[i] = su3matxsu3mat_retrace(&temp, &res);
        }
        total_s += omp_get_wtime() - t0;
    }

    double avg_s  = total_s / reps;
    double gflops = (double)VOLUME * 432.0 / avg_s * 1e-9;
    double gbytes = (double)VOLUME * 1160.0;

    printf("\nResults\n");
    printf("  total  = %.6f s  (%d reps)\n", total_s, reps);
    printf("  avg    = %.6f s  (%.3f ms)\n", avg_s, avg_s * 1e3);
    printf("  GFLOP/s = %.2f\n", gflops);
    printf("  GB     = %.2f\n", gbytes);

    // -----------------------------------------------------------------------
    // Verify one element
    // -----------------------------------------------------------------------
    #pragma omp target update from(h_res[0:VOLUME])
    if (idx >= 0 && (size_t)idx < (size_t)VOLUME)
        printf("  res[%d] = %.10f\n", idx, h_res[idx]);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    #pragma omp target exit data map(release: flush_buf[0:FLUSH_NELEMS])
    free(flush_buf);

    #pragma omp target exit data map(release: h_u[0:VOLUME], h_v[0:VOLUME], h_w[0:VOLUME], h_x[0:VOLUME], h_res[0:VOLUME])
    free(h_u); free(h_v); free(h_w); free(h_x); free(h_res);

    return 0;
}
