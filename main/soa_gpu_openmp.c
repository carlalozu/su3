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

    printf("SoA OpenMP offload benchmark\n");
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
    // Map data to device
    // -----------------------------------------------------------------------
    enter_su3_mat_field(&h_u);
    enter_su3_mat_field(&h_v);
    enter_su3_mat_field(&h_w);
    enter_su3_mat_field(&h_x);
    enter_double_field(&h_res);

    double *flush_buf = (double *)malloc(FLUSH_NELEMS * sizeof(double));
    #pragma omp target enter data map(alloc: flush_buf[0:FLUSH_NELEMS])

    // -----------------------------------------------------------------------
    // Warm-up
    // -----------------------------------------------------------------------
    for (int r = 0; r < 3; r++) {
        #pragma omp target teams distribute parallel for
        for (size_t i = 0; i < VOLUME; i++) {
            su3_mat_dble temp, res;
            fsu3matxsu3mat      (&temp, &h_u, &h_v, i);
            fsu3matdagxsu3matdag(&res,  &h_w, &h_x, i);
            h_res.base[i] = su3matdxsu3matd_retrace(&temp, &res);
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
            su3_mat_dble temp, res;
            fsu3matxsu3mat      (&temp, &h_u, &h_v, i);
            fsu3matdagxsu3matdag(&res,  &h_w, &h_x, i);
            h_res.base[i] = su3matdxsu3matd_retrace(&temp, &res);
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
    #pragma omp target update from(h_res.base[0 : h_res.volume])
    if (idx >= 0 && (size_t)idx < (size_t)VOLUME)
        printf("  res[%d] = %.10f\n", idx, h_res.base[idx]);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    #pragma omp target exit data map(release: flush_buf[0:FLUSH_NELEMS])
    free(flush_buf);

    su3_mat_field_free(&h_u);
    su3_mat_field_free(&h_v);
    su3_mat_field_free(&h_w);
    su3_mat_field_free(&h_x);
    free(h_res.base);

    return 0;
}
