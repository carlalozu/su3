#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "global.h"
#include "su3prod.h"
#include "su3v.h"
#include "su3v_openmp.h"
#include "uflds.h"

static const size_t FLUSH_NELEMS = 15728640UL;

int main(int argc, char *argv[])
{
    int reps = 100;
    int idx  = 0;
    if (argc > 1) reps = atoi(argv[1]);
    if (argc > 2) idx  = atoi(argv[2]);

    omp_set_num_threads(NTHREAD);
    int n_thrads = omp_get_num_threads();
    
    printf("AoS OpenMP CPU benchmark\n");
    printf("Volume:      %d\n", VOLUME);
    printf("Repetitions: %d\n", reps);
    printf("Threads:     %d\n", n_thrads);


    // -----------------------------------------------------------------------
    // Host fields
    // -----------------------------------------------------------------------
    su3_dble *u_fld = (su3_dble *)malloc(4*VOLUME * sizeof(su3_dble));
    double    *h_res = (double    *)malloc(VOLUME * sizeof(double));
    
    rlxd_init(1, 1, 1, 1);
    for (size_t i = 0; i <4*VOLUME; i++) {
        random_su3_dble(&u_fld[i]);
    }

    double *flush_buf = (double *)malloc(FLUSH_NELEMS * sizeof(double));

    // -----------------------------------------------------------------------
    // Warm-up
    // -----------------------------------------------------------------------
    for (int r = 0; r < 3; r++) {
        #pragma omp parallel for
        for (size_t i = 0; i < VOLUME; i++) {
            su3_dble temp, res;
            su3xsu3      (&temp, &u_fld[0*VOLUME+i], &u_fld[1*VOLUME+i]);
            su3dagxsu3dag(&res,  &u_fld[2*VOLUME+i], &u_fld[3*VOLUME+i]);
            h_res[i] = cm3x3_retr(&temp, &res);
        }
    }

    // -----------------------------------------------------------------------
    // Benchmark
    // -----------------------------------------------------------------------
    double total_s = 0.0;

    for (int r = 0; r < reps; r++) {
        #pragma omp parallel for
        for (size_t j = 0; j < FLUSH_NELEMS; j++)
            flush_buf[j] += 1.0;

        double t0 = omp_get_wtime();
        #pragma omp parallel for
        for (size_t i = 0; i < VOLUME; i++) {
            su3_dble temp, res;
            su3xsu3      (&temp, &u_fld[0*VOLUME+i], &u_fld[1*VOLUME+i]);
            su3dagxsu3dag(&res,  &u_fld[2*VOLUME+i], &u_fld[3*VOLUME+i]);
            h_res[i] = cm3x3_retr(&temp, &res);
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
    if (idx >= 0 && (size_t)idx < (size_t)VOLUME)
        printf("  res[%d] = %.10f\n", idx, h_res[idx]);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    free(flush_buf);
    free(u_fld); free(h_res);

    return 0;
}
