#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "global.h"
#include "su3prod.h"
#include "lattice.h"
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
    
    printf("\nplaq_dble AoS OpenMP CPU benchmark\n");
    printf("------------------------------------------\n");
    printf("Volume:      %d\n", VOLUME);
    printf("Repetitions: %d\n", reps);
    printf("OpenMP threads: %d\n", NTHREAD);
    printf("Data structure: AoS\n");
    printf("Lattice geometry: %ix%ix%ix%i\n", L0,L1,L2,L3);
    printf("Local lattice geometry: %ix%ix%ix%i\n\n", L0_TRD,L1_TRD,L2_TRD,L3_TRD);

    // -----------------------------------------------------------------------
    // Host fields
    // -----------------------------------------------------------------------
    start_ranlux(0, 12345);
    geometry();
    random_ud();

    double    *h_res = (double  *)malloc(VOLUME * sizeof(double));
    su3_dble  *u_fld = udfld();    

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

    printf("\nResults\n");
    printf("Local gauge field size (KB): %d\n",
           (int)(72 * VOLUME * sizeof(double) / 1024));
    printf("  total  = %.6f s  (%d reps)\n", total_s, reps);
    printf("  avg    = %.6f s  (%.3f ms)\n", avg_s, avg_s * 1e3);
    printf("  GFLOP/s = %.2f\n", gflops);
    printf("Time per lattice point (sec): %.9f\n", avg_s / (double)VOLUME);

    // -----------------------------------------------------------------------
    // Verify one element
    // -----------------------------------------------------------------------
    if (idx >= 0 && (size_t)idx < (size_t)VOLUME)
        printf("  res[%d] = %.10f\n", idx, h_res[idx]);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    free(flush_buf);
    free(h_res);

    return 0;
}
