#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include "global.h"
#include "su3prod.h"
#include "ufields.h"
#include "su3v_kokkos.hpp"

static const size_t FLUSH_NELEMS = 15728640UL;

void launch_plaq_aos_kokkos(
    KokkosDoublev      *d_res,
    const KokkosSu3Mat *d_fld,
    size_t volume)
{
    const su3_mat_c *fld = d_fld->data.data();
    double          *res = d_res->data.data();

    Kokkos::parallel_for("plaq_aos", volume, KOKKOS_LAMBDA(const size_t i) {
        su3_mat_c tmp_a, tmp_b;
        su3xsu3      (&tmp_a, &fld[0*volume+i], &fld[1*volume+i]);
        su3dagxsu3dag(&tmp_b, &fld[2*volume+i], &fld[3*volume+i]);
        res[i] = cm3x3_retr(&tmp_a, &tmp_b);
    });
    Kokkos::fence();
}

int main(int argc, char *argv[])
{
    int reps = 100;
    int idx  = 0;
    if (argc > 1) reps = atoi(argv[1]);
    if (argc > 2) idx  = atoi(argv[2]);

    Kokkos::initialize(argc, argv);
    {
        printf("AoS Kokkos kernel benchmark\n");
        printf("Volume:      %d\n", VOLUME);
        printf("Repetitions: %d\n", reps);

        // -------------------------------------------------------------------
        // Host fields
        // -------------------------------------------------------------------
        su3_mat_c *h_fld = (su3_mat_c *)malloc(4*VOLUME * sizeof(su3_mat_c));
        double    *h_res = (double    *)malloc(VOLUME * sizeof(double));

        for (size_t i = 0; i < 4*(size_t)VOLUME; i++) {
            uint64_t state = 12345ULL + i;
            random_su3mat(&h_fld[i], &state);
        }

        // -------------------------------------------------------------------
        // Device fields
        // -------------------------------------------------------------------
        KokkosSu3Mat  d_fld;
        KokkosDoublev d_res;

        su3_aos_kokkos_alloc(&d_fld, 4*VOLUME);
        doublev_kokkos_alloc(&d_res, VOLUME);

        su3_aos_kokkos_upload(&d_fld, h_fld);

        // Flush buffer
        KokkosDoublev d_flush;
        doublev_kokkos_alloc(&d_flush, FLUSH_NELEMS);

        // -------------------------------------------------------------------
        // Warm-up
        // -------------------------------------------------------------------
        for (int r = 0; r < 3; r++)
            launch_plaq_aos_kokkos(&d_res, &d_fld, VOLUME);
        Kokkos::fence();

        // -------------------------------------------------------------------
        // Benchmark
        // -------------------------------------------------------------------
        double total_s = 0.0;

        for (int r = 0; r < reps; r++) {
            launch_flush_kokkos(&d_flush);
            Kokkos::fence();

            Kokkos::Timer timer;
            launch_plaq_aos_kokkos(&d_res, &d_fld, VOLUME);
            Kokkos::fence();
            total_s += timer.seconds();
        }

        double avg_s  = total_s / reps;
        double gflops = (double)VOLUME * 432.0 / avg_s * 1e-9;
        double gbytes = (double)VOLUME * 1160.0;

        printf("\nResults\n");
        printf("  total  = %.6f s  (%d reps)\n", total_s, reps);
        printf("  avg    = %.6f s  (%.3f ms)\n", avg_s, avg_s * 1e3);
        printf("  GFLOP/s = %.2f\n", gflops);
        printf("  GB     = %.2f\n", gbytes);

        // -------------------------------------------------------------------
        // Verify one element
        // -------------------------------------------------------------------
        doublev tmp_dv = { (size_t)VOLUME, h_res};
        doublev_kokkos_download(&tmp_dv, &d_res);
        if (idx >= 0 && (size_t)idx < (size_t)VOLUME)
            printf("  res[%d] = %.10f\n", idx, h_res[idx]);

        // -------------------------------------------------------------------
        // Cleanup
        // -------------------------------------------------------------------
        su3_aos_kokkos_free(&d_fld);
        doublev_kokkos_free(&d_res);
        doublev_kokkos_free(&d_flush);

        free(h_fld); free(h_res);
    }
    Kokkos::finalize();
    return 0;
}
