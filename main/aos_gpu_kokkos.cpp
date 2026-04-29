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
    KokkosDoublev       *d_res,
    const KokkosSu3Mat  *d_u, const KokkosSu3Mat *d_v,
    const KokkosSu3Mat  *d_w, const KokkosSu3Mat *d_x,
    size_t volume)
{
    const su3_mat_c *u   = d_u->data.data();
    const su3_mat_c *v   = d_v->data.data();
    const su3_mat_c *w   = d_w->data.data();
    const su3_mat_c *x   = d_x->data.data();
    double          *res = d_res->data.data();

    Kokkos::parallel_for("plaq_aos", volume, KOKKOS_LAMBDA(const size_t i) {
        su3_mat_c tmp_a, tmp_b;
        su3matxsu3mat(&tmp_a, &u[i], &v[i]);
        su3matdagxsu3matdag(&tmp_b, &w[i], &x[i]);
        res[i] = su3matxsu3mat_retrace(&tmp_a, &tmp_b);
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
        su3_mat_c *h_u   = (su3_mat_c *)malloc(VOLUME * sizeof(su3_mat_c));
        su3_mat_c *h_v   = (su3_mat_c *)malloc(VOLUME * sizeof(su3_mat_c));
        su3_mat_c *h_w   = (su3_mat_c *)malloc(VOLUME * sizeof(su3_mat_c));
        su3_mat_c *h_x   = (su3_mat_c *)malloc(VOLUME * sizeof(su3_mat_c));
        double    *h_res = (double    *)malloc(VOLUME * sizeof(double));

        for (size_t i = 0; i < (size_t)VOLUME; i++) {
            uint64_t state = 12345ULL + i;
            random_su3mat(&h_u[i], &state);
            random_su3mat(&h_v[i], &state);
            random_su3mat(&h_w[i], &state);
            random_su3mat(&h_x[i], &state);
        }

        // -------------------------------------------------------------------
        // Device fields
        // -------------------------------------------------------------------
        KokkosSu3Mat  d_u, d_v, d_w, d_x;
        KokkosDoublev d_res;

        su3_aos_kokkos_alloc(&d_u, VOLUME);
        su3_aos_kokkos_alloc(&d_v, VOLUME);
        su3_aos_kokkos_alloc(&d_w, VOLUME);
        su3_aos_kokkos_alloc(&d_x, VOLUME);
        doublev_kokkos_alloc(&d_res, VOLUME);

        su3_aos_kokkos_upload(&d_u, h_u);
        su3_aos_kokkos_upload(&d_v, h_v);
        su3_aos_kokkos_upload(&d_w, h_w);
        su3_aos_kokkos_upload(&d_x, h_x);

        // Flush buffer
        KokkosDoublev d_flush;
        doublev_kokkos_alloc(&d_flush, FLUSH_NELEMS);

        // -------------------------------------------------------------------
        // Warm-up
        // -------------------------------------------------------------------
        for (int r = 0; r < 3; r++)
            launch_plaq_aos_kokkos(&d_res, &d_u, &d_v, &d_w, &d_x, VOLUME);
        Kokkos::fence();

        // -------------------------------------------------------------------
        // Benchmark
        // -------------------------------------------------------------------
        double total_s = 0.0;

        for (int r = 0; r < reps; r++) {
            launch_flush_kokkos(&d_flush);
            Kokkos::fence();

            Kokkos::Timer timer;
            launch_plaq_aos_kokkos(&d_res, &d_u, &d_v, &d_w, &d_x, VOLUME);
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
        su3_aos_kokkos_free(&d_u);
        su3_aos_kokkos_free(&d_v);
        su3_aos_kokkos_free(&d_w);
        su3_aos_kokkos_free(&d_x);
        doublev_kokkos_free(&d_res);
        doublev_kokkos_free(&d_flush);

        free(h_u); free(h_v); free(h_w); free(h_x); free(h_res);
    }
    Kokkos::finalize();
    return 0;
}
