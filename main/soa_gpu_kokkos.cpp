#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include "global.h"
#include "su3v.h"
#include "su3prod.h"
#include "ufields.h"
#include "su3v_kokkos.hpp"

// Flush L2 cache: ~120 MB, large enough for all current NVIDIA GPUs.
static const size_t FLUSH_NELEMS = 15728640UL;

void launch_plaq_dble_kokkos(
    KokkosDoublev           *d_res,
    const KokkosSu3MatField *d_u,
    const KokkosSu3MatField *d_v,
    const KokkosSu3MatField *d_w,
    const KokkosSu3MatField *d_x,
    size_t volume)
{
    // Capture field descriptors (raw device pointers) by value
    su3_mat_field u = d_u->field;
    su3_mat_field v = d_v->field;
    su3_mat_field w = d_w->field;
    su3_mat_field x = d_x->field;
    double *res_base = d_res->data.data();

    Kokkos::parallel_for("plaq_dble", volume, KOKKOS_LAMBDA(const size_t i) {
        su3_mat_dble temp, res;
        fsu3matxsu3mat      (&temp, &u, &v, i);
        fsu3matdagxsu3matdag(&res,  &w, &x, i);
        res_base[i] = su3matdxsu3matd_retrace(&temp, &res);
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
        printf("SoA Kokkos kernel benchmark\n");
        printf("Volume:      %d\n", VOLUME);
        printf("Repetitions: %d\n", reps);

        // -------------------------------------------------------------------
        // Host fields
        // -------------------------------------------------------------------
        su3_mat_field h_u, h_v, h_w, h_x;
        doublev h_res;

        su3_mat_field_init(&h_u, VOLUME);
        su3_mat_field_init(&h_v, VOLUME);
        su3_mat_field_init(&h_w, VOLUME);
        su3_mat_field_init(&h_x, VOLUME);
        doublev_init(&h_res, VOLUME);

        random_su3mat_field(&h_u);
        random_su3mat_field(&h_v);
        random_su3mat_field(&h_w);
        random_su3mat_field(&h_x);

        // -------------------------------------------------------------------
        // Device fields
        // -------------------------------------------------------------------
        KokkosSu3MatField d_u, d_v, d_w, d_x;
        KokkosDoublev     d_res;

        su3_mat_field_kokkos_alloc(&d_u, VOLUME);
        su3_mat_field_kokkos_alloc(&d_v, VOLUME);
        su3_mat_field_kokkos_alloc(&d_w, VOLUME);
        su3_mat_field_kokkos_alloc(&d_x, VOLUME);
        doublev_kokkos_alloc(&d_res, VOLUME);

        su3_mat_field_kokkos_upload(&d_u, &h_u);
        su3_mat_field_kokkos_upload(&d_v, &h_v);
        su3_mat_field_kokkos_upload(&d_w, &h_w);
        su3_mat_field_kokkos_upload(&d_x, &h_x);

        // Flush buffer
        KokkosDoublev d_flush;
        doublev_kokkos_alloc(&d_flush, FLUSH_NELEMS);

        // -------------------------------------------------------------------
        // Warm-up
        // -------------------------------------------------------------------
        for (int r = 0; r < 3; r++)
            launch_plaq_dble_kokkos(&d_res, &d_u, &d_v, &d_w, &d_x, VOLUME);
        Kokkos::fence();

        // -------------------------------------------------------------------
        // Benchmark
        // -------------------------------------------------------------------
        double total_s = 0.0;

        for (int r = 0; r < reps; r++) {
            launch_flush_kokkos(&d_flush);
            Kokkos::fence();

            Kokkos::Timer timer;
            launch_plaq_dble_kokkos(&d_res, &d_u, &d_v, &d_w, &d_x, VOLUME);
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
        doublev_kokkos_download(&h_res, &d_res);
        if (idx >= 0 && (size_t)idx < (size_t)VOLUME)
            printf("  res[%d] = %.10f\n", idx, h_res.base[idx]);

        // -------------------------------------------------------------------
        // Cleanup
        // -------------------------------------------------------------------
        su3_mat_field_kokkos_free(&d_u);
        su3_mat_field_kokkos_free(&d_v);
        su3_mat_field_kokkos_free(&d_w);
        su3_mat_field_kokkos_free(&d_x);
        doublev_kokkos_free(&d_res);
        doublev_kokkos_free(&d_flush);

        su3_mat_field_free(&h_u);
        su3_mat_field_free(&h_v);
        su3_mat_field_free(&h_w);
        su3_mat_field_free(&h_x);
        free(h_res.base);
    }
    Kokkos::finalize();
    return 0;
}
