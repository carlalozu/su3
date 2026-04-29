#ifndef SU3V_KOKKOS_HPP
#define SU3V_KOKKOS_HPP

#include <Kokkos_Core.hpp>
#include "su3v.h"

struct KokkosSu3MatField {
    Kokkos::View<double*> c1, c2, c3;
    su3_mat_field field;
};

struct KokkosDoublev {
    Kokkos::View<double*> data;
    doublev dv;
};

struct KokkosSu3Mat {
    Kokkos::View<su3_mat_c*> data;
    size_t volume = 0;
};

void su3_mat_field_kokkos_alloc(KokkosSu3MatField *kf, size_t volume);
void su3_mat_field_kokkos_free(KokkosSu3MatField *kf);
void su3_mat_field_kokkos_upload(KokkosSu3MatField *d, const su3_mat_field *h);
void su3_mat_field_kokkos_download(su3_mat_field *h, const KokkosSu3MatField *d);

void doublev_kokkos_alloc(KokkosDoublev *kd, size_t volume);
void doublev_kokkos_free(KokkosDoublev *kd);
void doublev_kokkos_download(doublev *h, const KokkosDoublev *d);

void su3_aos_kokkos_alloc(KokkosSu3Mat *km, size_t volume);
void su3_aos_kokkos_free(KokkosSu3Mat *km);
void su3_aos_kokkos_upload(KokkosSu3Mat *d, const su3_mat_c *h);
void su3_aos_kokkos_download(su3_mat_c *h, const KokkosSu3Mat *d);

void launch_flush_kokkos(KokkosDoublev *buf)
{
    double *ptr = buf->data.data();
    size_t  n   = buf->dv.volume;
    Kokkos::parallel_for("flush_cache", n, KOKKOS_LAMBDA(const size_t i) {
        ptr[i] += 1.0;
    });
    Kokkos::fence();
}

#endif // SU3V_KOKKOS_HPP
