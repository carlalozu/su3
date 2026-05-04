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

void su3_mat_field_kokkos_alloc(KokkosSu3MatField *kf, size_t volume);
void su3_mat_field_kokkos_free(KokkosSu3MatField *kf);
void su3_mat_field_kokkos_upload(KokkosSu3MatField *d, const su3_mat_field *h);
void su3_mat_field_kokkos_download(su3_mat_field *h, const KokkosSu3MatField *d);

void doublev_kokkos_alloc(KokkosDoublev *kd, size_t volume);
void doublev_kokkos_free(KokkosDoublev *kd);
void doublev_kokkos_download(doublev *h, const KokkosDoublev *d);

void launch_plaq_dble_kokkos(
    KokkosDoublev           *d_res,
    const KokkosSu3MatField *d_u,
    const KokkosSu3MatField *d_v,
    const KokkosSu3MatField *d_w,
    const KokkosSu3MatField *d_x,
    size_t volume);

void launch_flush_kokkos(KokkosDoublev *buf);

#endif // SU3V_KOKKOS_HPP
