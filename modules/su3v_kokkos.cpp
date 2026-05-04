#include "su3v_kokkos.hpp"
#include "ufields.h"

static size_t pad(size_t v) { return (v + 7) & ~(size_t)7; }

static void wire_vec_pointers(su3_vec_field *vf, double *base, size_t p)
{
    vf->volume = p;
    vf->base   = base;
    vf->c1re   = base + 0 * p;
    vf->c1im   = base + 1 * p;
    vf->c2re   = base + 2 * p;
    vf->c2im   = base + 3 * p;
    vf->c3re   = base + 4 * p;
    vf->c3im   = base + 5 * p;
}

// ---------------------------------------------------------------------------
// su3_mat_field Kokkos memory management
// ---------------------------------------------------------------------------

void su3_mat_field_kokkos_alloc(KokkosSu3MatField *kf, size_t volume)
{
    size_t p = pad(volume);
    size_t n = 6 * p;
    kf->c1 = Kokkos::View<double*>("su3mf_c1", n);
    kf->c2 = Kokkos::View<double*>("su3mf_c2", n);
    kf->c3 = Kokkos::View<double*>("su3mf_c3", n);
    wire_vec_pointers(&kf->field.c1, kf->c1.data(), p);
    wire_vec_pointers(&kf->field.c2, kf->c2.data(), p);
    wire_vec_pointers(&kf->field.c3, kf->c3.data(), p);
}

void su3_mat_field_kokkos_free(KokkosSu3MatField *kf)
{
    kf->c1    = Kokkos::View<double*>();
    kf->c2    = Kokkos::View<double*>();
    kf->c3    = Kokkos::View<double*>();
    kf->field = su3_mat_field{};
}

void su3_mat_field_kokkos_upload(KokkosSu3MatField *d, const su3_mat_field *h)
{
    size_t n = 6 * d->field.c1.volume;
    using HV = Kokkos::View<const double*, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::deep_copy(d->c1, HV(h->c1.base, n));
    Kokkos::deep_copy(d->c2, HV(h->c2.base, n));
    Kokkos::deep_copy(d->c3, HV(h->c3.base, n));
}

void su3_mat_field_kokkos_download(su3_mat_field *h, const KokkosSu3MatField *d)
{
    size_t n = 6 * d->field.c1.volume;
    using HV = Kokkos::View<double*, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::deep_copy(HV(h->c1.base, n), d->c1);
    Kokkos::deep_copy(HV(h->c2.base, n), d->c2);
    Kokkos::deep_copy(HV(h->c3.base, n), d->c3);
}

// ---------------------------------------------------------------------------
// doublev Kokkos memory management
// ---------------------------------------------------------------------------

void doublev_kokkos_alloc(KokkosDoublev *kd, size_t volume)
{
    size_t p      = pad(volume);
    kd->data      = Kokkos::View<double*>("doublev", p);
    kd->dv.volume = p;
    kd->dv.base   = kd->data.data();
}

void doublev_kokkos_free(KokkosDoublev *kd)
{
    kd->data = Kokkos::View<double*>();
    kd->dv   = doublev{};
}

void doublev_kokkos_download(doublev *h, const KokkosDoublev *d)
{
    using HV = Kokkos::View<double*, Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::deep_copy(HV(h->base, d->dv.volume), d->data);
}

// ---------------------------------------------------------------------------
// Cache flush kernel
// ---------------------------------------------------------------------------

void launch_flush_kokkos(KokkosDoublev *buf)
{
    double *ptr = buf->data.data();
    size_t  n   = buf->dv.volume;
    Kokkos::parallel_for("flush_cache", n, KOKKOS_LAMBDA(const size_t i) {
        ptr[i] += 1.0;
    });
    Kokkos::fence();
}

// ---------------------------------------------------------------------------
// Plaquette kernel: res[i] = Re Tr( (u*v) * (w†*x†) )
// ---------------------------------------------------------------------------

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
