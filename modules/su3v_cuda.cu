#include "su3v_cuda.cuh"

// ---------------------------------------------------------------------------
// Device-side memory management for su3_mat_field
// ---------------------------------------------------------------------------

void su3_mat_field_cuda_alloc(su3_mat_field *d, size_t volume)
{
    size_t padded = (volume + 7) & ~(size_t)7;
    size_t bytes  = 6 * padded * sizeof(double);

    auto alloc_vec = [&](su3_vec_field *vf) {
        vf->volume = padded;
        CUDA_CHECK(cudaMalloc(&vf->base, bytes));
        vf->c1re = vf->base + 0 * padded;
        vf->c1im = vf->base + 1 * padded;
        vf->c2re = vf->base + 2 * padded;
        vf->c2im = vf->base + 3 * padded;
        vf->c3re = vf->base + 4 * padded;
        vf->c3im = vf->base + 5 * padded;
    };

    alloc_vec(&d->c1);
    alloc_vec(&d->c2);
    alloc_vec(&d->c3);
}

void su3_mat_field_cuda_free(su3_mat_field *d)
{
    CUDA_CHECK(cudaFree(d->c1.base));
    CUDA_CHECK(cudaFree(d->c2.base));
    CUDA_CHECK(cudaFree(d->c3.base));
    d->c1.base = d->c2.base = d->c3.base = nullptr;
}

void su3_mat_field_cuda_upload(su3_mat_field *d, const su3_mat_field *h)
{
    size_t bytes = 6 * d->c1.volume * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d->c1.base, h->c1.base, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->c2.base, h->c2.base, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d->c3.base, h->c3.base, bytes, cudaMemcpyHostToDevice));
}

void su3_mat_field_cuda_download(su3_mat_field *h, const su3_mat_field *d)
{
    size_t bytes = 6 * d->c1.volume * sizeof(double);
    CUDA_CHECK(cudaMemcpy(h->c1.base, d->c1.base, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h->c2.base, d->c2.base, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h->c3.base, d->c3.base, bytes, cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Device-side memory management for doublev
// ---------------------------------------------------------------------------

void doublev_cuda_alloc(doublev *d, size_t volume)
{
    size_t padded = (volume + 7) & ~(size_t)7;
    d->volume = padded;
    CUDA_CHECK(cudaMalloc(&d->base, padded * sizeof(double)));
}

void doublev_cuda_free(doublev *d)
{
    CUDA_CHECK(cudaFree(d->base));
    d->base   = nullptr;
    d->volume = 0;
}

void doublev_cuda_download(doublev *h, const doublev *d)
{
    size_t bytes = d->volume * sizeof(double);
    CUDA_CHECK(cudaMemcpy(h->base, d->base, bytes, cudaMemcpyDeviceToHost));
}
