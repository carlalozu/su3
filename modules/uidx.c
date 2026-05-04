#include "global.h"
#include "lattice.h"

#define UIDX_C

#pragma omp declare target
int offset(int ix, int mu)
{
    /* Layout: [mu(4)] -> [lexicographical] */
    return mu * VOLUME + ix;
}
#pragma omp end declare target

#pragma omp declare target
void plaq_uidx(int mu, int nu, int ix, int *ip)
{
    int iy;

    ip[0] = offset(ix,mu);

    iy = iupT[mu][ix];
    ip[1] = offset(iy,nu);

    ip[2] = offset(ix,nu);

    iy = iupT[nu][ix];
    ip[3] = offset(iy,mu);
}
#pragma omp end declare target