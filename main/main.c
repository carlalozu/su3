#include <stdio.h>
#include <stdlib.h>
#include "su3.h"
#include "global.h"
#include "ufields.h"

int main(int argc, char *argv[])
{
    printf("Hello, C project!\n");

    su3_cdble *u, *v;
    // Initialize u as needed
    int ru = alloc_su3_cdble(&u);
    int rv = alloc_su3_cdble(&v);
    if (ru != 0 || rv != 0) {
        printf("Allocation failed\n");
        return -1;
    }
    unit_su3_cdble(u, 8.0);
    unit_su3_cdble(v, 8.0);

    // print volume
    printf("Volume: %d\n", VOLUME);
    su3_cdble *u_field[VOLUME];
    int r_uf = alloc_ufield(&u_field, VOLUME);
    if (r_uf != 0) {
        printf("Allocation of u_field failed\n");
        return -1;
    }
    
    for (int i = 0; i < VOLUME; i++) {
        unit_su3_cdble(u_field[i], i);
    }
    int idx = 17; // example index
    
    printf("u[%i]->c11 = (%f, %f)\n", idx, u_field[idx]->c11.re, u_field[idx]->c11.im);
    printf("u[%i]->c12 = (%f, %f)\n", idx, u_field[idx]->c12.re, u_field[idx]->c12.im);
    printf("u[%i]->c13 = (%f, %f)\n", idx, u_field[idx]->c13.re, u_field[idx]->c13.im);
    printf("u[%i]->c21 = (%f, %f)\n", idx, u_field[idx]->c21.re, u_field[idx]->c21.im);
    printf("u[%i]->c22 = (%f, %f)\n", idx, u_field[idx]->c22.re, u_field[idx]->c22.im);
    printf("u[%i]->c23 = (%f, %f)\n", idx, u_field[idx]->c23.re, u_field[idx]->c23.im);
    printf("u[%i]->c31 = (%f, %f)\n", idx, u_field[idx]->c31.re, u_field[idx]->c31.im);
    printf("u[%i]->c32 = (%f, %f)\n", idx, u_field[idx]->c32.re, u_field[idx]->c32.im);
    printf("u[%i]->c33 = (%f, %f)\n", idx, u_field[idx]->c33.re, u_field[idx]->c33.im);

    free(u);
    free(v);

    return 0;
}
