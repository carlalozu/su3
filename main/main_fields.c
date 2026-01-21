#include <stdio.h>
#include <stdlib.h>
#include "su3.h"
#include "global.h"
#include "ufields.h"

int main(int argc, char *argv[])
{
    // read idx from command line
    int idx = 0;
    if (argc > 1) {
        idx = atoi(argv[1]);
    }

    printf("Testing ufields structures\n");
    printf("Volume: %d\n", VOLUME);

    su3_cdble u_field[VOLUME];
    su3_cdble w_field[VOLUME];
    su3_vector_cdble v_field[VOLUME];

    for (size_t i = 0; i < VOLUME; i++) {
        unit_su3_cdble(&u_field[i], i+1);
        unit_su3_cdble(&w_field[i], i+1);
        unit_su3_vector_cdble(&v_field[i], i+1);
    }
    printf("u[%i]->c22 = (%f, %f)\n", idx, u_field[idx].c22.re, u_field[idx].c22.im);
    printf("u[%i]->c31 = (%f, %f)\n", idx, u_field[idx].c31.re, u_field[idx].c31.im);
    printf("u[%i]->c32 = (%f, %f)\n", idx, u_field[idx].c32.re, u_field[idx].c32.im);
    printf("u[%i]->c33 = (%f, %f)\n", idx, u_field[idx].c33.re, u_field[idx].c33.im);

    usu3xusu3(w_field, u_field, w_field, VOLUME);
    usu3xusu3vec(v_field, u_field, v_field, VOLUME);

    printf("w[%i]->c11 = (%f, %f)\n", idx, w_field[idx].c11.re, w_field[idx].c11.im);
    printf("w[%i]->c12 = (%f, %f)\n", idx, w_field[idx].c12.re, w_field[idx].c12.im);
    printf("w[%i]->c13 = (%f, %f)\n", idx, w_field[idx].c13.re, w_field[idx].c13.im);
    printf("v[%i]->c21 = (%f, %f)\n", idx, v_field[idx].c1.re, v_field[idx].c1.im);
    printf("v[%i]->c22 = (%f, %f)\n", idx, v_field[idx].c2.re, v_field[idx].c2.im);

    return 0;
}
