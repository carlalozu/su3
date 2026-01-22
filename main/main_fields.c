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

    su3_mat u_field[VOLUME];
    su3_mat v_field[VOLUME];
    su3_mat w_field[VOLUME];

    for (size_t i = 0; i < VOLUME; i++) {
        random_su3mat(&u_field[i]);
        random_su3mat(&w_field[i]);
        random_su3mat(&v_field[i]);
    }
    printf("u[%i]->c22 = (%f, %f)\n", idx, u_field[idx].c22.re, u_field[idx].c22.im);
    printf("u[%i]->c31 = (%f, %f)\n", idx, u_field[idx].c31.re, u_field[idx].c31.im);
    printf("u[%i]->c32 = (%f, %f)\n", idx, u_field[idx].c32.re, u_field[idx].c32.im);
    printf("u[%i]->c33 = (%f, %f)\n", idx, u_field[idx].c33.re, u_field[idx].c33.im);

    usu3matxusu3mat(w_field, u_field, w_field, VOLUME);
    usu3matxusu3mat(w_field, w_field, v_field, VOLUME);

    printf("w[%i]->c11 = (%f, %f)\n", idx, w_field[idx].c11.re, w_field[idx].c11.im);
    printf("w[%i]->c12 = (%f, %f)\n", idx, w_field[idx].c12.re, w_field[idx].c12.im);
    printf("w[%i]->c13 = (%f, %f)\n", idx, w_field[idx].c13.re, w_field[idx].c13.im);

    // take trace of w_field[idx]
    complex tr = su3_trace(&w_field[idx]);
    printf("Trace of w[%i]: (%f, %f)\n", idx, tr.re, tr.im);
    return 0;
}
