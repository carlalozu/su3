#include <stdio.h>
#include <stdlib.h>
#include "su3.h"
#include "global.h"
#include "ufields.h"

int main(int argc, char *argv[])
{
    printf("Hello, C project!\n");

    su3_cdble *u;
    // Initialize u as needed
    int r = alloc_su3_cdble(&u);
    if (r != 0) {
        printf("Allocation failed\n");
        return -1;
    }
    unit_su3_cdble(u, 8.0);

    printf("u->c11 = (%f, %f)\n", u->c11.re, u->c11.im);
    printf("u->c12 = (%f, %f)\n", u->c12.re, u->c12.im);
    printf("u->c13 = (%f, %f)\n", u->c13.re, u->c13.im);
    printf("u->c21 = (%f, %f)\n", u->c21.re, u->c21.im);
    printf("u->c22 = (%f, %f)\n", u->c22.re, u->c22.im);
    printf("u->c23 = (%f, %f)\n", u->c23.re, u->c23.im);
    printf("u->c31 = (%f, %f)\n", u->c31.re, u->c31.im);
    printf("u->c32 = (%f, %f)\n", u->c32.re, u->c32.im);
    printf("u->c33 = (%f, %f)\n", u->c33.re, u->c33.im);

    free(u);

    return 0;
}
