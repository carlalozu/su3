#include <stdio.h>
#include "su3.h"
#include "global.h"
#include "su3prod.c"

int main(int argc, char *argv[])
{
    printf("Hello, C project!\n");

    complex_dble u = {1.0, 0.0};
    printf("Complex number: re = %f, im = %f\n", u.re, u.im);
    complex_dble v = {0.0, 1.0};
    printf("Complex number: re = %f, im = %f\n", v.re, v.im);

    complex_dble result = add(u, v);
    printf("Addition result: re = %f, im = %f\n", result.re, result.im);

    return 0;
}
