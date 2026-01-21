#include <stdio.h>
#include <stdlib.h>
#include "su3.h"
#include "global.h"

int main(int argc, char *argv[])
{
    printf("Hello, C project!\n");

    su3_cdble u;
    su3_cdble w;
    su3_vector_cdble v;

    unit_su3_cdble(&w, 1.0);
    unit_su3_cdble(&u, 2.0);
    unit_su3_vector_cdble(&v, 3.0);

    // multiply u and v
    su3_vector_cdble result_vec;
    su3xsu3vec(&result_vec, &u, &v);
    printf("result_vec vector:\n");
    printf("c1 = (%f, %f)\n", result_vec.c1.re, result_vec.c1.im);
    printf("c2 = (%f, %f)\n", result_vec.c2.re, result_vec.c2.im);
    printf("c3 = (%f, %f)\n", result_vec.c3.re, result_vec.c3.im);

    // multiply u and w
    su3_cdble result_mat;
    su3xsu3(&result_mat, &u, &w);
    printf("result matrix:\n");
    printf("c11 = (%f, %f)\n", result_mat.c11.re, result_mat.c11.im);
    printf("c12 = (%f, %f)\n", result_mat.c12.re, result_mat.c12.im);
    printf("c13 = (%f, %f)\n", result_mat.c13.re, result_mat.c13.im);
    printf("c21 = (%f, %f)\n", result_mat.c21.re, result_mat.c21.im);
    printf("c22 = (%f, %f)\n", result_mat.c22.re, result_mat.c22.im);
    printf("c23 = (%f, %f)\n", result_mat.c23.re, result_mat.c23.im);
    printf("c31 = (%f, %f)\n", result_mat.c31.re, result_mat.c31.im);
    printf("c32 = (%f, %f)\n", result_mat.c32.re, result_mat.c32.im);
    printf("c33 = (%f, %f)\n", result_mat.c33.re, result_mat.c33.im);


    return 0;
}
