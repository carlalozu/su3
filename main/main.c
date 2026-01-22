#include <stdio.h>
#include <stdlib.h>
#include "su3.h"
#include "global.h"

int main(int argc, char *argv[])
{
    printf("SU(3) algebra!\n");

    su3_mat u;
    su3_mat w;
    su3_vec v;

    unit_su3mat(&w);
    unit_su3mat(&u);
    unit_su3vec(&v);
    printf("v vector:\n");
    printf("c1 = (%f, %f)\n", v.c1.re, v.c1.im);
    printf("c2 = (%f, %f)\n", v.c2.re, v.c2.im);
    printf("c3 = (%f, %f)\n", v.c3.re, v.c3.im);

    // multiply u and v
    su3_vec result_vec;
    su3matxsu3vec(&result_vec, &u, &v);
    printf("result_vec vector:\n");
    printf("c1 = (%f, %f)\n", result_vec.c1.re, result_vec.c1.im);
    printf("c2 = (%f, %f)\n", result_vec.c2.re, result_vec.c2.im);
    printf("c3 = (%f, %f)\n", result_vec.c3.re, result_vec.c3.im);

    // multiply u and w
    su3_mat result_mat;
    su3matxsu3mat(&result_mat, &u, &w);
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

    // take trace of result_mat
    complex tr = su3_trace(&result_mat);
    printf("Trace of result matrix: (%f, %f)\n", tr.re, tr.im);

    return 0;
}
