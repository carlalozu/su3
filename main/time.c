#include <stdio.h>
#include <stdlib.h>
#include "su3v.h"
#include "global.h"
#include "ufields.h"
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char *argv[])
{
    double start_time, end_time;
    double init_AoS_time = 0.0;
    double compute_AoS_time = 0.0;
    double init_SoA_time = 0.0;
    double compute_SoA_time = 0.0;


    // print if openmp is enabled
    #ifdef _OPENMP
    printf("OpenMP is enabled\n");
    printf("Number of threads: %d\n", omp_get_max_threads());
    #else
    printf("OpenMP is not enabled\n");
    #endif

    // read reps from command line
    int reps = 100;
    int idx = 0;
    if (argc > 1)
    {
        reps = atoi(argv[1]);
    }
    if (argc > 2)
    {
        idx = atoi(argv[2]);
    }


    printf("Timing SoA vs AoS structures\n");
    printf("Volume: %d\n", VOLUME);

    // AoS
    su3_mat u_field[VOLUME];
    su3_mat v_field[VOLUME];
    su3_mat w_field[VOLUME];
    su3_mat temp_field[VOLUME];
    su3_mat res_field[VOLUME];
    complex res1[VOLUME];

    // SoA
    su3_mat_field u_fieldv;
    su3_mat_field v_fieldv;
    su3_mat_field w_fieldv;
    su3_mat_field temp_fieldv;
    su3_mat_field res_fieldv;
    complexv res2[VOLUME];

    for (int r = 0; r < reps; r++)
    {
        // initialize AoS fields
        start_time = (double)clock() / CLOCKS_PER_SEC;
        for (size_t i = 0; i < VOLUME; i++)
        {
            unit_su3mat(&u_field[i]);
            unit_su3mat(&v_field[i]);
            unit_su3mat(&w_field[i]);
        }
        end_time = (double)clock() / CLOCKS_PER_SEC;
        if (r>10) init_AoS_time += end_time - start_time;

        // u*v*w AoS
        start_time = (double)clock() / CLOCKS_PER_SEC;
        usu3matxusu3mat(temp_field, u_field, v_field, VOLUME);
        usu3matxusu3mat(res_field, temp_field, w_field, VOLUME);
        usu3mattrace(res1, res_field, VOLUME);
        end_time = (double)clock() / CLOCKS_PER_SEC;
        if (r>10) compute_AoS_time += end_time - start_time;

        // initialize SoA fields
        start_time = (double)clock() / CLOCKS_PER_SEC;
        unit_su3mat_field(&u_fieldv);
        unit_su3mat_field(&v_fieldv);
        unit_su3mat_field(&w_fieldv);
        end_time = (double)clock() / CLOCKS_PER_SEC;
        if (r>10) init_SoA_time += end_time - start_time;

        // u*v*w SoA
        start_time = (double)clock() / CLOCKS_PER_SEC;
        fsu3matxsu3mat(&temp_fieldv, &u_fieldv, &v_fieldv, VOLUME);
        fsu3matxsu3mat(&res_fieldv, &temp_fieldv, &w_fieldv, VOLUME);
        fsu3mattrace(res2, &res_fieldv, VOLUME);
        end_time = (double)clock() / CLOCKS_PER_SEC;
        if (r>10) compute_SoA_time += end_time - start_time;
    }
    printf("AoS initialization time: %f seconds\n", init_AoS_time/(double)reps);
    printf("AoS time for u*v*w: %f seconds\n", compute_AoS_time/(double)reps);
    printf("SoA initialization time: %f seconds\n", init_SoA_time/(double)reps);
    printf("SoA time for u*v*w: %f seconds\n", compute_SoA_time/(double)reps);

    // print some results
    printf("res1[%i] (re[%i], im[%i]) = (%f, %f) \n", idx, idx, idx, res1[idx].re, res1[idx].im);
    printf("res2[%i] (re[%i], im[%i]) = (%f, %f) \n", idx, idx, idx, res2->re[idx], res2->im[idx]);
    return 0;
}
