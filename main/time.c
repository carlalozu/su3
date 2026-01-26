#include <stdio.h>
#include <stdlib.h>
#include "su3v.h"
#include "global.h"
#include "ufields.h"
#include <time.h>
#include "profiler.h"
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char *argv[])
{
#ifdef _OPENMP
    // Make OpenMP behavior predictable for benchmarking:
    printf("OpenMP is enabled\n");
    int n_threads = omp_get_max_threads();
    printf("Number of threads: %d\n", n_threads);
    printf("Number of size per thread: %d\n", VOLUME/n_threads);
    omp_set_dynamic(0); // no changing thread counts behind your back
    omp_set_nested(0);
// Optional: warm up the runtime once (thread team creation can cost time)
#pragma omp parallel
    { /* nothing */
    }
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

    prof_section init_AoS = {.name = "AoS init"};
    prof_section comp_AoS = {.name = "AoS compute"};
    prof_section init_SoA = {.name = "SoA init"};
    prof_section comp_SoA = {.name = "SoA compute"};

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
    complexv res2;

    su3_mat_field_init(&u_fieldv, VOLUME);
    su3_mat_field_init(&v_fieldv, VOLUME);
    su3_mat_field_init(&w_fieldv, VOLUME);
    su3_mat_field_init(&temp_fieldv, VOLUME);
    su3_mat_field_init(&res_fieldv, VOLUME);
    complexv_init(&res2, VOLUME);

    #pragma omp parallel
    {
        for (int r = 0; r < reps; r++)
        {
            #pragma omp single
            prof_begin(&init_AoS);
            #pragma omp for schedule(static)
            for (size_t i = 0; i < VOLUME; i++)
            {
                unit_su3mat(&u_field[i]);
                unit_su3mat(&v_field[i]);
                unit_su3mat(&w_field[i]);
            }
            #pragma omp single
            prof_end(&init_AoS);

            #pragma omp single
            prof_begin(&comp_AoS);
            usu3matxusu3mat(temp_field, u_field, v_field, VOLUME);
            usu3matxusu3mat(res_field, temp_field, w_field, VOLUME);
            usu3mattrace(res1, res_field, VOLUME);
            #pragma omp single
            prof_end(&comp_AoS);
        }
    }

    #pragma omp parallel
    {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        int begin = VOLUME/n_threads*tid;
        int end = VOLUME/n_threads*(tid+1);
        #else
        int begin = 0;
        int end = VOLUME;
        #endif

        for (int r = 0; r < reps; r++)
        {   
            #pragma omp single
            prof_begin(&init_SoA);
            unit_su3mat_field(&u_fieldv);
            unit_su3mat_field(&v_fieldv);
            unit_su3mat_field(&w_fieldv);
            #pragma omp single
            prof_end(&init_SoA);

            #pragma omp single
            prof_begin(&comp_SoA);
            fsu3matxsu3mat(&temp_fieldv, &u_fieldv, &v_fieldv, begin, end);
            fsu3matxsu3mat(&res_fieldv, &temp_fieldv, &w_fieldv, begin, end);
            fsu3mattrace(&res2, &res_fieldv, begin, end);
            #pragma omp single
            prof_end(&comp_SoA);
        }
    }

    prof_report(&init_AoS);
    prof_report(&comp_AoS);
    prof_report(&init_SoA);
    prof_report(&comp_SoA);

    printf("res1[%i] (re[%i], im[%i]) = (%f, %f) \n", idx, idx, idx, res1[idx].re, res1[idx].im);
    printf("res2[%i] (re[%i], im[%i]) = (%f, %f) \n", idx, idx, idx, res2.re[idx], res2.im[idx]);

    su3_mat_field_free(&u_fieldv);
    su3_mat_field_free(&v_fieldv);
    su3_mat_field_free(&w_fieldv);
    su3_mat_field_free(&temp_fieldv);
    su3_mat_field_free(&res_fieldv);
    complexv_free(&res2);
}
