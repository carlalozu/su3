#include <stdio.h>
#include <stdlib.h>
#include "su3v.h"
#include "global.h"
#include "ufields.h"
#include <time.h>
#include "profiler.h"

int main(int argc, char *argv[])
{
#ifdef _OPENMP
    // Make OpenMP behavior predictable for benchmarking:
    printf("OpenMP is enabled\n");
    int n_threads = omp_get_max_threads();
    printf("Number of threads: %d\n", n_threads);
    printf("Number of size per thread: %d\n", VOLUME/n_threads);
    omp_set_dynamic(0); 
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
    printf("Cacheline: %d\n", CACHELINE);

    prof_section init_AoS = {.name = "AoS init"};
    prof_section comp_AoS = {.name = "AoS compute"};
    prof_section init_SoA = {.name = "SoA init"};
    prof_section comp_SoA = {.name = "SoA compute"};
    prof_section init_AoSoA = {.name = "AoSoA init"};
    prof_section comp_AoSoA = {.name = "AoSoA compute"};

    // AoS
    su3_mat u_field[VOLUME];
    su3_mat v_field[VOLUME];
    su3_mat w_field[VOLUME];
    complex res_aos[VOLUME];

    // SoA
    su3_mat_field u_fieldv;
    su3_mat_field v_fieldv;
    su3_mat_field w_fieldv;
    su3_mat_field temp_fieldv;
    su3_mat_field res_fieldv;
    complexv res_soa;

    su3_mat_field_init(&u_fieldv, VOLUME);
    su3_mat_field_init(&v_fieldv, VOLUME);
    su3_mat_field_init(&w_fieldv, VOLUME);
    su3_mat_field_init(&temp_fieldv, VOLUME);
    su3_mat_field_init(&res_fieldv, VOLUME);
    complexv_init(&res_soa, VOLUME);

    // AoSoA
    int n_blocks = VOLUME/CACHELINE;
    su3_mat_field u_fieldva[n_blocks];
    su3_mat_field v_fieldva[n_blocks];
    su3_mat_field w_fieldva[n_blocks];
    complexv res_aosoa[n_blocks];

    for (size_t i = 0; i < n_blocks; i++)
    {
        su3_mat_field_init(&u_fieldva[i], CACHELINE);
        su3_mat_field_init(&v_fieldva[i], CACHELINE);
        su3_mat_field_init(&w_fieldva[i], CACHELINE);
        complexv_init(&res_aosoa[i], CACHELINE);
    }

    // AoS
    #pragma omp parallel
    {
        su3_mat temp_field;
        su3_mat res_field;
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
            #pragma omp for schedule(static)
            for (size_t i = 0; i < VOLUME; i++)
            {
                su3matxsu3mat(&temp_field, &u_field[i], &v_field[i]);
                su3matxsu3mat(&res_field, &temp_field, &w_field[i]);
                res_aos[i] = su3_trace(&res_field);
            }
            #pragma omp single
            prof_end(&comp_AoS);
        }
    }

    // SoA
    #pragma omp parallel
    {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        int begin = VOLUME/n_threads*tid;
        int end = VOLUME/n_threads*(tid+1);
        if (end > VOLUME) end = VOLUME;
        #else
        int begin = 0;
        int end = VOLUME;
        #endif

        for (int r = 0; r < reps; r++)
        {   
            #pragma omp single
            {
            prof_begin(&init_SoA);
            unit_su3mat_field(&u_fieldv);
            unit_su3mat_field(&v_fieldv);
            unit_su3mat_field(&w_fieldv);
            prof_end(&init_SoA);
            }

            #pragma omp single
            prof_begin(&comp_SoA);
            #pragma clang loop vectorize(enable)
            for (size_t i=begin; i<end; i++)
                fsu3matxsu3mat(&temp_fieldv, &u_fieldv, &v_fieldv, i);
            #pragma clang loop vectorize(enable)
            for (size_t i=begin; i<end; i++)
                fsu3matxsu3mat(&res_fieldv, &temp_fieldv, &w_fieldv, i);
            #pragma clang loop vectorize(enable)
            for (size_t i=begin; i<end; i++)
                fsu3mattrace(&res_soa, &res_fieldv, i);
            #pragma omp single
            prof_end(&comp_SoA);
        }
    }

    // AoSoA
    #pragma omp parallel
    {
        su3_mat_field temp_fieldva;
        su3_mat_field res_fieldva;

        su3_mat_field_init(&temp_fieldva, CACHELINE);
        su3_mat_field_init(&res_fieldva, CACHELINE);

        for (int r = 0; r < reps; r++)
        {
            #pragma omp single
            prof_begin(&init_AoSoA);
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n_blocks; i++)
            {
                unit_su3mat_field(&u_fieldva[i]);
                unit_su3mat_field(&v_fieldva[i]);
                unit_su3mat_field(&w_fieldva[i]);
            }
            #pragma omp single
            prof_end(&init_AoSoA);

            #pragma omp single
            prof_begin(&comp_AoSoA);
            #pragma omp for schedule(static)
            for (size_t b = 0; b < n_blocks; b++)
            {
                #pragma clang loop vectorize(enable)
                for (size_t i=0; i<CACHELINE; i++)
                    fsu3matxsu3mat(&temp_fieldva, &u_fieldva[b], &v_fieldva[b], i);
                #pragma clang loop vectorize(enable)
                for (size_t i=0; i<CACHELINE; i++)
                    fsu3matxsu3mat(&res_fieldva, &temp_fieldva, &w_fieldva[b], i);
                #pragma clang loop vectorize(enable)
                for (size_t i=0; i<CACHELINE; i++)
                    fsu3mattrace(&res_aosoa[b], &res_fieldva, i);
            }
            #pragma omp single
            prof_end(&comp_AoSoA);
        }
    }

    printf("\n Init \n");
    prof_report(&init_AoS);
    prof_report(&init_SoA);
    prof_report(&init_AoSoA);

    printf("\n Compute \n");
    prof_report(&comp_AoS);
    prof_report(&comp_SoA);
    prof_report(&comp_AoSoA);

    int idx_a = idx/CACHELINE;
    int idx_b = idx%CACHELINE;
    printf("res_aos[%i] (re[%i], im[%i]) = (%f, %f) \n", idx, idx, idx, res_aos[idx].re, res_aos[idx].im);
    printf("res_soa[%i] (re[%i], im[%i]) = (%f, %f) \n", idx, idx, idx, res_soa.re[idx], res_soa.im[idx]);
    printf("res_aosoa[%i] (re[%i], im[%i]) = (%f, %f) \n", idx, idx, idx, res_aosoa[idx_a].re[idx_b], res_aosoa[idx_a].im[idx_b]);

}
