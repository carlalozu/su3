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
    int n_threads = omp_get_max_threads();
    omp_set_dynamic(0);
#else
    printf("OpenMP is not enabled\n");
    int n_threads = 1;
#endif

    int reps = 100;
    if (argc > 1) reps = atoi(argv[1]);

    prof_section init_AoS = {.name = "AoS init", .threads = n_threads};
    prof_section comp_AoS = {.name = "AoS compute", .threads = n_threads};

    // Pro-Tip: Consider aligned_alloc(64, VOLUME * sizeof(su3_mat)) for SIMD performance
    su3_mat *u_field = (su3_mat *)malloc(VOLUME * sizeof(su3_mat));
    su3_mat *v_field = (su3_mat *)malloc(VOLUME * sizeof(su3_mat));
    su3_mat *w_field = (su3_mat *)malloc(VOLUME * sizeof(su3_mat));
    su3_mat *x_field = (su3_mat *)malloc(VOLUME * sizeof(su3_mat));
    float *res_aos = (float *)malloc(VOLUME * sizeof(float));

    prof_begin(&init_AoS);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < VOLUME; i++)
    {
        uint64_t thread_state = 12345ULL + omp_get_thread_num();
        random_su3mat(&u_field[i], &thread_state);
        random_su3mat(&v_field[i], &thread_state);
        random_su3mat(&w_field[i], &thread_state);
        random_su3mat(&x_field[i], &thread_state);
    }
    prof_end(&init_AoS);


    // geno 64MiB L3 cache
    size_t flush_size = 128 * 1024 * 1024 / sizeof(double);
    double *flush_buf = malloc(flush_size * sizeof(double));
    
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < flush_size; j++) {
        flush_buf[j] += 1.0; 
    }

    // warm-up run
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < VOLUME; i++)
    {

        su3_mat temp_field;
        su3_mat res_field; 
        su3matxsu3mat(&temp_field, &u_field[i], &v_field[i]);
        su3matdagxsu3matdag(&res_field, &w_field[i], &x_field[i]);
        res_aos[i] = su3matxsu3mat_retrace(&temp_field, &res_field);
    }

    for (int r = 0; r < reps; r++) 
    {
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < flush_size; j++) {
            flush_buf[j] += 1.0; 
        }

        prof_begin(&comp_AoS);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < VOLUME; i++)
        {
            su3_mat temp_field; 
            su3_mat res_field;  
            su3matxsu3mat(&temp_field, &u_field[i], &v_field[i]);
            su3matdagxsu3matdag(&res_field, &w_field[i], &x_field[i]);
            res_aos[i] += su3matxsu3mat_retrace(&temp_field, &res_field);
        }
        prof_end(&comp_AoS);
    }

    // Sum the array to prevent dead-code elimination by the compiler
    double total_sum = 0.0;
    for(size_t i = 0; i < VOLUME; i++) {
        total_sum += res_aos[i];
    }

    prof_report(&init_AoS);
    prof_report(&comp_AoS);
    
    printf("Average to prevent optimization: %f \n", total_sum/reps);

    free(u_field); free(v_field); free(w_field); free(x_field); free(res_aos);
    return 0;
}