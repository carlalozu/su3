#include <stdio.h>
#include <stdlib.h>
#include "su3v.h"
#include "global.h"
#include "ufields.h"
#include <time.h>
#include "profiler.h"
#include "utils.h"

int main(int argc, char *argv[])
{

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

    
    size_t n_blocks = (VOLUME + CACHELINE - 1)/CACHELINE;
    
    prof_section init_AoS = {.name = "AoS init_GPU", .threads = n_blocks};
    prof_section comp_AoS = {.name = "AoS compute_GPU", .threads = n_blocks};
    
    // AoS
    su3_mat *u_field = (su3_mat *)malloc(VOLUME * sizeof(su3_mat));
    su3_mat *v_field = (su3_mat *)malloc(VOLUME * sizeof(su3_mat));
    su3_mat *w_field = (su3_mat *)malloc(VOLUME * sizeof(su3_mat));
    su3_mat *x_field = (su3_mat *)malloc(VOLUME * sizeof(su3_mat));
    double *res_aos = (double *)malloc(VOLUME * sizeof(double));

    #pragma omp target enter data map(to : v_field[0:VOLUME], u_field[0:VOLUME], w_field[0:VOLUME], x_field[0:VOLUME])
    #pragma omp target enter data map(alloc : res_aos[0:VOLUME])

    
    prof_begin(&init_AoS);
    #pragma omp target teams distribute parallel for num_teams(n_blocks) 
    for (size_t i = 0; i < VOLUME; i++)
    {
        uint64_t thread_state = 12345ULL + i;
        random_su3mat(&u_field[i], &thread_state);
        random_su3mat(&v_field[i], &thread_state);
        random_su3mat(&w_field[i], &thread_state);
        random_su3mat(&x_field[i], &thread_state);
    }
    prof_end(&init_AoS);

    // geno 3145728 Byes L2 cache
    size_t flush_size = 3145728 / sizeof(double);
    double *flush_buf = malloc(flush_size * sizeof(double));
    #pragma omp target enter data map(alloc : flush_buf[0:flush_size])


    #pragma omp target teams distribute parallel for num_teams(n_blocks) 
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
        #pragma omp target teams distribute parallel for num_teams(n_blocks) 
        for (size_t j = 0; j < flush_size; j++) {
            flush_buf[j] += 1.0; 
        }

        prof_begin(&comp_AoS);
        #pragma omp target teams distribute parallel for num_teams(n_blocks) 
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
    
    #pragma omp target update from(res_aos[0:VOLUME])

    double total_sum = 0.0;
    for(size_t i = 0; i < VOLUME; i++) {
        total_sum += res_aos[i]/reps;
    }
        
    prof_report(&init_AoS);
    prof_report(&comp_AoS);

    printf("Average to prevent optimization: %f \n", total_sum/reps);
    
    #pragma omp target exit data map(release: u_field[0:VOLUME], v_field[0:VOLUME], w_field[0:VOLUME], x_field[0:VOLUME], res_aos[0:VOLUME])
    free(u_field); free(v_field); free(w_field); free(x_field); free(res_aos);
    return 0;
}
