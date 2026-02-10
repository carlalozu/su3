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
#ifdef _OPENMP
    // Make OpenMP behavior predictable for benchmarking:
    int n_threads = omp_get_max_threads();
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

    
    size_t n_blocks = (VOLUME + CACHELINE - 1)/CACHELINE;
    
    prof_section init_AoS = {.name = "AoS init GPU", .threads = n_blocks};
    prof_section comp_AoS = {.name = "AoS compute GPU", .threads = n_blocks};
    // AoS
    su3_mat u_field[VOLUME];
    su3_mat v_field[VOLUME];
    su3_mat w_field[VOLUME];
    su3_mat x_field[VOLUME];
    double res_aos[VOLUME];

    prof_begin(&init_AoS);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < VOLUME; i++)
    {
        unit_su3mat(&u_field[i]);
        unit_su3mat(&v_field[i]);
        unit_su3mat(&w_field[i]);
        unit_su3mat(&x_field[i]);
    }
    prof_end(&init_AoS);
    #pragma omp target enter data map(to : v_field[0 : VOLUME], u_field[0 : VOLUME], w_field[0: VOLUME])
    #pragma omp target enter data map(to : res_aos[0 : VOLUME])
    
    prof_begin(&comp_AoS);
    #pragma omp target teams num_teams(n_blocks) 
    {
        su3_mat temp_field;
        su3_mat res_field;
        
        for (int r = 0; r < reps; r++)
        {
            #pragma omp distribute parallel for
            for (size_t i = 0; i < VOLUME; i++)
            {
                // if (r==0 && i==0) is_gpu();
                su3matxsu3mat(&temp_field, &u_field[i], &v_field[i]);
                su3matdagxsu3matdag(&res_field, &w_field, &x_field[i]);
                res_aos[i] = su3matxsu3mat_retrace(&temp_field, &res_field);
            }
        }
    }
    prof_end(&comp_AoS);
    comp_AoS.count *= reps;
    
    #pragma omp target update from(res_aos[0 : VOLUME])

    prof_report(&init_AoS);
    prof_report(&comp_AoS);
}
