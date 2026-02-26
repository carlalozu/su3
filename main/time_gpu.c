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
    printf("OpenMP is enabled\n");
    int n_threads = omp_get_max_threads();
    printf("Number of threads: %d\n", n_threads);
    printf("Number of size per thread: %d\n", VOLUME/n_threads);
    omp_set_dynamic(0); 
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

    prof_section init_AoS = {.name = "AoS init GPU"};
    prof_section comp_AoS = {.name = "AoS compute GPU"};
    prof_section init_SoA = {.name = "SoA init GPU"};
    prof_section comp_SoA = {.name = "SoA compute GPU"};
    prof_section init_AoSoA = {.name = "AoSoA init GPU"};
    prof_section comp_AoSoA = {.name = "AoSoA compute GPU"};

    size_t n_blocks = (VOLUME + CACHELINE - 1)/CACHELINE;
    printf("n_blocks: %zu\n", n_blocks);

    // AoS
    su3_mat u_field[VOLUME];
    su3_mat v_field[VOLUME];
    su3_mat w_field[VOLUME];
    su3_mat x_field[VOLUME];
    float res_aos[VOLUME];

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
    
    prof_begin(&comp_AoS);
    #pragma omp target teams \
    map(to : v_field[0 : VOLUME], u_field[0 : VOLUME], w_field[0: VOLUME]) \
    map(from : res_aos[0 : VOLUME]) num_teams(n_blocks) 
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

    // SoA
    su3_mat_field *u_fieldv = (su3_mat_field*)malloc(sizeof(su3_mat_field));
    su3_mat_field *v_fieldv = (su3_mat_field*)malloc(sizeof(su3_mat_field));
    su3_mat_field *w_fieldv = (su3_mat_field*)malloc(sizeof(su3_mat_field));
    su3_mat_field *x_fieldv = (su3_mat_field*)malloc(sizeof(su3_mat_field));
    su3_mat_field *temp_fieldv = (su3_mat_field*)malloc(sizeof(su3_mat_field));
    su3_mat_field *res_fieldv = (su3_mat_field*)malloc(sizeof(su3_mat_field));
    floatv *res_soa  = (floatv*)malloc(sizeof(floatv));

    su3_mat_field_init(u_fieldv, VOLUME);
    su3_mat_field_init(v_fieldv, VOLUME);
    su3_mat_field_init(w_fieldv, VOLUME);
    su3_mat_field_init(x_fieldv, VOLUME);
    su3_mat_field_init(temp_fieldv, VOLUME);
    su3_mat_field_init(res_fieldv, VOLUME);
    floatv_init(res_soa, VOLUME);

    prof_begin(&init_SoA);
    unit_su3mat_field(u_fieldv);
    unit_su3mat_field(v_fieldv);
    unit_su3mat_field(w_fieldv);
    unit_su3mat_field(x_fieldv);
    prof_end(&init_SoA);

    enter_su3_mat_field(u_fieldv);
    enter_su3_mat_field(v_fieldv);    
    enter_su3_mat_field(w_fieldv);
    enter_su3_mat_field(x_fieldv);
    enter_su3_mat_field(temp_fieldv);
    enter_su3_mat_field(res_fieldv);
    enter_float_field(res_soa);

    prof_begin(&comp_SoA);
    #pragma omp target teams num_teams(n_blocks)
    {   
        for (int r = 0; r < reps; r++)
        {
            #pragma omp distribute parallel for
            for (size_t i=0; i<VOLUME; i++)
            {
                // if (r==0 && i==0) is_gpu();
                fsu3matxsu3mat(temp_fieldv, u_fieldv, v_fieldv, i);
                fsu3matdagxsu3matdag(res_fieldv, w_fieldv, x_fieldv, i);
                fsu3matxsu3mat_retrace(res_soa, temp_fieldv, res_fieldv, i);
            }
        }
    }
    prof_end(&comp_SoA);
    comp_SoA.count *= reps;

    #pragma omp target update from(res_soa->base[0 : res_soa->volume])


    // AoSoA
    su3_mat_field *u_fieldva;
    su3_mat_field *v_fieldva;
    su3_mat_field *w_fieldva;
    su3_mat_field *x_fieldva;
    floatv *res_aosoa;

    su3_mat_field *temp_fieldva;
    su3_mat_field *res_fieldva;

    u_fieldva = malloc(n_blocks * sizeof(su3_mat_field));
    v_fieldva = malloc(n_blocks * sizeof(su3_mat_field));
    w_fieldva = malloc(n_blocks * sizeof(su3_mat_field));
    x_fieldva = malloc(n_blocks * sizeof(su3_mat_field));
    res_aosoa = malloc(n_blocks * sizeof(floatv));
    temp_fieldva = malloc(sizeof(su3_mat_field));
    res_fieldva = malloc(sizeof(su3_mat_field));

    su3_mat_field_init(temp_fieldva, CACHELINE);
    su3_mat_field_init(res_fieldva, CACHELINE);
    for (size_t i = 0; i < n_blocks; i++)
    {
        su3_mat_field_init(&u_fieldva[i], CACHELINE);
        su3_mat_field_init(&v_fieldva[i], CACHELINE);
        su3_mat_field_init(&w_fieldva[i], CACHELINE);
        su3_mat_field_init(&x_fieldva[i], CACHELINE);
        floatv_init(&res_aosoa[i], CACHELINE);
    }

    prof_begin(&init_AoSoA);
    #pragma omp parallel for
    for (size_t i = 0; i < n_blocks; i++)
    {
        unit_su3mat_field(&u_fieldva[i]);
        unit_su3mat_field(&v_fieldva[i]);
        unit_su3mat_field(&w_fieldva[i]);
        unit_su3mat_field(&x_fieldva[i]);
    }
    prof_end(&init_AoSoA);

    enter_su3_mat_field(temp_fieldva);
    enter_su3_mat_field(res_fieldva);
    enter_su3_mat_field_array(u_fieldva, n_blocks);
    enter_su3_mat_field_array(v_fieldva, n_blocks);
    enter_su3_mat_field_array(w_fieldva, n_blocks);
    enter_su3_mat_field_array(x_fieldva, n_blocks);
    enter_float_field_array(res_aosoa, n_blocks);
    
    prof_begin(&comp_AoSoA);
    #pragma omp target teams firstprivate(temp_fieldva, res_fieldva) num_teams(n_blocks)
    {
        for (int r = 0; r < reps; r++)
        {
            #pragma omp distribute parallel for collapse(2)
            for (size_t b = 0; b < n_blocks; b++)
            {
                // if (r==0 & b==0) is_gpu();
                for (size_t i=0; i<CACHELINE; i++)
                {
                    fsu3matxsu3mat(temp_fieldva, &u_fieldva[b], &v_fieldva[b], i);
                    fsu3matdagxsu3matdag(res_fieldva, &w_fieldva[b], &x_fieldva[b], i);
                    fsu3matxsu3mat_retrace(&res_aosoa[b], temp_fieldva, res_fieldva, i);
                }
            }
        }
    }
    prof_end(&comp_AoSoA);
    comp_AoSoA.count *= reps;
    update_host_float_field_array(res_aosoa, n_blocks);

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
    printf("res_aos[%i] = %f \n", idx, res_aos[idx]);
    printf("res_soa[%i] = %f \n", idx, res_soa->base[idx]);
    printf("res_aosoa[%i] = %f \n", idx, res_aosoa[idx_a].base[idx_b]);
}
