#include <stdio.h>
#include <stdlib.h>
#include "su3v.h"
#include "global.h"
#include "ufields.h"
#include <time.h>
#include "profiler.h"
#include "utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char *argv[])
{

    // read reps from command line
    int reps = 10;
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
    print_parallel_info();
    printf("Volume: %d\n", VOLUME);
    printf("Volume block: %d\n", VOLUME_TRD);

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

#pragma omp target teams map(to : v_field[0 : VOLUME], u_field[0 : VOLUME], w_field[0 : VOLUME]) \
    map(from : res_aos[0 : VOLUME])
    {
        int th_id = omp_get_team_num();
        int te_id = omp_get_thread_num();
        is_gpu();

        if (te_id == 0 && th_id == 0)
            prof_begin(&init_AoS);
        #pragma omp distribute parallel for
        for (size_t i = 0; i < VOLUME; i++)
        {
            unit_su3mat(&u_field[i]);
            unit_su3mat(&v_field[i]);
            unit_su3mat(&w_field[i]);
        }
        if (te_id == 0 && th_id == 0)
            prof_end(&init_AoS);

        for (int r = 0; r < reps; r++)
        {
            if (te_id == 0 && th_id == 0)
                prof_begin(&comp_AoS);
            #pragma omp distribute parallel for
            for (size_t i = 0; i < VOLUME; i++)
            {
                su3_mat tmp;
                su3_mat res;
                su3matxsu3mat(&tmp, &u_field[i], &v_field[i]);
                su3matxsu3mat(&res, &tmp, &w_field[i]);
                res_aos[i] = su3_trace(&res);
            }
            if (te_id == 0 && th_id == 0)
                prof_end(&comp_AoS);
        }
    }

    // SoA
    su3_mat_field *u_fieldv = malloc(sizeof(su3_mat_field));
    su3_mat_field *v_fieldv = malloc(sizeof(su3_mat_field));
    su3_mat_field *w_fieldv = malloc(sizeof(su3_mat_field));
    su3_mat_field *temp_fieldv = malloc(sizeof(su3_mat_field));
    su3_mat_field *res_fieldv = malloc(sizeof(su3_mat_field));
    complexv *res_soa = malloc(sizeof(complexv));

    prof_begin(&init_SoA);
    unit_su3mat_field(u_fieldv);
    unit_su3mat_field(v_fieldv);
    unit_su3mat_field(w_fieldv);
    prof_end(&init_SoA);

    for (int r = 0; r < reps; r++)
    {

        prof_begin(&comp_SoA);
        fsu3matxsu3mat(temp_fieldv, u_fieldv, v_fieldv, 0, VOLUME_TRD);
        fsu3matxsu3mat(res_fieldv, temp_fieldv, w_fieldv, 0, VOLUME_TRD);
        fsu3mattrace(res_soa, res_fieldv, 0, VOLUME_TRD);
        prof_end(&comp_SoA);
    }

    // AoSoA
    int n_blocks = (VOLUME + VOLUME_TRD - 1) / VOLUME_TRD;
    printf("Number of blocks: %d\n", n_blocks);
    su3_mat_field *u_fieldva;
    su3_mat_field *v_fieldva;
    su3_mat_field *w_fieldva;
    complexv *res_aosoa;

    u_fieldva = malloc(n_blocks * sizeof(su3_mat_field));
    v_fieldva = malloc(n_blocks * sizeof(su3_mat_field));
    w_fieldva = malloc(n_blocks * sizeof(su3_mat_field));
    res_aosoa = malloc(n_blocks * sizeof(complexv));

#pragma omp target teams map(to : u_fieldva[0 : n_blocks],                         \
                                 v_fieldva[0 : n_blocks], w_fieldva[0 : n_blocks]) \
    map(from : res_aosoa[0 : n_blocks])
    {
        is_gpu();
        su3_mat_field temp_fieldva;
        su3_mat_field res_fieldva;

        int th_id = omp_get_team_num();
        int te_id = omp_get_thread_num();

        if (te_id == 0 && th_id == 0)
            prof_begin(&init_AoSoA);
        #pragma omp distribute parallel for
        for (size_t i = 0; i < n_blocks; i++)
        {
            unit_su3mat_field(&u_fieldva[i]);
            unit_su3mat_field(&v_fieldva[i]);
            unit_su3mat_field(&w_fieldva[i]);
        }
        if (te_id == 0 && th_id == 0)
            prof_end(&init_AoSoA);

        for (int r = 0; r < reps; r++)
        {
            if (te_id == 0 && th_id == 0)
                prof_begin(&comp_AoSoA);
            #pragma omp distribute parallel for
            for (size_t b = 0; b < n_blocks; b++)
            {
                fsu3matxsu3mat(&temp_fieldva, &u_fieldva[b], &v_fieldva[b], 0, VOLUME_TRD);
                fsu3matxsu3mat(&res_fieldva, &temp_fieldva, &w_fieldva[b], 0, VOLUME_TRD);
                fsu3mattrace(&res_aosoa[b], &res_fieldva, 0, VOLUME_TRD);
            }
            if (te_id == 0 && th_id == 0)
                prof_end(&comp_AoSoA);
        }
    }

    // print report
    printf("\n Init \n");
    prof_report(&init_AoS);
    prof_report(&init_SoA);
    prof_report(&init_AoSoA);

    printf("\n Compute \n");
    prof_report(&comp_AoS);
    prof_report(&comp_SoA);
    prof_report(&comp_AoSoA);

    int idx_a = idx / VOLUME_TRD;
    int idx_b = idx % VOLUME_TRD;
    printf("res_aos[%i] (re[%i], im[%i]) = (%f, %f) \n", idx, idx, idx, res_aos[idx].re, res_aos[idx].im);
    printf("res_soa[%i] (re[%i], im[%i]) = (%f, %f) \n", idx, idx, idx, res_soa->re[idx], res_soa->im[idx]);
    printf("res_aosoa[%i] (re[%i], im[%i]) = (%f, %f) \n", idx, idx, idx, res_aosoa[idx_a].re[idx_b], res_aosoa[idx_a].im[idx_b]);
}
