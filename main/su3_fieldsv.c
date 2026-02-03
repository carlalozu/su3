#include <stdio.h>
#include <stdlib.h>
#include "su3v.h"
#include "global.h"
#include "ufields.h"
#include "utils.c"
#include "profiler.h"

int main(int argc, char *argv[])
{
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

    prof_section comp_SoA = {.name = "SoA compute"};

    printf("Testing ufields structures\n");
    printf("Volume: %d\n", VOLUME);

    su3_vec_field *v_field = (su3_vec_field*)malloc(sizeof(su3_vec_field));
    su3_mat_field *m_field = (su3_mat_field*)malloc(sizeof(su3_mat_field));
    su3_mat_field *u_field = (su3_mat_field*)malloc(sizeof(su3_mat_field));
    su3_vec_field *resv_field = (su3_vec_field*)malloc(sizeof(su3_vec_field));
    su3_mat_field *resm_field = (su3_mat_field*)malloc(sizeof(su3_mat_field));

    su3_vec_field_init(v_field, VOLUME);
    su3_mat_field_init(m_field, VOLUME);
    su3_mat_field_init(u_field, VOLUME);
    su3_mat_field_init(resm_field, VOLUME);
    su3_vec_field_init(resv_field, VOLUME);

    random_su3vec_field(v_field);
    random_su3mat_field(m_field);
    random_su3mat_field(u_field);
    printf("v_field[%i]->c1re[%i] = %f\n", idx, idx, v_field->c1re[idx]);
    printf("v_field[%i]->c2im[%i] = %f\n", idx, idx, v_field->c2im[idx]);
    printf("m_field[%i]->c2.c3re[%i] = %f\n", idx, idx, m_field->c2.c3re[idx]);
    printf("m_field[%i]->c3.c1im[%i] = %f\n", idx, idx, m_field->c3.c1im[idx]);

    // move data to the gpu, move struct pointer and data inside
    enter_su3_vec_field(v_field);
    enter_su3_vec_field(resv_field);    
    enter_su3_mat_field(m_field);

    #pragma omp target
    {
        // remap the pointers on the gpu
        if (!omp_is_initial_device())
        {
            su3_vec_field_map_pointers(v_field);
            su3_vec_field_map_pointers(resv_field);
            su3_mat_field_map_pointers(m_field);
        }
    }
    
    prof_begin(&comp_SoA);
    #pragma omp target teams distribute parallel for
    {
        // matrix-vector field multiplication
        for (size_t i = 0; i < VOLUME; i++)
            fsu3matxsu3vec(resv_field, m_field, v_field, i);
    }
    prof_end(&comp_SoA);

    #pragma omp target update from(resv_field->base[0 : 6*resv_field->volume])

    printf("resv_field[%i]->c1re[%i] = %f\n", idx, idx, resv_field->c1re[idx]);
    printf("resv_field[%i]->c2im[%i] = %f\n", idx, idx, resv_field->c2im[idx]);

    // matrix-matrix field multiplication
    for (size_t i = 0; i < VOLUME; i++)
        fsu3matxsu3mat(resm_field, u_field, m_field, i);
    printf("resm_field[%i]->c1.c1re[%i] = %f\n", idx, idx, resm_field->c1.c1re[idx]);
    printf("resm_field[%i]->c3.c3im[%i] = %f\n", idx, idx, resm_field->c3.c3im[idx]);

    prof_report(&comp_SoA);
    return 0;
}
