#include <stdio.h>
#include <stdlib.h>
#include "su3v.h"
#include "global.h"
#include "ufields.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char *argv[])
{
    int idx = 0;
    if (argc > 1)
    {
        idx = atoi(argv[1]);
    }

    // NOTE: Careful with VOLUME size
    printf("Testing ufields structures\n");
    printf("Volume: %d\n", VOLUME);

    su3_vec_field *v_field = malloc(sizeof(su3_vec_field));
    su3_mat_field *m_field = malloc(sizeof(su3_mat_field));
    su3_mat_field *u_field = malloc(sizeof(su3_mat_field));
    su3_mat_field *t_field = malloc(sizeof(su3_mat_field));
    su3_vec_field *res_field = malloc(sizeof(su3_vec_field));

    random_su3vec_field(v_field);
    random_su3mat_field(m_field);
    random_su3mat_field(u_field);

    printf("Initial values on host: \n");
    printf("v_field->c1re[%i] = %f\n", idx, v_field->c1re[idx]);
    printf("v_field->c2im[%i] = %f\n", idx, v_field->c2im[idx]);
    printf("m_field->c2.c3re[%i] = %f\n", idx, m_field->c23re[idx]);
    printf("m_field->c3.c1im[%i] = %f\n", idx, m_field->c31im[idx]);

#pragma omp target teams map(to : v_field[0 : 1], m_field[0 : 1])
    {
        int tid = omp_get_team_num();
        if (omp_is_initial_device())
        {
            printf("Running on host\n");
        }
        else
        {
            int nteams = omp_get_num_teams();
            int nthreads = omp_get_num_threads();
            if (tid == 0)
            {
                printf("Running on device with %d teams in total and %d threads in each team\n", nteams, nthreads);
                printf("v_field->c1re[%i] = %f\n", idx, v_field->c1re[idx]);
                printf("v_field->c2im[%i] = %f\n", idx, v_field->c2im[idx]);
                printf("m_field->c2.c3re[%i] = %f\n", idx, m_field->c23re[idx]);
                printf("m_field->c3.c1im[%i] = %f\n", idx, m_field->c31im[idx]);
            }
        }
    }

    printf("Initial value of res_field: \n");
    printf("res_field->c1re[%i] = %f\n", idx, res_field->c1re[idx]);
    printf("res_field->c2im[%i] = %f\n", idx, res_field->c2im[idx]);

// matrix-vector field multiplication
#pragma omp target teams distribute parallel for map(to : v_field[0 : 1], m_field[0 : 1]) map(from : res_field[0 : 1])
    for (size_t i = 0; i < VOLUME; i++)
    {
        if (omp_is_initial_device())
        {
            printf("Running on host\n");
        }
        fsu3matxsu3vec(res_field, m_field, v_field, i);
    }

    printf("Result on host: \n");
    printf("res_field->c1re[%i] = %f\n", idx, res_field->c1re[idx]);
    printf("res_field->c2im[%i] = %f\n", idx, res_field->c2im[idx]);

    printf("Initial value of t_field: \n");
    printf("t_field->c1re[%i] = %f\n", idx, t_field->c11re[idx]);
    printf("t_field->c3im[%i] = %f\n", idx, t_field->c21im[idx]);

// matrix-matrix field multiplication
#pragma omp target teams distribute parallel for map(to : u_field[0 : 1], m_field[0 : 1]) map(from : t_field[0 : 1])
    for (size_t i = 0; i < VOLUME; i++)
    {
        if (omp_is_initial_device())
        {
            printf("Running on host\n");
        }
        fsu3matxsu3mat(t_field, u_field, m_field, i);
    }
    printf("Result of t_field: \n");
    printf("t_field->c1re[%i] = %f\n", idx, t_field->c11re[idx]);
    printf("t_field->c3im[%i] = %f\n", idx, t_field->c21im[idx]);

    return 0;
}
