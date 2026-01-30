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

    printf("Testing ufields structures\n");
    printf("Volume: %d\n", VOLUME);
    printf("Local volume: %i\n", VOLUME_TRD);

    int n_blocks = (VOLUME + VOLUME_TRD - 1) / VOLUME_TRD;
    int n_rem = n_blocks * VOLUME_TRD - VOLUME;
    printf("Number of blocks in the array: %i\n", n_blocks);
    printf("Extra elements: %i\n ", n_rem);

    su3_vec_field *v_field;
    su3_mat_field *m_field;
    su3_mat_field *u_field;
    su3_mat_field *t_field;
    su3_vec_field *res_field;

    v_field = malloc(n_blocks * sizeof(su3_vec_field));
    m_field = malloc(n_blocks * sizeof(su3_mat_field));
    u_field = malloc(n_blocks * sizeof(su3_mat_field));
    t_field = malloc(n_blocks * sizeof(su3_mat_field));
    res_field = malloc(n_blocks * sizeof(su3_vec_field));

    for (size_t n = 0; n < n_blocks; n++)
    {
        random_su3vec_field(&v_field[n]);
        random_su3mat_field(&m_field[n]);
        random_su3mat_field(&u_field[n]);
    }

    int idxo = idx / VOLUME_TRD;
    int idxi = idx % VOLUME_TRD;
    printf("Initial values on host: \n");
    printf("v_field[%i]->c1re[%i] = %f\n", idxo, idxi, v_field[idxo].c1re[idxi]);
    printf("v_field[%i]->c2im[%i] = %f\n", idxo, idxi, v_field[idxo].c2im[idxi]);
    printf("m_field[%i]->c23re[%i] = %f\n", idxo, idxi, m_field[idxo].c23re[idxi]);
    printf("m_field[%i]->c31im[%i] = %f\n", idxo, idxi, m_field[idxo].c31im[idxi]);

#pragma omp target teams map(to : v_field[0 : n_blocks], m_field[0 : n_blocks])
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
                printf("v_field[%i]->c1re[%i] = %f\n", idxo, idxi, v_field[idxo].c1re[idxi]);
                printf("v_field[%i]->c2im[%i] = %f\n", idxo, idxi, v_field[idxo].c2im[idxi]);
                printf("m_field[%i]->c23re[%i] = %f\n", idxo, idxi, m_field[idxo].c23re[idxi]);
                printf("m_field[%i]->c31im[%i] = %f\n", idxo, idxi, m_field[idxo].c31im[idxi]);
            }
        }
    }

    printf("Initial value of res_field: \n");
    printf("res_field[%i]->c1re[%i] = %f\n", idxo, idxi, res_field[idxo].c1re[idxi]);
    printf("res_field[%i]->c2im[%i] = %f\n", idxo, idxi, res_field[idxo].c2im[idxi]);

// matrix-vector field multiplication
#pragma omp target teams distribute parallel for map(to : v_field[0 : n_blocks], m_field[0 : n_blocks]) map(from : res_field[0 : n_blocks])
    for (size_t n = 0; n < n_blocks; n++)
    {
        for (size_t i = 0; i < VOLUME_TRD; i++)
        {
            fsu3matxsu3vec(&res_field[n], &m_field[n], &v_field[n], i);
        }
    }

    printf("Result on host: \n");
    printf("res_field[%i]->c1re[%i] = %f\n", idxo, idxi, res_field[idxo].c1re[idxi]);
    printf("res_field[%i]->c2im[%i] = %f\n", idxo, idxi, res_field[idxo].c2im[idxi]);

    printf("Initial value of t_field: \n");
    printf("t_field[%i]->c23re[%i] = %f\n", idxo, idxi, t_field[idxo].c23re[idxi]);
    printf("t_field[%i]->c31im[%i] = %f\n", idxo, idxi, t_field[idxo].c31im[idxi]);

// matrix-matrix field multiplication
#pragma omp target teams distribute parallel for map(to : u_field[0 : n_blocks], m_field[0 : n_blocks]) map(from : t_field[0 : n_blocks])
    for (size_t n = 0; n < n_blocks; n++)
    {
        for (size_t i = 0; i < VOLUME_TRD; i++)
        {
            fsu3matxsu3mat(&t_field[n], &u_field[n], &m_field[n], i);
        }
    }
    printf("Result of t_field: \n");
    printf("t_field[%i]->c23re[%i] = %f\n", idxo, idxi, t_field[idxo].c23re[idxi]);
    printf("t_field[%i]->c31im[%i] = %f\n", idxo, idxi, t_field[idxo].c31im[idxi]);

    return 0;
}
