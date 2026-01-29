#include <stdio.h>
#include <stdlib.h>
#include "su3.h"
#include "global.h"
#include "ufields.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char *argv[])
{
    // read idx from command line
    int idx = 0;
    if (argc > 1)
    {
        idx = atoi(argv[1]);
    }

    printf("Testing ufields structures\n");
    printf("Volume: %d\n", VOLUME);

    su3_mat u_field[VOLUME];
    su3_mat v_field[VOLUME];
    su3_mat w_field[VOLUME];

    for (size_t i = 0; i < VOLUME; i++)
    {
        random_su3mat(&u_field[i]);
        random_su3mat(&v_field[i]);
    }
    printf("u[%i]->c22 = (%f, %f)\n", idx, u_field[idx].c22.re, u_field[idx].c22.im);
    printf("u[%i]->c31 = (%f, %f)\n", idx, u_field[idx].c31.re, u_field[idx].c31.im);

#pragma omp target teams map(to : v_field[0 : VOLUME], u_field[0 : VOLUME]) \
    map(from : w_field[0 : VOLUME])
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
                printf("device u[%i]->c32 = (%f, %f)\n", idx, u_field[idx].c11.re, u_field[idx].c11.im);
                printf("device u[%i]->c33 = (%f, %f)\n", idx, u_field[idx].c12.re, u_field[idx].c12.im);
            }
        }

#pragma omp distribute parallel for
        for (int i = 0; i < VOLUME; i++)
        {
            su3matxsu3mat(&w_field[i], &u_field[i], &v_field[i]);
        }
        if (!omp_is_initial_device() && tid == 0)
        {
            printf("device w[%i]->c11 = (%f, %f)\n", idx, w_field[idx].c11.re, w_field[idx].c11.im);
            printf("device w[%i]->c12 = (%f, %f)\n", idx, w_field[idx].c12.re, w_field[idx].c12.im);
        }
    }
    printf("w[%i]->c11 = (%f, %f)\n", idx, w_field[idx].c11.re, w_field[idx].c11.im);
    printf("w[%i]->c12 = (%f, %f)\n", idx, w_field[idx].c12.re, w_field[idx].c12.im);
    printf("w[%i]->c13 = (%f, %f)\n", idx, w_field[idx].c13.re, w_field[idx].c13.im);

    // take trace of w_field[idx]
    complex tr = su3_trace(&w_field[idx]);
    printf("Trace of w[%i]: (%f, %f)\n", idx, tr.re, tr.im);
    return 0;
}
