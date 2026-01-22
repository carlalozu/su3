#include <stdio.h>
#include <stdlib.h>
#include "su3v.h"
#include "global.h"
#include "ufields.h"
#include <time.h>

int main(int argc, char *argv[])
{
    double start_time, end_time;
    double init_AoS_time = 0.0;
    double compute_AoS_time = 0.0;
    double init_SoA_time = 0.0;
    double compute_SoA_time = 0.0;

    // read idx from command line
    int idx = 0;
    if (argc > 1)
    {
        idx = atoi(argv[1]);
    }

    printf("Timing SoA vs AoS structures\n");
    printf("Volume: %d\n", VOLUME);

    // AoS
    su3_mat u_field[VOLUME];
    su3_mat v_field[VOLUME];
    su3_mat w_field[VOLUME];

    // SoA
    su3_mat_field u_fieldv;
    su3_mat_field v_fieldv;
    su3_mat_field w_fieldv;

    int reps = 20;

    for (int r = 0; r < reps; r++)
    {
        // initialize AoS fields
        start_time = (double)clock() / CLOCKS_PER_SEC;
        for (size_t i = 0; i < VOLUME; i++)
        {
            random_su3mat(&u_field[i]);
            random_su3mat(&v_field[i]);
            random_su3mat(&w_field[i]);
        }
        end_time = (double)clock() / CLOCKS_PER_SEC;
        init_AoS_time += end_time - start_time;

        // u*v*w AoS
        start_time = (double)clock() / CLOCKS_PER_SEC;
        usu3matxusu3mat(u_field, u_field, v_field, VOLUME);
        usu3matxusu3mat(u_field, u_field, w_field, VOLUME);
        end_time = (double)clock() / CLOCKS_PER_SEC;
        compute_AoS_time += end_time - start_time;

        // initialize SoA fields
        start_time = (double)clock() / CLOCKS_PER_SEC;
        random_su3mat_field(&u_fieldv);
        random_su3mat_field(&v_fieldv);
        random_su3mat_field(&w_fieldv);
        end_time = (double)clock() / CLOCKS_PER_SEC;
        init_SoA_time += end_time - start_time;

        // u*v*w SoA
        start_time = (double)clock() / CLOCKS_PER_SEC;
        fsu3matxsu3mat(&u_fieldv, &u_fieldv, &v_fieldv, VOLUME);
        fsu3matxsu3mat(&u_fieldv, &u_fieldv, &w_fieldv, VOLUME);
        end_time = (double)clock() / CLOCKS_PER_SEC;
        compute_SoA_time += end_time - start_time;
    }
    printf("AoS initialization time: %f seconds\n", init_AoS_time/(double)reps);
    printf("AoS time for u*v*w: %f seconds\n", compute_AoS_time/(double)reps);
    printf("SoA initialization time: %f seconds\n", init_SoA_time/(double)reps);
    printf("SoA time for u*v*w: %f seconds\n", compute_SoA_time/(double)reps);
    return 0;
}
