#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include "global.h"

#ifdef _OPENMP
#include <omp.h>

void is_gpu()
{
    int th_id = omp_get_team_num();
    int te_id = omp_get_thread_num();
    if (omp_is_initial_device())
    {
        printf("Running on host\n");
    }
    else
    {
        int nteams = omp_get_num_teams();
        int nthreads = omp_get_num_threads();
        if (te_id == 0 && th_id == 0)
        {
            printf("Running on device with %d teams in total and %d threads in each team\n", nteams, nthreads);
        }
    }
}

void print_parallel_info()
{
    printf("OpenMP is enabled\n");
    int n_threads = omp_get_max_threads();
    printf("Number of threads: %d\n", n_threads);
    printf("Number of size per thread: %d\n", VOLUME / n_threads);
}
#else
void is_gpu()
{
    printf("Running on host\n");
}

void print_parallel_info()
{
    printf("OpenMP is not enabled\n");
}

#endif

#endif