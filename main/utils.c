#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include "global.h"
#include "su3v.h"

#ifdef _OPENMP
#include <omp.h>

void is_gpu()
{
    int th_id = omp_get_team_num();
    int te_id = omp_get_thread_num();
    int nteams = omp_get_num_teams();
    int nthreads = omp_get_num_threads();
    if (omp_is_initial_device())
    {
        if (te_id == 0 && th_id == 0)
        {
        printf("Running on host with %i threads\n", nthreads);
        }
    }
    else
    {
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

void enter_complex_field(complexv* c_field){
    #pragma omp target enter data \
    map(to : c_field[0], c_field->base[0:2*c_field->volume])

}

void enter_su3_vec_field(su3_vec_field* v_field){
    #pragma omp target enter data \
    map(to : v_field[0], v_field->base[0:6*v_field->volume])

}

void enter_su3_mat_field(su3_mat_field* m_field){
    #pragma omp target enter data map(to: m_field[0]) \
    map(to: m_field->c1.base[0 : 6*m_field->c1.volume]) \
    map(to: m_field->c2.base[0 : 6*m_field->c2.volume]) \
    map(to: m_field->c3.base[0 : 6*m_field->c3.volume])
}


#endif