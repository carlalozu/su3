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
    size_t total_size = 2*c_field->volume;
    double* base_ptr = c_field->base;
    #pragma omp target enter data map(to : c_field[0], base_ptr[0:total_size])

    #pragma omp target
    complex_field_map_pointers(c_field);

}

void enter_double_field(doublev* d_field){
    size_t total_size = d_field->volume;
    double* base_ptr = d_field->base;
    #pragma omp target enter data map(to : d_field[0], base_ptr[0:total_size])
}

void enter_su3_vec_field(su3_vec_field* v_field){
    size_t total_size = 6*v_field->volume;
    double* base_ptr = v_field->base;
    #pragma omp target enter data map(to : v_field[0], base_ptr[0:total_size])

    #pragma omp target
    su3_vec_field_map_pointers(v_field);

}

void enter_su3_mat_field(su3_mat_field* m_field){

    size_t total_size = 6 * m_field->c1.volume;

    double* base_ptr_c1 = m_field->c1.base;
    double* base_ptr_c2 = m_field->c2.base;
    double* base_ptr_c3 = m_field->c3.base;

    #pragma omp target enter data map(to: m_field[0]) \
    map(to: base_ptr_c1[0 : total_size]) \
    map(to: base_ptr_c2[0 : total_size]) \
    map(to: base_ptr_c3[0 : total_size])

    #pragma omp target
    su3_mat_field_map_pointers(m_field);
}

void enter_complex_field_array(complexv* c_field, int n_blocks) {
    #pragma omp target enter data map(to: c_field[0:n_blocks])

    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = 2 * c_field[i].volume;
        double* base_ptr = c_field[i].base;
        #pragma omp target enter data map(to: base_ptr[0:total_size])

        #pragma omp target teams distribute parallel for
        for (int j = 0; j < 1; j++) { 
            // fake loop to trigger update done by each thread
            complex_field_map_pointers(&c_field[i]);
        }
    }
}

void update_host_complex_field_array(complexv* c_field, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = 2 * c_field[i].volume;
        double* base_ptr = c_field[i].base;
        // pull data from device to host for each block's data buffer
        #pragma omp target update from(base_ptr[0:total_size])
    }
}

void enter_double_field_array(doublev* d_field, int n_blocks) {
    #pragma omp target enter data map(to: d_field[0:n_blocks])

    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = d_field[i].volume;
        double* base_ptr = d_field[i].base;
        #pragma omp target enter data map(to: base_ptr[0:total_size])
    }
}


void update_host_double_field_array(doublev* d_field, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = d_field[i].volume;
        double* base_ptr = d_field[i].base;
        // pull data from device to host for each block's data buffer
        #pragma omp target update from(base_ptr[0:total_size])
    }
}

void enter_su3_vec_field_array(su3_vec_field* v_field, int n_blocks) {
    // 1. Map the array of structs itself
    #pragma omp target enter data map(to: v_field[0:n_blocks])

    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = 6 * v_field[i].volume;
        double* base_ptr = v_field[i].base;
        // 2. Map the data buffer for each struct
        // This also handles the "attachment" so v_field[i].base on the 
        // device points to the device-allocated buffer.
        #pragma omp target enter data map(to: base_ptr[0:total_size])

        // 3. Fix the internal pointers (c1re, c1im, etc.) on the device
        #pragma omp target teams distribute parallel for
        for (int j = 0; j < 1; j++) { 
            // fake loop to trigger update done by each thread
            su3_vec_field_map_pointers(&v_field[i]);
        }
    }
}

void update_host_su3_vec_field_array(su3_vec_field* v_field, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = 6 * v_field[i].volume;
        double* base_ptr = v_field[i].base;
        // pull data from device to host for each block's data buffer
        #pragma omp target update from(base_ptr[0:total_size])
    }
}

void enter_su3_mat_field_array(su3_mat_field* m_field, int n_blocks) {
    // map the array of structs itself
    #pragma omp target enter data map(to: m_field[0:n_blocks])

    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = 6 * m_field[i].c1.volume;

        double* base_ptr_c1 = m_field[i].c1.base;
        double* base_ptr_c2 = m_field[i].c2.base;
        double* base_ptr_c3 = m_field[i].c3.base;

        // map the data buffer for each struct
        #pragma omp target enter data map(to: base_ptr_c1[0:total_size])
        #pragma omp target enter data map(to: base_ptr_c2[0:total_size])
        #pragma omp target enter data map(to: base_ptr_c3[0:total_size])

        // fix the internal pointers on the device
        #pragma omp target teams distribute parallel for
        for (int j = 0; j < 1; j++) {
            su3_mat_field_map_pointers(&m_field[i]);
        }
    }
}

void update_host_su3_mat_field_array(su3_mat_field* m_field, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        size_t total_size = 6 * m_field[i].c1.volume;
        
        double* base_ptr_c1 = m_field[i].c1.base;
        double* base_ptr_c2 = m_field[i].c2.base;
        double* base_ptr_c3 = m_field[i].c3.base;

        // pull data from device to host for each block's data buffer
        #pragma omp target update from(base_ptr_c1[0:total_size])
        #pragma omp target update from(base_ptr_c2[0:total_size])
        #pragma omp target update from(base_ptr_c3[0:total_size])
    }
}

#endif
