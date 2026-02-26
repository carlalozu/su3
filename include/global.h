/*******************************************************************************
*
* File global.h
*
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Global parameters and arrays
*
*******************************************************************************/

#ifndef GLOBAL_H
#define GLOBAL_H

#define L0 128
#define L1 8
#define L2 8
#define L3 8

#define VOLUME (L0*L1*L2*L3)
#define ALIGN 64
#define CACHELINE 8

#ifdef _OPENMP
#include <omp.h>
#else
    #define omp_get_num_threads() 1
    #define omp_get_num_teams() 1
    #define omp_is_initial_device() 1
    #define omp_get_team_num() 0
    #define omp_get_thread_num() 0
#endif

#endif
