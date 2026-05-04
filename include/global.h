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

#define L0 4
#define L1 4
#define L2 4
#define L3 4

#define NPROC0 1
#define NPROC1 1
#define NPROC2 1
#define NPROC3 1

#define L0_TRD 4
#define L1_TRD 4
#define L2_TRD 4
#define L3_TRD 4


#define VOLUME (L0*L1*L2*L3)
#define ALIGN 64
#define CACHELINE 128

#ifdef _OPENMP
#include <omp.h>
#else
    #define omp_get_num_threads() 1
    #define omp_get_num_teams() 1
    #define omp_is_initial_device() 1
    #define omp_get_team_num() 0
    #define omp_get_thread_num() 0
#endif

#define VOLUME_TRD (L0_TRD*L1_TRD*L2_TRD*L3_TRD)
#define NTHREAD (VOLUME/VOLUME_TRD)

#define N0 (NPROC0*L0)

#if defined MAIN_PROGRAM

#pragma omp declare target
int *ipt=NULL;
int (*iup)[4]=NULL;
int (*idn)[4]=NULL;
int (*iupT)[VOLUME]=NULL;
#pragma omp end declare target

#else

#pragma omp declare target
extern int *ipt;
extern int (*iup)[4];
extern int (*idn)[4];
extern int (*iupT)[VOLUME];
#pragma omp end declare target

#endif

#endif
