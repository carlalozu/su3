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
#include <stdio.h>
#include <stdlib.h>

#ifndef GLOBAL_H
#define GLOBAL_H

#define L0 4
#define L1 4
#define L2 4
#define L3 2

#define L0_TRD 3
#define L1_TRD 3
#define L2_TRD 3
#define L3_TRD 3

#define VOLUME (L0*L1*L2*L3)
#define VOLUME_TRD (L0_TRD*L1_TRD*L2_TRD*L3_TRD)
#define ALIGN 9
#define CACHELINE 8

// #if defined(__NVPTX__) || defined(__AMDGCN__)
    /* DEVICE VERSION: No stderr or exit() allowed */
    #define ERROR(msg) do { \
        printf("GPU ERROR: " msg "\n"); \
    } while(0)
// #else
//     /* HOST VERSION: Standard behavior */
//     #define ERROR(msg, ...) do { \
//         fprintf(stderr, "HOST ERROR [%s:%d]: " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
//         exit(EXIT_FAILURE); \
//     } while(0)
// #endif

#endif
