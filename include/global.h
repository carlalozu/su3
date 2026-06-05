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


#define VOLUME_TRD (L0_TRD*L1_TRD*L2_TRD*L3_TRD)
#define NTHREAD (VOLUME/VOLUME_TRD)

#define N0 (NPROC0*L0)

#pragma omp declare target
extern int *ipt;
extern int (*iup)[4];
extern int (*idn)[4];
extern int (*iupT)[VOLUME];
#pragma omp end declare target

#if defined(KOKKOS_CORE_HPP)
  #define DEVICE_KEYWORD KOKKOS_INLINE_FUNCTION
  #define PRAGMA_OMP_BEGIN
  #define PRAGMA_OMP_END
#elif defined(__CUDACC__)
  #define DEVICE_KEYWORD __device__ static inline
  #define PRAGMA_OMP_BEGIN
  #define PRAGMA_OMP_END
#else
  #define DEVICE_KEYWORD static inline
  #define PRAGMA_OMP_BEGIN _Pragma("omp declare target")
  #define PRAGMA_OMP_END _Pragma("omp end declare target")
#endif

#endif
