
/*******************************************************************************
*
* File random.h
*
* Copyright (C) 2005, 2011, 2013, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef RANDOM_H
#define RANDOM_H

#if ((defined RANLUX_C)||(defined RANLXS_C)||(defined RANLXD_C)|| \
     (defined RANLUX_COMMON_C))

#include <stdint.h>


typedef struct
{
   int pr,ir;
   uint64_t (*state)[4];
} rlx_state_t;

/* RANLUX_COMMON_C */
extern void rlx_alloc_state(int n,rlx_state_t *s);
extern void rlx_free_state(int n,rlx_state_t *s);
extern void rlx_init(rlx_state_t *s,int seed,int flag);
extern void rlx_get_state(rlx_state_t *s,int *is);
extern void rlx_set_state(int *is,rlx_state_t *s);
extern void rlx_update(rlx_state_t *s);
extern void rlx_converts(rlx_state_t *s,float *rs);
extern void rlx_convertd(rlx_state_t *s,double *rd);

#endif

#ifdef __cplusplus
extern "C" {
#endif

/* GAUSS_C */
extern void gauss_dble(double *r,int n);

/* RANLXD_C */
extern void ranlxd(double *r,int n);
extern void rlxd_init(int n,int level,int seed,int seed_shift);
extern int rlxd_size(void);
extern void rlxd_get(int *state);
extern void rlxd_reset(int *state);

#ifdef __cplusplus
}
#endif

#endif
