
/*******************************************************************************
*
* File lattice.h
*
* Copyright (C) 2011, 2012, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef LATTICE_H
#define LATTICE_H


/* GEOGEN_C */
#if ((defined GEOGEN_C)||(defined GEOMETRY_C))
extern void set_iupdn(void);
#endif

#ifndef UTILS_H
#include "utils.h"
#endif
#include "global.h"

/* GEOMETRY_C */
#ifdef __cplusplus
extern "C" {
#endif
extern void geometry(void);
#ifdef __cplusplus
}
#endif
#pragma omp declare target
extern int global_time(int ix);
extern int *tms;
#pragma omp end declare target

/* UIDX_C */
#pragma omp declare target
extern int offset(int ix,int mu);
extern void plaq_uidx(int mu,int nu,int ix,int *ip);
#pragma omp end declare target

#endif
