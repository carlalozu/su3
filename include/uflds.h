
/*******************************************************************************
*
* File uflds.h
*
* Copyright (C) 2011, 2012, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef UFLDS_H
#define UFLDS_H

#ifndef SU3_H
#include "su3.h"
#endif

#include "profiler.h"

/* PLAQ_SUM_C */
extern double plaq_sum_dble(int icom);

/* UFLDS_C */
extern su3_dble *udfld(void);
extern void random_ud(void);
extern prof_section compute;

#endif
