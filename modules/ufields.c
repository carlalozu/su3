/*******************************************************************************
 *
 * File uflds.c
 *
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *******************************************************************************/

#ifndef UFLDS_C
#define UFLDS_C

#include "su3v.h"
#include "global.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * SU(3) matrix u times SU(3) vector SoA
 *
 * r.c1=(u*s).c1
 * r.c2=(u*s).c2
 * r.c3=(u*s).c3
 */

/* SU(3) matrix-matrix field multiplication SoA
 *
 * res_field = m_field * v_field
 * Where each field is a su3_mat_field of given size
 *
 * Bandwidth
 * from u = 18 load streams, 9 complex numbers
 * from v = 18 load streams
 * from res = 18 write streams
 * total = 54 doubles = 54 x 8 bytes = 432 bytes
 *
 * FLOPS
 * per matrix element = 11 operations, 6 muls, 5 adds
 * par complex matrix element = 22 operations
 * total = 9 * 22 = 198 FLOPS
 *
 * Intensity
 * 198 / 432 ≈ 0.46 flops/byte
 * = Memory bound
 */


#endif // UFLDS_C
