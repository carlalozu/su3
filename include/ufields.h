/*******************************************************************************
*
* File uflds.h
*
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef UFLDS_H
#define UFLDS_H


#include "su3.h"
#include "global.h"

int alloc_su3_cdble(su3_cdble **su3);
void unit_su3_cdble(su3_cdble *su3, int value);


int alloc_ufield(su3_cdble **field, int size);

# endif // UFLDS_H
