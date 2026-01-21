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

int alloc_ufield(su3_cdble **field, size_t size);
void usu3xusu3(su3_cdble *res, su3_cdble *u_field, su3_cdble *v_field, size_t size);
void usu3xusu3vec(su3_vector_cdble *res_field, su3_cdble *u_field, su3_vector_cdble *v_field, size_t size);


# endif // UFLDS_H
