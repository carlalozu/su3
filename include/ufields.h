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
#include "su3v.h"
#include "global.h"

void usu3xusu3(su3_cdble *res, su3_cdble *u_field, su3_cdble *v_field, size_t size);
void usu3xusu3vec(su3_vector_cdble *res_field, su3_cdble *u_field, su3_vector_cdble *v_field, size_t size);

void usu3xusu3v(su3_field_dble *res, su3_field_dble *u_field, su3_field_dble *v_field, size_t size);

# endif // UFLDS_H
