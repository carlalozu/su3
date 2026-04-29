
#ifndef SU3V_OPENMP_H
#define SU3V_OPENMP_H

#include <stdlib.h>
#include <string.h>
#include "su3.h"
#include "su3v.h"


#pragma omp declare target
void complex_field_map_pointers(complexv *v);
void su3_vec_field_map_pointers(su3_vec_field *v);
void su3_mat_field_map_pointers(su3_mat_field *m);
#pragma omp end declare target

void enter_complex_field(complexv* c_field);
void enter_double_field(doublev* d_field);
void enter_su3_vec_field(su3_vec_field* v_field);
void enter_su3_mat_field(su3_mat_field* m_field);

void enter_complex_field_array(complexv* c_field, int n_blocks) ;
void enter_double_field_array(doublev* d_field, int n_blocks) ;
void enter_su3_vec_field_array(su3_vec_field* v_field, int n_blocks) ;
void enter_su3_mat_field_array(su3_mat_field* m_field, int n_blocks) ;

void update_host_complex_field_array(complexv* c_field, int n_blocks);
void update_host_double_field_array(doublev* d_field, int n_blocks);
void update_host_su3_vec_field_array(su3_vec_field* v_field, int n_blocks);
void update_host_su3_mat_field_array(su3_mat_field* m_field, int n_blocks);

#endif // SU3V_OPENMP_H
