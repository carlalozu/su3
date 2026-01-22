
#ifndef SU3V_C
#define SU3V_C

#include <stdlib.h>
#include <string.h>
#include "su3v.h"
#include "global.h"


void random_su3_field_vdble(su3_vec_field *su3vf)
{
    double *d = (double *)su3vf;
    for (int i = 0; i < 6*VOLUME; i++)
        d[i] = (double)rand() / RAND_MAX;
}

void random_su3_matrix_field(su3_mat_field *su3mf)
{
    random_su3_field_vdble(&su3mf->c1);
    random_su3_field_vdble(&su3mf->c2);
    random_su3_field_vdble(&su3mf->c3);
}

#endif // SU3V_C