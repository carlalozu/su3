#include <stdio.h>
#include <stdlib.h>
#include "su3v.h"
#include "global.h"
#include "ufields.h"

int main(int argc, char *argv[])
{
    // read idx from command line
    int idx = 0;
    if (argc > 1) {
        idx = atoi(argv[1]);
    }

    printf("Testing ufields structures\n");
    printf("Volume: %d\n", VOLUME);

    su3_vec_field v_field;
    su3_mat_field m_field;
    su3_mat_field u_field;
    su3_vec_field res_field;

    random_su3vec_field(&v_field, VOLUME);
    random_su3mat_field(&m_field, VOLUME);
    random_su3mat_field(&u_field, VOLUME);
    printf("v_field[%i]->c1re[%i] = %f\n", idx, idx, v_field.c1re[idx]);
    printf("v_field[%i]->c2im[%i] = %f\n", idx, idx, v_field.c2im[idx]);
    printf("m_field[%i]->c2.c3re[%i] = %f\n", idx, idx, m_field.c2.c3re[idx]);
    printf("m_field[%i]->c3.c1im[%i] = %f\n", idx, idx, m_field.c3.c1im[idx]);

    // matrix-vector field multiplication
    fsu3matxsu3vec(&res_field, &m_field, &v_field, 0, VOLUME);
    printf("res_field[%i]->c1re[%i] = %f\n", idx, idx, res_field.c1re[idx]);
    printf("res_field[%i]->c2im[%i] = %f\n", idx, idx, res_field.c2im[idx]);

    // matrix-matrix field multiplication
    fsu3matxsu3mat(&m_field, &u_field, &m_field, 0, VOLUME);
    printf("m_field[%i]->c1.c1re[%i] = %f\n", idx, idx, m_field.c1.c1re[idx]);
    printf("m_field[%i]->c3.c3im[%i] = %f\n", idx, idx, m_field.c3.c3im[idx]);
    return 0;
}
