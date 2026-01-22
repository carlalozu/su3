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

    v_field.c1im[idx] = (idx + 1) * 1.0;
    printf("v_field[%i]->c1im[%i] = %f\n", idx, idx, v_field.c1im[idx]);

    su3_mat_field m_field;

    return 0;
}
