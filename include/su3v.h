
#ifndef SU3V_H
#define SU3V_H

#include <stdlib.h>
#include <string.h>
#include "global.h"

/* SU3 vector
* Each component is a pointer to an array of doubles
* Real and imaginary parts are stored separately
*/
typedef struct
{
    double c1re[VOLUME], c1im[VOLUME];
    double c2re[VOLUME], c2im[VOLUME];
    double c3re[VOLUME], c3im[VOLUME];
} su3_vec_field;

/* SU3 matrix 
* Each element is a su3_vec_vdble
* Represents a column of the SU3 matrix
*/
typedef struct
{
    su3_vec_field c1, c2, c3;

} su3_mat_field;


#endif // SU3V_H
