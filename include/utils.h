#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "global.h"
#include "su3v.h"

#ifdef _OPENMP
#include <omp.h>

static void _is_gpu()
{
    int th_id = omp_get_team_num();
    int te_id = omp_get_thread_num();
    int nteams = omp_get_num_teams();
    int nthreads = omp_get_num_threads();
    if (omp_is_initial_device())
    {
        if (te_id == 0 && th_id == 0)
        {
        printf("Running on host with %i threads\n", nthreads);
        }
    }
    else
    {
        if (te_id == 0 && th_id == 0)
        {
            printf("Running on device with %d teams in total and %d threads in each team\n", nteams, nthreads);
        }
    }
}

static void print_parallel_info()
{
    printf("OpenMP is enabled\n");
    int n_threads = omp_get_max_threads();
    printf("Number of threads: %d\n", n_threads);
    printf("Number of size per thread: %d\n", VOLUME / n_threads);
}
#else
static void _is_gpu()
{
    printf("Running on host\n");
}

static void print_parallel_info()
{
    printf("OpenMP is not enabled\n");
}

#endif


static inline void acc_qflt(double u,double *qr)
{
   double a,b,qp,up;
   double c,d;

   a=qr[0]+u;
   qp=a-u;
   up=a-qp;
   b=(qr[0]-qp)+(u-up);

   c=qr[1]+b;
   d=a+c;

   qr[0]=d;
   qr[1]=c-(d-a);
}


/* Modular arithmetic safe for negative numerators. */
static inline int safe_mod(int a, int b)
{
   return ((a % b) + b) % b;
}

/* Boundary condition type: 3 = fully periodic (no open/SF boundaries). */
static inline int bc_type(void) { return 3; }

/* Aligned memory allocation (POSIX). */
static inline void *amalloc(size_t n, int align)
{
   void *ptr = NULL;
   if (posix_memalign(&ptr, (size_t)align, n) != 0)
      ptr = NULL;
   return ptr;
}

/* Error handlers — print message and abort. */
static inline void error(int cond, int no, const char *name, const char *fmt, ...)
{
   if (cond) {
      fprintf(stderr, "Error in %s [code %d]: ", name, no);
      va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
      fprintf(stderr, "\n");
      exit(EXIT_FAILURE);
   }
}

static inline void error_root(int cond, int no, const char *name, const char *fmt, ...)
{
   if (cond) {
      fprintf(stderr, "Error in %s [code %d]: ", name, no);
      va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
      fprintf(stderr, "\n");
      exit(EXIT_FAILURE);
   }
}

/* set_bc: no-op for periodic boundary conditions. */
static inline void set_bc(void) {}


#endif
