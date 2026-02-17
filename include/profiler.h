#pragma once
#include <stdio.h>
#include <stdint.h>

#ifdef _OPENMP
  #include <omp.h>
  static inline double prof_now(void) {
    return omp_get_wtime();
  }
#else
  #include <time.h>
  static inline double prof_now(void) {
    return (double)clock() / CLOCKS_PER_SEC;
  }
#endif

typedef struct {
  const char *name;
  double total;
  int64_t count;
  double t0;
  int threads;
} prof_section;

static inline void prof_begin(prof_section *s) { s->t0 = prof_now(); }
static inline void prof_end(prof_section *s) {
  double t1 = prof_now();
  s->total += (t1 - s->t0);
  s->count += 1;
}

static inline void prof_report(const prof_section *s) {
  double avg = (s->count > 0) ? (s->total / (double)s->count) : 0.0;
  printf("%-24s total=%0.6f s | avg=%0.6f s | n=%lld | vol=%i | cache=%i | threads=%i \n",
         s->name, s->total, avg, (long long)s->count, VOLUME, CACHELINE, s->threads);
}
