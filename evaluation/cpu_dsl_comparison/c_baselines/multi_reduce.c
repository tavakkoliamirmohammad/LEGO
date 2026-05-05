/* multi_reduce.c — sum + max + min in a single pass over A[].
 *
 * Three iter_args; clang/gcc handle one reduction tree well but the
 * combined sum+max+min often falls back to partial-vector code on
 * Zen 4. LEGO's generalised filtered-reduce pass keeps N parallel
 * vector accumulators in registers.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include "_bench_assume.h"
#define DEFAULT_N (1 << 20)
#define WARMUP    50
#define TIMED     500

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

__attribute__((noinline))
static void kernel(const float * __restrict A,
                   float * __restrict os,
                   float * __restrict om,
                   float * __restrict on, int N) {
    BENCH_ASSUME_N(N);
    float s = 0.0f, mx = -1.0e30f, mn = 1.0e30f;
    for (int i = 0; i < N; i++) {
        float v = A[i];
        s += v;
        if (v > mx) mx = v;
        if (v < mn) mn = v;
    }
    os[0] = s; om[0] = mx; on[0] = mn;
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    float *A = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    float os[1] = {0.0f}, om[1] = {0.0f}, on[1] = {0.0f};
    if (!A) { perror("alloc"); return 1; }

    unsigned int seed = 42u;
    for (int i = 0; i < N; i++) {
        seed = seed * 1103515245u + 12345u;
        A[i] = (float)((seed >> 16) & 0xff) / 128.0f - 1.0f;
    }

    for (int w = 0; w < WARMUP; w++) kernel(A, os, om, on, N);
    long long mn = -1;
    for (int t = 0; t < 5; t++) {
        long long s = clock_ns();
        for (int it = 0; it < TIMED; it++) kernel(A, os, om, on, N);
        long long e = clock_ns();
        long long ns = (e - s) / TIMED;
        if (mn < 0 || ns < mn) mn = ns;
    }
    volatile float sk = os[0] + om[0] + on[0]; (void)sk;
    printf("{\"kernel\":\"multi_reduce\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, mn / 1.0e6);
    free(A);
    return 0;
}
