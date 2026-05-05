/*
 * mixed_precision.c — Y[i] += (double)X[i]  C-O3-march=native baseline.
 *
 * Models mixed f32→f64 accumulation pattern from cpu_dsl candidate 07.
 *
 * Compile: gcc -O3 -march=native -mavx512f -ffast-math -o mixed_precision mixed_precision.c
 *
 * Usage: ./mixed_precision [N]   (default N = 1048576 = 1M)
 * Output (last line): JSON {"kernel":"mixed_precision","N":<N>,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "_bench_assume.h"
#define DEFAULT_N (1 << 20)
#define WARMUP    100
#define TIMED     1000

static void __attribute__((noinline))
mixed_kernel(const float *X, double *Y, int N) {
    BENCH_ASSUME_N(N);
    for (int i = 0; i < N; i++)
        Y[i] += (double)X[i];
}

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    int N = DEFAULT_N;
    if (argc > 1) N = atoi(argv[1]);

    float *X = (float *)malloc(N * sizeof(float));
    double *Y = (double *)malloc(N * sizeof(double));
    if (!X || !Y) { fprintf(stderr, "OOM\n"); return 1; }

    for (int i = 0; i < N; i++) {
        X[i] = (float)(i % 1000) * 0.001f;
        Y[i] = 0.0;
    }

    for (int w = 0; w < WARMUP; w++)
        mixed_kernel(X, Y, N);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++)
        mixed_kernel(X, Y, N);
    long long t_total = clock_ns() - t0;

    double ms_per_call = (double)t_total / TIMED / 1e6;

    printf("{\"kernel\":\"mixed_precision\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, ms_per_call);

    free(X); free(Y);
    return 0;
}
