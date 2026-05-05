/*
 * stencil_3pt.c — B[i] = A[i-1] + A[i] + A[i+1]  C-O3-march=native baseline.
 *
 * Compile: gcc -O3 -march=native -mavx512f -ffast-math -o stencil_3pt stencil_3pt.c
 *
 * Usage: ./stencil_3pt [N]        (default N = 8192 matching cpu_dsl candidate)
 * Output (last line): JSON {"kernel":"stencil_3pt","N":<N>,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "_bench_assume.h"
#define DEFAULT_N 1024
#define WARMUP    100
#define TIMED     1000

static void __attribute__((noinline))
stencil_kernel(const float *A, float *B, int N) {
    BENCH_ASSUME_N(N);
    for (int i = 1; i < N - 1; i++)
        B[i] = A[i-1] + A[i] + A[i+1];
}

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    int N = DEFAULT_N;
    if (argc > 1) N = atoi(argv[1]);

    float *A = (float *)malloc(N * sizeof(float));
    float *B = (float *)malloc(N * sizeof(float));
    if (!A || !B) { fprintf(stderr, "OOM\n"); return 1; }

    for (int i = 0; i < N; i++) {
        A[i] = (float)(i % 1000) * 0.001f;
        B[i] = 0.0f;
    }

    for (int w = 0; w < WARMUP; w++)
        stencil_kernel(A, B, N);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++)
        stencil_kernel(A, B, N);
    long long t_total = clock_ns() - t0;

    double ms_per_call = (double)t_total / TIMED / 1e6;

    printf("{\"kernel\":\"stencil_3pt\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, ms_per_call);

    free(A); free(B);
    return 0;
}
