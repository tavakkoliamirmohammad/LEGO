/*
 * brick_within_cell.c — B[i] = A[i] * 2.0f + 1.0f  C-O3-march=native baseline.
 *
 * Models the "brick within cell" access pattern (fully within a single brick,
 * i.e., unit-stride). This is the BrickLib within-cell kernel equivalent.
 *
 * Compile: gcc -O3 -march=native -mavx512f -ffast-math -o brick_within_cell brick_within_cell.c
 *
 * Usage: ./brick_within_cell [N]   (default N = 1048576 = 1M)
 * Output (last line): JSON {"kernel":"brick_within_cell","N":<N>,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "_bench_assume.h"
#define DEFAULT_N (1 << 20)
#define WARMUP    100
#define TIMED     1000

static void __attribute__((noinline))
brick_kernel(const float *A, float *B, int N) {
    BENCH_ASSUME_N(N);
    for (int i = 0; i < N; i++)
        B[i] = A[i] * 2.0f + 1.0f;
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
        brick_kernel(A, B, N);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++)
        brick_kernel(A, B, N);
    long long t_total = clock_ns() - t0;

    double ms_per_call = (double)t_total / TIMED / 1e6;

    printf("{\"kernel\":\"brick_within_cell\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, ms_per_call);

    free(A); free(B);
    return 0;
}
