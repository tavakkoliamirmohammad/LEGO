/*
 * self_update.c — A[i] = A[i-1] + A[i]  (loop-carried dep, not vectorizable).
 *
 * Models the self_update stencil (candidate 06). This kernel has a loop-carried
 * dependence and should NOT be vectorized by gcc either.
 *
 * Compile: gcc -O3 -march=native -o self_update self_update.c
 *
 * Usage: ./self_update [N]   (default N = 1024)
 * Output (last line): JSON {"kernel":"self_update","N":<N>,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_N 1024
#define WARMUP    100
#define TIMED     1000

static void __attribute__((noinline))
self_update_kernel(float *A, float *B, int N) {
    /* B[i] = A[i-1] + A[i] — reads A sequentially, writes B (no dep on B) */
    for (int i = 1; i < N; i++)
        B[i] = A[i-1] + A[i];
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

    for (int i = 0; i < N; i++) { A[i] = (float)i * 0.001f; B[i] = 0.0f; }

    for (int w = 0; w < WARMUP; w++)
        self_update_kernel(A, B, N);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++)
        self_update_kernel(A, B, N);
    long long t_total = clock_ns() - t0;

    double ms_per_call = (double)t_total / TIMED / 1e6;

    printf("{\"kernel\":\"self_update\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, ms_per_call);

    free(A); free(B);
    return 0;
}
