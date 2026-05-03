/*
 * vecadd.c — C[i] = A[i] + B[i]  C-O3-march=native baseline.
 *
 * Compile: gcc -O3 -march=native -mavx512f -ffast-math -o vecadd vecadd.c
 *
 * Usage: ./vecadd [N]        (default N = 1048576 = 1M)
 * Output (last line): JSON {"kernel":"vecadd","N":<N>,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_N (1 << 20)
#define WARMUP    100
#define TIMED     1000

static void __attribute__((noinline))
vecadd_kernel(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
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
    float *C = (float *)malloc(N * sizeof(float));
    if (!A || !B || !C) { fprintf(stderr, "OOM\n"); return 1; }

    for (int i = 0; i < N; i++) {
        A[i] = (float)(i % 1000) * 0.001f;
        B[i] = (float)(i % 997) * 0.001f;
        C[i] = 0.0f;
    }

    for (int w = 0; w < WARMUP; w++)
        vecadd_kernel(A, B, C, N);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++)
        vecadd_kernel(A, B, C, N);
    long long t_total = clock_ns() - t0;

    double ms_per_call = (double)t_total / TIMED / 1e6;

    printf("{\"kernel\":\"vecadd\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, ms_per_call);

    free(A); free(B); free(C);
    return 0;
}
