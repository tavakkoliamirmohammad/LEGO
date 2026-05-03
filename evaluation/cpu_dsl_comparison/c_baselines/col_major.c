/*
 * col_major.c — column-major strided access C[i,j] = A[j,i]  baseline.
 *
 * Models strided gather pattern from cpu_dsl candidate 04.
 * N=64 matching the cpu_dsl candidate.
 *
 * Compile: gcc -O3 -march=native -mavx512f -ffast-math -o col_major col_major.c
 *
 * Usage: ./col_major [N]   (default N = 64)
 * Output (last line): JSON {"kernel":"col_major","N":<N>,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_N 64
#define WARMUP    100
#define TIMED     1000

static void col_major_kernel(const float *A, float *C, int N) {
    /* Transpose: C[i*N+j] = A[j*N+i]  — strided access on A */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = A[j * N + i];
}

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    int N = DEFAULT_N;
    if (argc > 1) N = atoi(argv[1]);
    int NN = N * N;

    float *A = (float *)malloc(NN * sizeof(float));
    float *C = (float *)malloc(NN * sizeof(float));
    if (!A || !C) { fprintf(stderr, "OOM\n"); return 1; }

    for (int i = 0; i < NN; i++) { A[i] = (float)i * 0.001f; C[i] = 0.0f; }

    for (int w = 0; w < WARMUP; w++)
        col_major_kernel(A, C, N);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++)
        col_major_kernel(A, C, N);
    long long t_total = clock_ns() - t0;

    double ms_per_call = (double)t_total / TIMED / 1e6;

    printf("{\"kernel\":\"col_major\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, ms_per_call);

    free(A); free(C);
    return 0;
}
