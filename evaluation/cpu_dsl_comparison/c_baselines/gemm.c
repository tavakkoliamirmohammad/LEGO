/*
 * gemm.c — C[i,j] += A[i,k]*B[k,j]  C-O3-march=native baseline.
 *
 * Compile: gcc -O3 -march=native -mavx512f -ffast-math -o gemm gemm.c
 *
 * Usage: ./gemm [N]        (default N = 64 matching cpu_dsl candidate)
 * Output (last line): JSON {"kernel":"gemm","N":<N>,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_N 64
#define WARMUP    100
#define TIMED     1000

static void __attribute__((noinline))
gemm_kernel(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float aik = A[i * N + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += aik * B[k * N + j];
        }
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
    float *B = (float *)malloc(NN * sizeof(float));
    float *C = (float *)malloc(NN * sizeof(float));
    if (!A || !B || !C) { fprintf(stderr, "OOM\n"); return 1; }

    for (int i = 0; i < NN; i++) {
        A[i] = (float)(i % 100) * 0.01f;
        B[i] = (float)(i % 97) * 0.01f;
        C[i] = 0.0f;
    }

    for (int w = 0; w < WARMUP; w++)
        gemm_kernel(A, B, C, N);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++)
        gemm_kernel(A, B, C, N);
    long long t_total = clock_ns() - t0;

    double ms_per_call = (double)t_total / TIMED / 1e6;

    printf("{\"kernel\":\"gemm\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, ms_per_call);

    free(A); free(B); free(C);
    return 0;
}
