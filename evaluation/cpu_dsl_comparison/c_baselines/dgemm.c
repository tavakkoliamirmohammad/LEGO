/*
 * dgemm.c — C[i,j] += A[i,k]*B[k,j]  double-precision C-O3 baseline.
 *
 * Used for candidate 42_dgemm_reg_L1_L2_tile (f64 DGEMM).
 *
 * Compile:
 *   gcc -O3 -o dgemm_O3 dgemm.c
 *   gcc -O3 -march=native -mavx512f -ffast-math -o dgemm_agg dgemm.c
 * Usage: ./dgemm_O3 [N]       (default N = 256)
 * Output: {"kernel":"dgemm","N":<N>,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_N 256
#define WARMUP    20
#define TIMED     100

static void __attribute__((noinline))
dgemm_kernel(const double * __restrict__ A, const double * __restrict__ B,
             double * __restrict__ C, int N) {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            double aik = A[i * N + k];
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

    double *A = (double *)malloc(NN * sizeof(double));
    double *B = (double *)malloc(NN * sizeof(double));
    double *C = (double *)malloc(NN * sizeof(double));
    if (!A || !B || !C) { fprintf(stderr, "OOM\n"); return 1; }

    for (int i = 0; i < NN; i++) {
        A[i] = (double)(i % 100) * 0.01;
        B[i] = (double)(i % 97) * 0.01;
        C[i] = 0.0;
    }

    for (int w = 0; w < WARMUP; w++)
        dgemm_kernel(A, B, C, N);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++)
        dgemm_kernel(A, B, C, N);
    long long t_total = clock_ns() - t0;

    double ms_per_call = (double)t_total / TIMED / 1e6;

    printf("{\"kernel\":\"dgemm\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, ms_per_call);

    free(A); free(B); free(C);
    return 0;
}
