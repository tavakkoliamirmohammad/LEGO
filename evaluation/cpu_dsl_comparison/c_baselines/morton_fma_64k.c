/*
 * morton_fma_64k.c — Z-Morton gather FMA: C[i] += A[morton(i)] * B[i] for N=64K.
 *
 * Used as the reference for candidates 09_gemm_zmorton, 10_lu_zmorton, 11_chol_zmorton,
 * 36_gemm_nonpow2_morton.
 *
 * This represents the access pattern these candidates are actually testing:
 * a gather-indexed FMA loop where the gather index is a z-morton interleave.
 *
 * Compile:
 *   gcc -O3 -march=native -mavx512f -ffast-math -o morton_fma_64k morton_fma_64k.c
 * Output: {"kernel":"morton_fma_64k","N":65536,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_N (1 << 16)
#define WARMUP    50
#define TIMED     300

static void __attribute__((noinline))
morton_fma(const float * __restrict__ A, const float * __restrict__ B,
           float * __restrict__ C, int N) {
    for (int i = 0; i < N; i++) {
        int ti = i & 0x5555;
        int tj = (i >> 1) & 0x5555;
        int morton = (ti | (tj << 1)) & (N - 1);
        C[i] = C[i] + A[morton] * B[i];
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

    float *A = (float *)malloc(N * sizeof(float));
    float *B = (float *)malloc(N * sizeof(float));
    float *C = (float *)malloc(N * sizeof(float));
    if (!A || !B || !C) { fprintf(stderr, "OOM\n"); return 1; }

    for (int i = 0; i < N; i++) { A[i] = i * 0.001f; B[i] = 1.0f; C[i] = 0.0f; }

    for (int w = 0; w < WARMUP; w++) morton_fma(A, B, C, N);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++) morton_fma(A, B, C, N);
    double ms = (double)(clock_ns() - t0) / TIMED / 1e6;

    printf("{\"kernel\":\"morton_fma_64k\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n", N, ms);
    free(A); free(B); free(C);
    return 0;
}
