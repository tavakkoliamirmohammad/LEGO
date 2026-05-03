/*
 * fma_64k.c — unit-stride FMA kernel for N=65536 floats.
 *
 * Used as the reference for candidates 09_gemm_zmorton, 10_lu_zmorton,
 * 11_chol_zmorton, 34_gemm_pow2_pad, 35_heat3d_pow2_pad, 36_gemm_nonpow2_morton.
 *
 * These candidates use N=64K and test either gather (Morton) or unit-stride
 * (pow2-pad) access patterns. This file provides the naive C reference for
 * the unit-stride case to complete the dual-baseline picture.
 *
 * Compile:
 *   gcc -O3 -march=native -mavx512f -ffast-math -o fma_64k fma_64k.c
 * Output: {"kernel":"fma_64k","N":65536,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_N (1 << 16)
#define WARMUP    100
#define TIMED     300

static void __attribute__((noinline))
fma_kernel(const float * __restrict__ A, const float * __restrict__ B,
           float * __restrict__ C, int N) {
    for (int i = 0; i < N; i++)
        C[i] = A[i] * B[i] + C[i];
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

    for (int w = 0; w < WARMUP; w++) fma_kernel(A, B, C, N);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++) fma_kernel(A, B, C, N);
    double ms = (double)(clock_ns() - t0) / TIMED / 1e6;

    printf("{\"kernel\":\"fma_64k\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n", N, ms);
    free(A); free(B); free(C);
    return 0;
}
