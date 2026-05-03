/*
 * stride2_16k.c — stride-2 gather kernel: B[i] = A[i*2] * 2.0 for N=16384.
 *
 * Used as the reference for candidates:
 *   23_symm_rfp, 24_syrk_rfp, 25_nw_antidiag, 38_nussinov_nonpow2_skew
 *
 * These candidates model stride-2 access patterns (RFP format, antidiag tile,
 * non-pow2 skew). The C baseline measures what a plain scalar loop (with
 * compiler auto-vectorization) achieves on the same access pattern.
 *
 * Compile:
 *   gcc -O3 -march=native -mavx512f -ffast-math -o stride2_16k stride2_16k.c
 * Output: {"kernel":"stride2_16k","N":16384,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_N  16384
#define STRIDE     2
#define WARMUP     50
#define TIMED      300

static void __attribute__((noinline))
stride2_kernel(const float * __restrict__ A, float * __restrict__ B, int N) {
    for (int i = 0; i < N; i++)
        B[i] = A[i * STRIDE] * 2.0f;
}

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    int N = DEFAULT_N;
    if (argc > 1) N = atoi(argv[1]);

    float *A = (float *)malloc(N * STRIDE * sizeof(float));
    float *B = (float *)malloc(N * sizeof(float));
    if (!A || !B) { fprintf(stderr, "OOM\n"); return 1; }

    for (int i = 0; i < N * STRIDE; i++) A[i] = i * 0.001f;
    for (int i = 0; i < N; i++) B[i] = 0.0f;

    for (int w = 0; w < WARMUP; w++) stride2_kernel(A, B, N);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++) stride2_kernel(A, B, N);
    double ms = (double)(clock_ns() - t0) / TIMED / 1e6;

    printf("{\"kernel\":\"stride2_16k\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n", N, ms);
    free(A); free(B);
    return 0;
}
