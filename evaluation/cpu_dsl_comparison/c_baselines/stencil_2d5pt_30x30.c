/*
 * stencil_2d5pt_30x30.c — 2D 5-point (jacobi-style) stencil on flat 30×30 buffer.
 *
 * Used as the reference for candidate 37_stencil_nonpow2_brick.
 *
 * Matches the LEGO kernel exactly:
 *   grid = (_INNER,), _INNER = (NX-2)*NY = 28*30 = 840
 *   B[flat] = (A[flat-NY] + A[flat+NY] + A[flat-1] + A[flat+1]) * 0.25
 *   where flat = n + NY  (skip first row of NY=30 elements)
 *
 * The 30x30 grid is non-power-of-2 — the key feature being tested in
 * candidate 37 is that the vectorizer correctly handles tail loops when
 * the trip count (840) is not divisible by the vector width (16).
 *
 * Compile:
 *   gcc -O3 -o stencil_2d5pt_30x30_O3 stencil_2d5pt_30x30.c
 *   gcc -O3 -march=native -mavx512f -ffast-math -o stencil_2d5pt_30x30_agg stencil_2d5pt_30x30.c
 * Output: {"kernel":"stencil_2d5pt_30x30","N":840,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NX      30
#define NY      30
#define N_FLAT  (NX * NY)
#define N_INNER ((NX - 2) * NY)   /* 28 * 30 = 840 interior elements */
#define OFFSET  NY                 /* skip first row: flat = n + NY */
#define WARMUP  200
#define TIMED   2000

static void __attribute__((noinline))
jacobi2d_flat(const float * __restrict__ A, float * __restrict__ B) {
    for (int n = 0; n < N_INNER; n++) {
        int f = n + OFFSET;
        B[f] = (A[f - NY] + A[f + NY] + A[f - 1] + A[f + 1]) * 0.25f;
    }
}

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(void) {
    float *A = (float *)malloc(N_FLAT * sizeof(float));
    float *B = (float *)malloc(N_FLAT * sizeof(float));
    if (!A || !B) { fprintf(stderr, "OOM\n"); return 1; }

    for (int i = 0; i < N_FLAT; i++) { A[i] = i * 0.001f; B[i] = 0.0f; }

    for (int w = 0; w < WARMUP; w++) jacobi2d_flat(A, B);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++) jacobi2d_flat(A, B);
    double ms = (double)(clock_ns() - t0) / TIMED / 1e6;

    printf("{\"kernel\":\"stencil_2d5pt_30x30\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N_INNER, ms);
    free(A); free(B);
    return 0;
}
