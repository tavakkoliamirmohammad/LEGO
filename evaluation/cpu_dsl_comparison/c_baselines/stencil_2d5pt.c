/*
 * stencil_2d5pt.c — 2D 5-point (jacobi-style) stencil on flat 256×256 buffer.
 *
 * Used as the reference for candidate 22_jacobi2d_brick.
 *
 * Matches the LEGO kernel exactly:
 *   grid = (N_INNER,), N_INNER = (NX-2)*NY = 254*256 = 65024
 *   B[flat] = (A[flat-NY] + A[flat+NY] + A[flat-1] + A[flat+1]) * 0.25
 *   where flat = n + NY  (skip first NY = 256 elements)
 *
 * This is a flat-loop formulation — gcc -O3 auto-vectorizes it because the
 * strided neighbors (±NY = ±256) are at a FIXED offset and the inner loop
 * is unit-stride in the index variable n.
 *
 * Compile:
 *   gcc -O3 -o stencil_2d5pt_O3 stencil_2d5pt.c
 *   gcc -O3 -march=native -mavx512f -ffast-math -o stencil_2d5pt_agg stencil_2d5pt.c
 * Output: {"kernel":"stencil_2d5pt","N":65024,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NX      256
#define NY      256
#define N_FLAT  (NX * NY)
#define N_INNER ((NX - 2) * NY)   /* 254 * 256 = 65024 interior elements */
#define OFFSET  NY                 /* skip first row: flat = n + NY */
#define WARMUP  100
#define TIMED   1000

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

    printf("{\"kernel\":\"stencil_2d5pt\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N_INNER, ms);
    free(A); free(B);
    return 0;
}
