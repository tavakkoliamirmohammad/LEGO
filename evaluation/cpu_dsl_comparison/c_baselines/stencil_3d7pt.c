/*
 * stencil_3d7pt.c — 3D 7-point stencil on 32×32×32 flat buffer.
 *
 * Used as the reference for candidates 19_bricklib_3d7pt, 21_heat3d_brick.
 * Also partially related to: 20_bricklib_3d13pt, 22_jacobi2d_brick, 37_stencil_nonpow2_brick.
 *
 * Kernel: B[flat] = (A[flat] + A[flat±NZ] + A[flat±NYNZ] + A[flat±1]) * 0.5
 *
 * Compile:
 *   gcc -O3 -march=native -mavx512f -ffast-math -o stencil_3d7pt stencil_3d7pt.c
 * Output: {"kernel":"stencil_3d7pt","N":30720,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NX 32
#define NY 32
#define NZ 32
#define N_FLAT (NX * NY * NZ)
#define NYNZ   (NY * NZ)
#define OFFSET (NX * NY)
#define INNER  ((NX-2) * NY * NZ)
#define WARMUP 100
#define TIMED  1000

static void __attribute__((noinline))
stencil_7pt(const float * __restrict__ A, float * __restrict__ B, int n_inner, int offset,
            int nz, int nynz) {
    for (int n = 0; n < n_inner; n++) {
        int f = n + offset;
        B[f] = (A[f]
                + A[f - nynz]
                + A[f + nynz]
                + A[f - nz]
                + A[f + nz]
                + A[f - 1]
                + A[f + 1]) * 0.5f;
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

    for (int w = 0; w < WARMUP; w++) stencil_7pt(A, B, INNER, OFFSET, NZ, NYNZ);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++) stencil_7pt(A, B, INNER, OFFSET, NZ, NYNZ);
    double ms = (double)(clock_ns() - t0) / TIMED / 1e6;

    printf("{\"kernel\":\"stencil_3d7pt\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           INNER, ms);
    free(A); free(B);
    return 0;
}
