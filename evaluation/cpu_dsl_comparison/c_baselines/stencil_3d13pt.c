/*
 * stencil_3d13pt.c — 3D 13-point stencil on 32×32×32 flat buffer.
 *
 * Used as the reference for candidate 20_bricklib_3d13pt.
 *
 * Matches the LEGO kernel exactly:
 *   grid = (_INNER,), _INNER = (NX-2)*NY*NZ = 30720
 *   B[flat] = (A[flat] + 6 face + 6 edge_diagonal) * (1/13)
 *   where flat = n + NX*NY  (skip first XY-plane)
 *
 * Compile:
 *   gcc -O3 -o stencil_3d13pt_O3 stencil_3d13pt.c
 *   gcc -O3 -march=native -mavx512f -ffast-math -o stencil_3d13pt_agg stencil_3d13pt.c
 * Output: {"kernel":"stencil_3d13pt","N":30720,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NX 32
#define NY 32
#define NZ 32
#define N_FLAT  (NX * NY * NZ)
#define NYNZ    (NY * NZ)
#define N_INNER ((NX - 2) * NY * NZ)
#define OFFSET  (NX * NY)
#define WARMUP  100
#define TIMED   1000

static void __attribute__((noinline))
stencil_13pt(const float * __restrict__ A, float * __restrict__ B) {
    const float inv13 = 1.0f / 13.0f;
    for (int n = 0; n < N_INNER; n++) {
        int f = n + OFFSET;
        B[f] = (A[f]
                + A[f - NYNZ]        /* -x */
                + A[f + NYNZ]        /* +x */
                + A[f - NZ]          /* -y */
                + A[f + NZ]          /* +y */
                + A[f - 1]           /* -z */
                + A[f + 1]           /* +z */
                + A[f - NYNZ - NZ]   /* -x-y diagonal */
                + A[f + NYNZ + NZ]   /* +x+y diagonal */
                + A[f - NYNZ + NZ]   /* -x+y diagonal */
                + A[f + NYNZ - NZ]   /* +x-y diagonal */
                + A[f - NYNZ - 1]    /* -x-z diagonal */
                + A[f + NYNZ + 1]    /* +x+z diagonal */
                ) * inv13;
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

    for (int w = 0; w < WARMUP; w++) stencil_13pt(A, B);

    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++) stencil_13pt(A, B);
    double ms = (double)(clock_ns() - t0) / TIMED / 1e6;

    printf("{\"kernel\":\"stencil_3d13pt\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N_INNER, ms);
    free(A); free(B);
    return 0;
}
