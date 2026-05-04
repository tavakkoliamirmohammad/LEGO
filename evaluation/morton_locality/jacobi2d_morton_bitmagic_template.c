/* Jacobi 2D 5-point stencil — Morton (Z-order) layout, BIT-MAGIC encoding.
 * Lightweight (low warmup/iter counts) — for large N where each kernel call
 * already costs ~hundreds of milliseconds, fewer reps suffice.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define N __N__
#define NBITS __NBITS__
#define WARMUP __WARMUP__
#define ITERS  __ITERS__
#define TRIALS __TRIALS__

static long long clock_ns(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static inline uint32_t spread16(uint32_t x) {
    x = (x | (x << 8)) & 0x00FF00FFu;
    x = (x | (x << 4)) & 0x0F0F0F0Fu;
    x = (x | (x << 2)) & 0x33333333u;
    x = (x | (x << 1)) & 0x55555555u;
    return x;
}

static inline uint32_t morton(uint32_t i, uint32_t j) {
    return spread16(i) | (spread16(j) << 1);
}

__attribute__((noinline))
void jacobi_morton(const float * __restrict A, float * __restrict B) {
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            B[morton(i, j)] = 0.25f * (A[morton(i - 1, j)] + A[morton(i + 1, j)]
                                       + A[morton(i, j - 1)] + A[morton(i, j + 1)]);
        }
    }
}

int main(void) {
    size_t NN = (size_t)N * (size_t)N;
    float *A = aligned_alloc(64, NN * sizeof(float));
    float *B = aligned_alloc(64, NN * sizeof(float));
    for (size_t k = 0; k < NN; k++) { A[k] = (float)((k * 1103515245u + 12345u) & 0xffff) / 65536.0f; B[k] = 0; }

    for (int w = 0; w < WARMUP; w++) jacobi_morton(A, B);
    long long mn = -1;
    for (int t = 0; t < TRIALS; t++) {
        long long s = clock_ns();
        for (int it = 0; it < ITERS; it++) {
            jacobi_morton(A, B);
            float *tmp = A; A = (float *)B; B = tmp;
        }
        long long e = clock_ns();
        long long ns = (e - s) / ITERS;
        if (mn < 0 || ns < mn) mn = ns;
    }
    volatile float sk = B[N + 1]; (void)sk;
    printf("{\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n", N, mn / 1.0e6);
    free(A); free(B); return 0;
}
