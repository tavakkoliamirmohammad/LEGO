/* 2D 25-point box stencil (5x5 footprint).  R=2.
 * For each cell, read all 25 cells within a 5x5 box centred at (i, j).
 * This stresses spatial locality: row-major touches 5 different rows;
 * Morton keeps all 25 cells inside one or two 8x8 Morton blocks.
 *
 * Templates: __N__, __NBITS__, __WARMUP__, __ITERS__, __TRIALS__, __MORTON__
 *   __MORTON__ = 0 → row-major addressing
 *   __MORTON__ = 1 → bit-magic Morton addressing
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
#define USE_MORTON __MORTON__
#define R 2

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

#if USE_MORTON
static inline uint32_t addr(uint32_t i, uint32_t j) {
    return spread16(i) | (spread16(j) << 1);
}
#else
static inline uint32_t addr(uint32_t i, uint32_t j) {
    return i * (uint32_t)N + j;
}
#endif

__attribute__((noinline))
void stencil25(const float * __restrict A, float * __restrict B) {
    const float w = 1.0f / 25.0f;
    for (int i = R; i < N - R; i++) {
        for (int j = R; j < N - R; j++) {
            float sum = 0.0f;
            for (int di = -R; di <= R; di++) {
                for (int dj = -R; dj <= R; dj++) {
                    sum += A[addr(i + di, j + dj)];
                }
            }
            B[addr(i, j)] = sum * w;
        }
    }
}

int main(void) {
    size_t NN = (size_t)N * (size_t)N;
    float *A = aligned_alloc(64, NN * sizeof(float));
    float *B = aligned_alloc(64, NN * sizeof(float));
    for (size_t k = 0; k < NN; k++) {
        A[k] = (float)((k * 1103515245u + 12345u) & 0xffff) / 65536.0f;
        B[k] = 0;
    }
    for (int w = 0; w < WARMUP; w++) stencil25(A, B);
    long long mn = -1;
    for (int t = 0; t < TRIALS; t++) {
        long long s = clock_ns();
        for (int it = 0; it < ITERS; it++) {
            stencil25(A, B);
            float *tmp = A; A = (float *)B; B = tmp;
        }
        long long e = clock_ns();
        long long ns = (e - s) / ITERS;
        if (mn < 0 || ns < mn) mn = ns;
    }
    volatile float sk = B[2 * N + 2]; (void)sk;
    printf("{\"N\":%d,\"morton\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, USE_MORTON, mn / 1.0e6);
    free(A); free(B); return 0;
}
