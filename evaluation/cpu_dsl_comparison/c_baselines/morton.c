/*
 * morton.c — Morton-encoded (Z-order) indexing C-O3-march=native baseline.
 *
 * Computes B[i] = A[morton_decode(i)] where morton_decode performs a full
 * 2D Z-Morton deinterleave (separate even and odd bits → row/col indices,
 * then row*N + col → flat linear index).
 *
 * GCC-O3 cannot auto-vectorize Morton decode due to bit-interleave operations
 * (andi/shri/ori with non-trivial masks) that form irregular scatter/gather
 * access patterns.  The LEGO vectorizer should outperform this baseline by
 * emitting vector.gather (AVX-512 vgatherdpd / vpgatherdq).
 *
 * Compile: gcc -O3 -march=native -mavx512f -ffast-math -o morton morton.c
 *
 * Usage: ./morton [N]      (default N = 1048576 = 1M; must be power of 2)
 * Output: {"kernel":"morton","N":<N>,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define DEFAULT_N (1 << 20)   /* 1M elements */
#define WARMUP    50
#define TIMED     200

/* Separate even bits → one coordinate of the Morton index.
 * Uses the standard "bit scatter" method (Knuth, TAOCP vol 4A). */
static inline uint32_t morton_extract_even(uint32_t x) {
    x = x & 0x55555555u;
    x = (x | (x >> 1u))  & 0x33333333u;
    x = (x | (x >> 2u))  & 0x0F0F0F0Fu;
    x = (x | (x >> 4u))  & 0x00FF00FFu;
    x = (x | (x >> 8u))  & 0x0000FFFFu;
    return x;
}

/* row = even bits of z, col = even bits of (z >> 1). */
static inline int morton_row(uint32_t z) { return (int)morton_extract_even(z); }
static inline int morton_col(uint32_t z) { return (int)morton_extract_even(z >> 1u); }

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    int N = DEFAULT_N;
    if (argc > 1) N = atoi(argv[1]);

    /* N must be a perfect square so that row/col indices stay in-bounds. */
    int side = 1;
    while (side * side < N) side <<= 1;
    int NN = side * side;  /* may differ from N if N was not a perfect square */

    double *A = (double *)aligned_alloc(64, (size_t)NN * sizeof(double));
    double *B = (double *)aligned_alloc(64, (size_t)NN * sizeof(double));
    if (!A || !B) { perror("alloc"); return 1; }

    for (int i = 0; i < NN; i++) A[i] = (double)i;

    /* noinline kernel wrapper to prevent the outer timing loop from
     * being fused with the inner element loop (loop interchange). */
    /* Warmup — touches A and B, fills caches, trains branch predictors. */
    for (int w = 0; w < WARMUP; w++) {
        for (int i = 0; i < NN; i++) {
            int r = morton_row((uint32_t)i);
            int c = morton_col((uint32_t)i);
            B[i] = A[(r * side + c) & (NN - 1)];
            __asm__ volatile("" ::: "memory");  /* prevent loop interchange */
        }
    }

    /* Timed block: a single wall-clock interval over TIMED calls. */
    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++) {
        for (int i = 0; i < NN; i++) {
            int r = morton_row((uint32_t)i);
            int c = morton_col((uint32_t)i);
            B[i] = A[(r * side + c) & (NN - 1)];
        }
        __asm__ volatile("" ::: "memory");  /* prevent loop interchange across t-iterations */
    }
    long long t_total = clock_ns() - t0;

    double ms_per_call = (double)t_total / TIMED / 1e6;

    /* Prevent dead-code elimination: print a checksum. */
    double sum = 0.0;
    for (int i = 0; i < NN; i++) sum += B[i];
    (void)sum;

    printf("{\"kernel\":\"morton\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           NN, ms_per_call);

    free(A); free(B);
    return 0;
}
