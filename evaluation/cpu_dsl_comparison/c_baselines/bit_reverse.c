/* bit_reverse.c — scatter A[i] to B[bitrev20(i)].
 *
 * Tier-3 verification.  Non-affine scatter; clang -O3 emits scalar.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define DEFAULT_N (1 << 20)  /* 20-bit indices */
#define WARMUP    10
#define TIMED     100

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

__attribute__((noinline))
static void kernel(const float * __restrict A, float * __restrict B, int N) {
    for (int i = 0; i < N; i++) {
        unsigned int j = ((i & 0x55555u) << 1) | ((i >> 1) & 0x55555u);
        j = ((j & 0x33333u) << 2) | ((j >> 2) & 0x33333u);
        j = ((j & 0x0F0F0u) << 4) | ((j >> 4) & 0x0F0F0u);
        j = ((j & 0x00FF0u) << 8) | ((j >> 8) & 0x000FFu);
        B[j] = A[i];
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    float *A = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    float *B = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    if (!A || !B) { perror("alloc"); return 1; }

    unsigned int seed = 42u;
    for (int i = 0; i < N; i++) {
        seed = seed * 1103515245u + 12345u;
        A[i] = (float)((seed >> 16) & 0xff) / 128.0f - 1.0f;
        B[i] = 0.0f;
    }

    for (int w = 0; w < WARMUP; w++) kernel(A, B, N);
    long long mn = -1;
    for (int t = 0; t < 5; t++) {
        long long s = clock_ns();
        for (int it = 0; it < TIMED; it++) kernel(A, B, N);
        long long e = clock_ns();
        long long ns = (e - s) / TIMED;
        if (mn < 0 || ns < mn) mn = ns;
    }
    volatile float sk = B[0]; (void)sk;
    printf("{\"kernel\":\"bit_reverse\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, mn / 1.0e6);
    free(A); free(B);
    return 0;
}
