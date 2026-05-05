/* predicated_fma.c — predicated FMA:
 *   if (mask[i] > 0) C[i] = A[i] * B[i] + C[i];
 *
 * Pattern: data-dependent predicated update. clang/gcc may emit a
 * masked store with vector compare-and-select, OR scalarize. LEGO's
 * R17 path emits ``vector.maskedstore`` cleanly.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "_bench_assume.h"
#define DEFAULT_N (1 << 20)
#define WARMUP    100
#define TIMED     500

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

__attribute__((noinline))
static void kernel(const float * __restrict A,
                   const float * __restrict B,
                   const float * __restrict mask,
                   float * __restrict C, int N) {
    BENCH_ASSUME_N(N);
    for (int i = 0; i < N; i++) {
        if (mask[i] > 0.0f) {
            C[i] = A[i] * B[i] + C[i];
        }
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    float *A = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    float *B = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    float *m = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    float *C = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    if (!A || !B || !m || !C) { perror("alloc"); return 1; }

    unsigned int seed = 42u;
    for (int i = 0; i < N; i++) {
        seed = seed * 1103515245u + 12345u;
        A[i] = (float)((seed >> 16) & 0xff) / 128.0f - 1.0f;
        seed = seed * 1103515245u + 12345u;
        B[i] = (float)((seed >> 16) & 0xff) / 128.0f - 1.0f;
        seed = seed * 1103515245u + 12345u;
        m[i] = (float)((seed >> 16) & 0xff) / 128.0f - 1.0f;
        C[i] = 0.0f;
    }

    for (int w = 0; w < WARMUP; w++) kernel(A, B, m, C, N);

    long long mn = -1;
    for (int t = 0; t < 5; t++) {
        long long s = clock_ns();
        for (int it = 0; it < TIMED; it++) kernel(A, B, m, C, N);
        long long e = clock_ns();
        long long ns = (e - s) / TIMED;
        if (mn < 0 || ns < mn) mn = ns;
    }
    volatile float sk = C[0]; (void)sk;
    printf("{\"kernel\":\"predicated_fma\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, mn / 1.0e6);
    free(A); free(B); free(m); free(C);
    return 0;
}
