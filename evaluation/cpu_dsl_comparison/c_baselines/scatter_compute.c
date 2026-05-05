/* scatter_compute.c — polynomial evaluation + scatter store.
 *
 * The non-affine scatter store at the end forces clang/gcc to scalarise
 * the entire loop (no auto-vectoriser speculates vector compute when
 * the final write is non-affine). LEGO's vector.scatter path keeps the
 * polynomial in vector form.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "_bench_assume.h"
#define DEFAULT_N (1 << 20)
#define WARMUP    50
#define TIMED     500

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

__attribute__((noinline))
static void kernel(const float * __restrict A,
                   const long  * __restrict idx,
                   float       * __restrict B, int N) {
    BENCH_ASSUME_N(N);
    for (int i = 0; i < N; i++) {
        float x = A[i];
        float p = ((((x * 0.1f + 0.2f) * x + 0.3f) * x + 0.4f) * x + 0.5f) * x + 0.6f;
        B[idx[i]] = p;
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    float *A = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    long  *idx = (long *)aligned_alloc(64, (size_t)N * sizeof(long));
    float *B = (float *)aligned_alloc(64, (size_t)N * sizeof(float));

    for (int i = 0; i < N; i++) idx[i] = i;
    unsigned int seed = 42u;
    for (int i = N - 1; i > 0; i--) {
        seed = seed * 1103515245u + 12345u;
        int j = (int)(seed % (unsigned)(i + 1));
        long t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    for (int i = 0; i < N; i++) A[i] = (float)((i * 31u + 7u) & 0xff) / 255.0f - 0.5f;

    for (int w = 0; w < WARMUP; w++) kernel(A, idx, B, N);

    long long mn = -1;
    for (int t = 0; t < 5; t++) {
        long long s = clock_ns();
        for (int it = 0; it < TIMED; it++) kernel(A, idx, B, N);
        long long e = clock_ns();
        long long ns = (e - s) / TIMED;
        if (mn < 0 || ns < mn) mn = ns;
    }
    volatile float sk = B[0]; (void)sk;
    printf("{\"kernel\":\"scatter_compute\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, mn / 1.0e6);
    free(A); free(idx); free(B);
    return 0;
}
