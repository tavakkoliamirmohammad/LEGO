/* spmv_indirect.c — indirect load: B[i] = A[idx[i]] * 2.0
 *
 * Real-world pattern: sparse matrix-vector multiply (CSR), particle
 * neighbour lookups, mesh connectivity. Both gcc and clang typically
 * scalarize this — the index expression is loaded from memory, defeating
 * affine analysis. The vector.gather instruction (vpgatherqd) is what
 * LEGO emits.
 *
 * Compile:
 *   gcc -O3 -o spmv_indirect_O3 spmv_indirect.c
 *   gcc -O3 -march=native -mavx512f -ffast-math -o spmv_indirect_agg spmv_indirect.c
 *   clang -O3 -march=native -mavx512f -ffast-math -o spmv_indirect_clang spmv_indirect.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define DEFAULT_N (1 << 20)
#define WARMUP    100
#define TIMED     500

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

__attribute__((noinline))
static void kernel(const float *A, const long *idx, float *B, int N) {
    for (int i = 0; i < N; i++) {
        B[i] = A[idx[i]] * 2.0f;
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    float *A = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    long  *idx = (long *)aligned_alloc(64, (size_t)N * sizeof(long));
    float *B = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    if (!A || !idx || !B) { perror("alloc"); return 1; }

    /* Pseudorandom permutation via Fisher–Yates with a fixed seed. */
    for (int i = 0; i < N; i++) idx[i] = i;
    unsigned int seed = 42u;
    for (int i = N - 1; i > 0; i--) {
        seed = seed * 1103515245u + 12345u;
        int j = (int)(seed % (unsigned)(i + 1));
        long t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    for (int i = 0; i < N; i++) A[i] = (float)((i * 31u + 7u) & 0xff) / 255.0f;

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
    printf("{\"kernel\":\"spmv_indirect\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, mn / 1.0e6);
    free(A); free(idx); free(B);
    return 0;
}
