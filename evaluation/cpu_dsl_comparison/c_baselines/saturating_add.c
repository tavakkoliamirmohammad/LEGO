/* saturating_add.c — clamp(A[i] + B[i], -1.0f, 1.0f).
 *
 * Tier-3 parity check.  clang -O3 -march=native vectorises with
 * vminps / vmaxps cleanly.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_N (1 << 20)
#define WARMUP    50
#define TIMED     500

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

__attribute__((noinline))
static void kernel(const float * __restrict A, const float * __restrict B,
                   float * __restrict C, int N) {
    for (int i = 0; i < N; i++) {
        float s = A[i] + B[i];
        if (s < -1.0f) s = -1.0f;
        if (s >  1.0f) s =  1.0f;
        C[i] = s;
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    float *A = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    float *B = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    float *C = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    if (!A || !B || !C) { perror("alloc"); return 1; }

    unsigned int seed = 42u;
    for (int i = 0; i < N; i++) {
        seed = seed * 1103515245u + 12345u;
        A[i] = (float)((seed >> 16) & 0xff) / 128.0f - 1.0f;
        seed = seed * 1103515245u + 12345u;
        B[i] = (float)((seed >> 16) & 0xff) / 128.0f - 1.0f;
        C[i] = 0.0f;
    }

    for (int w = 0; w < WARMUP; w++) kernel(A, B, C, N);
    long long mn = -1;
    for (int t = 0; t < 5; t++) {
        long long s = clock_ns();
        for (int it = 0; it < TIMED; it++) kernel(A, B, C, N);
        long long e = clock_ns();
        long long ns = (e - s) / TIMED;
        if (mn < 0 || ns < mn) mn = ns;
    }
    volatile float sk = C[0]; (void)sk;
    printf("{\"kernel\":\"saturating_add\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, mn / 1.0e6);
    free(A); free(B); free(C);
    return 0;
}
