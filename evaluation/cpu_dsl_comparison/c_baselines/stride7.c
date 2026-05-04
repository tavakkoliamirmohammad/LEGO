/* stride7.c — non-power-of-2 strided gather: B[i] = A[i*7] * 2.0
 *
 * Pattern: stride-7 access (or any non-pow-2 stride). clang's
 * deinterleave heuristics target stride-2/4/8; non-pow-2 strides
 * typically fall through to a scalar loop. gcc similar.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_N (1 << 18)   /* 256K logical */
#define STRIDE    7
#define WARMUP    100
#define TIMED     500

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

__attribute__((noinline))
static void kernel(const float * __restrict A, float * __restrict B, int N) {
    for (int i = 0; i < N; i++) {
        B[i] = A[i * STRIDE] * 2.0f;
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    int N_PHYS = N * STRIDE;
    float *A = (float *)aligned_alloc(64, (size_t)N_PHYS * sizeof(float));
    float *B = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    if (!A || !B) { perror("alloc"); return 1; }
    for (int i = 0; i < N_PHYS; i++) A[i] = (float)((i * 31u + 7u) & 0xff) / 255.0f;

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
    printf("{\"kernel\":\"stride7\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, mn / 1.0e6);
    free(A); free(B);
    return 0;
}
