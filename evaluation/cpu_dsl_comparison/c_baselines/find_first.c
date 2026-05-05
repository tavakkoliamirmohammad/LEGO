/* find_first.c — branchless find-first-index where A[i] > 0.5.
 *
 * Tier-1 find-first.  Mirrors the cpu_dsl candidate's float-counter
 * encoding (idx_f kept alongside i because cpu_dsl can't unify INDEX
 * vs F32 in compares).
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_N (1 << 20)
#define WARMUP    20
#define TIMED     200

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

__attribute__((noinline))
static void kernel(const float * __restrict A, float * __restrict out, int N) {
    float idx_f = (float)N;
    float i_f = 0.0f;
    for (int i = 0; i < N; i++) {
        if (A[i] > 0.5f) {
            if (i_f < idx_f) idx_f = i_f;
        }
        i_f += 1.0f;
    }
    out[0] = idx_f;
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    float *A = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    float out[1] = {0.0f};
    if (!A) { perror("alloc"); return 1; }

    unsigned int seed = 42u;
    for (int i = 0; i < N; i++) {
        seed = seed * 1103515245u + 12345u;
        A[i] = (float)((seed >> 16) & 0x7fff) / 32767.0f;
    }

    for (int w = 0; w < WARMUP; w++) kernel(A, out, N);
    long long mn = -1;
    for (int t = 0; t < 5; t++) {
        long long s = clock_ns();
        for (int it = 0; it < TIMED; it++) kernel(A, out, N);
        long long e = clock_ns();
        long long ns = (e - s) / TIMED;
        if (mn < 0 || ns < mn) mn = ns;
    }
    volatile float sk = out[0]; (void)sk;
    printf("{\"kernel\":\"find_first\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, mn / 1.0e6);
    free(A);
    return 0;
}
