/* count_if.c — predicated count: cnt += (A[i] > 0).
 *
 * Tier-1 predicated count.  clang -O3 -march=native lowers to a popcount-
 * over-mask using vpcmpps + vpopcntq.
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
static void kernel(const float * __restrict A, float * __restrict out, int N) {
    float cnt = 0.0f;
    for (int i = 0; i < N; i++) {
        if (A[i] > 0.0f) cnt += 1.0f;
    }
    out[0] = cnt;
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    float *A = (float *)aligned_alloc(64, (size_t)N * sizeof(float));
    float out[1] = {0.0f};
    if (!A) { perror("alloc"); return 1; }

    unsigned int seed = 42u;
    for (int i = 0; i < N; i++) {
        seed = seed * 1103515245u + 12345u;
        A[i] = (float)((seed >> 16) & 0xff) / 128.0f - 1.0f;
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
    printf("{\"kernel\":\"count_if\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, mn / 1.0e6);
    free(A);
    return 0;
}
