/*
 * saxpy.c — y[i] = a*x[i] + y[i]  C-O3-march=native baseline.
 *
 * Compile: gcc -O3 -march=native -mavx512f -ffast-math -o saxpy saxpy.c
 *
 * Usage: ./saxpy [N]        (default N = 1048576 = 1M)
 * Output (last line): JSON {"kernel":"saxpy","N":<N>,"c_baseline_ms_per_call":<ms>}
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define DEFAULT_N (1 << 20)   /* 1M elements */
#define WARMUP    100
#define TIMED     1000

static void saxpy_kernel(float a, const float *X, float *Y, int N) {
    for (int i = 0; i < N; i++)
        Y[i] = a * X[i] + Y[i];
}

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    int N = DEFAULT_N;
    if (argc > 1) N = atoi(argv[1]);

    float *X = (float *)malloc(N * sizeof(float));
    float *Y = (float *)malloc(N * sizeof(float));
    if (!X || !Y) { fprintf(stderr, "OOM\n"); return 1; }

    /* Initialize with simple values */
    for (int i = 0; i < N; i++) {
        X[i] = (float)(i % 1000) * 0.001f;
        Y[i] = (float)(i % 997) * 0.001f;
    }
    float a = 2.5f;

    /* Warmup */
    for (int w = 0; w < WARMUP; w++)
        saxpy_kernel(a, X, Y, N);

    /* Timed: one large block */
    long long t0 = clock_ns();
    for (int t = 0; t < TIMED; t++)
        saxpy_kernel(a, X, Y, N);
    long long t_total = clock_ns() - t0;

    double ms_per_call = (double)t_total / TIMED / 1e6;

    /* Print result as JSON on last line */
    printf("{\"kernel\":\"saxpy\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, ms_per_call);

    free(X); free(Y);
    return 0;
}
