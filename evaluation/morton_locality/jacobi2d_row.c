/* Jacobi 2D 5-point stencil — row-major.  N from argv[1]. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARMUP 5
#define ITERS  50

static long long clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static int N_g;
__attribute__((noinline))
void jacobi_row(const float * __restrict A, float * __restrict B) {
    int N = N_g;
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            B[i*N + j] = 0.25f * (A[(i-1)*N + j] + A[(i+1)*N + j]
                                  + A[i*N + j-1] + A[i*N + j+1]);
        }
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    N_g = N;
    size_t NN = (size_t)N * (size_t)N;
    float *A = aligned_alloc(64, NN * sizeof(float));
    float *B = aligned_alloc(64, NN * sizeof(float));
    for (size_t k = 0; k < NN; k++) { A[k] = (float)((k * 1103515245u + 12345u) & 0xffff) / 65536.0f; B[k] = 0; }

    for (int w = 0; w < WARMUP; w++) jacobi_row(A, B);
    long long min_ns = -1;
    for (int t = 0; t < 5; t++) {
        long long start = clock_ns();
        for (int it = 0; it < ITERS; it++) {
            jacobi_row(A, B);
            float *tmp = A; A = (float *)B; B = tmp;
        }
        long long end = clock_ns();
        long long ns = (end - start) / ITERS;
        if (min_ns < 0 || ns < min_ns) min_ns = ns;
    }
    volatile float sink = B[N + 1]; (void)sink;
    printf("{\"kernel\":\"jacobi2d_row\",\"N\":%d,\"c_baseline_ms_per_call\":%.6f}\n",
           N, min_ns / 1.0e6);
    free(A); free(B);
    return 0;
}
