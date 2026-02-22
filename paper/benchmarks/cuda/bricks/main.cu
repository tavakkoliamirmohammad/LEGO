#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do { \
    cudaError_t err__ = (x); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while(0)
#endif

#ifdef RD_RADIUS
    #define SIZE RD_RADIUS
#else
    #define SIZE 1   // default radius -> 7x7x7 window
#endif

__global__ void f3d_naive(float *in, float *out, float *c) {
    const int radius = SIZE;

    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;

    unsigned bx = blockIdx.x;
    unsigned by = blockIdx.y;
    unsigned bz = blockIdx.z;

    const int Bx = blockDim.x, By = blockDim.y, Bz = blockDim.z;

    // global coords
    const int gx = bx * Bx + i;
    const int gy = by * By + j;
    const int gz = bz * Bz + k;

    // --- Boundary condition check (skip threads whose stencil would go OOB) ---
    if (gx < radius || gx >= 384 - radius ||
        gy < radius || gy >= 384 - radius ||
        gz < radius || gz >= 384 - radius) {
        return;   // or write a clamped/ghost value here if you prefer
    }

    float base = in[1179648*bx + 3072*by + 8*bz + 147456*i + 384*j + k] * c[73*radius];

    #pragma unroll
    for (int i_diff = -radius; i_diff <= radius; i_diff++) {
        #pragma unroll
        for (int j_diff = -radius; j_diff <= radius; j_diff++) {
            #pragma unroll
            for (int k_diff = -radius; k_diff <= radius; k_diff++) {
                base += in[1179648*bx + 3072*by + 8*bz + 147456*i + 147456*i_diff + 384*j + 384*j_diff + k + k_diff] *
                        c[64*i_diff + 8*j_diff + k_diff + 73*radius];
            }
        }
    }

    out[1179648*bx + 3072*by + 8*bz + 147456*i + 384*j + k] = base;
}


__global__ void f3d_bricks(float *in, float *out, float *c) {
    const int radius = SIZE;

    unsigned i = threadIdx.x;
    unsigned j = threadIdx.y;
    unsigned k = threadIdx.z;

    const int Bx = blockDim.x, By = blockDim.y, Bz = blockDim.z;

    unsigned bx = blockIdx.x;
    unsigned by = blockIdx.y;
    unsigned bz = blockIdx.z;

    // global coords
    const int gx = bx * Bx + i;
    const int gy = by * By + j;
    const int gz = bz * Bz + k;

    // --- Boundary condition check (skip threads whose stencil would go OOB) ---
    if (gx < radius || gx >= 384 - radius ||
        gy < radius || gy >= 384 - radius ||
        gz < radius || gz >= 384 - radius) {
        return;   // or write a clamped/ghost value here if you prefer
    }

    float base = in[1179648*bx + 24576*by + 512*bz + 64*i + 8*j + k] * c[73*radius];

    #pragma unroll
    for (int i_diff = -radius; i_diff <= radius; i_diff++) {
        #pragma unroll
        for (int j_diff = -radius; j_diff <= radius; j_diff++) {
            #pragma unroll
            for (int k_diff = -radius; k_diff <= radius; k_diff++) {
                base += in[1179648*bx + 24576*by + 512*bz + 64*i + 64*i_diff + 8*j + 8*j_diff + k + k_diff] *
                        c[64*i_diff + 8*j_diff + k_diff + 73*radius];
            }
        }
    }

    out[1179648*bx + 24576*by + 512*bz + 64*i + 8*j + k] = base;
}

// ---------------------------
// Helper: ceilDiv
// ---------------------------
static inline dim3 ceilDiv3D(int nx, int ny, int nz, dim3 block) {
    return dim3((nx + block.x - 1) / block.x,
                (ny + block.y - 1) / block.y,
                (nz + block.z - 1) / block.z);
}

// ---------------------------
// Main program
// ---------------------------
int main(int argc, char** argv) {
    // Problem size (can override via CLI)
    int NX = 384, NY = 384, NZ = 384;

    const size_t N = static_cast<size_t>(NX) * NY * NZ;
    const size_t bytes = N * sizeof(float);

    printf("Dims: NX=%d NY=%d NZ=%d (total elements=%zu) | radius=%d | c is 8x8x8\n",
           NX, NY, NZ, N, SIZE);

    // Host allocations
    std::vector<float> h_in(N), h_out(N, 0.0f), h_out2(N, 0.0f);
    std::vector<float> h_c(8*8*8, 0.0f);

    // Initialize input with randoms and c with something deterministic
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t t = 0; t < N; ++t) h_in[t] = dist(rng);

    // Fill c[0..6] along each dim meaningfully; leave the last slice as zero padding.
    // Ex: simple separable-like mask where center is bigger
    auto cidx = [](int a,int b,int c){ return a*64 + b*8 + c; };
    for (int a = 0; a < 7; ++a)
        for (int b = 0; b < 7; ++b)
            for (int c = 0; c < 7; ++c) {
                float da = (a - SIZE), db = (b - SIZE), dc = (c - SIZE);
                float d2 = da*da + db*db + dc*dc;
                h_c[cidx(a,b,c)] = 1.0f / (1.0f + d2); // center largest, decays with distance
            }
    // Ensure center is visible
    h_c[cidx(SIZE,SIZE,SIZE)] += 1.0f;

    // Device allocations
    float *d_in = nullptr, *d_out = nullptr, *d_out2 = nullptr, *d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMalloc(&d_out2, bytes));
    CHECK_CUDA(cudaMalloc(&d_c,  8*8*8*sizeof(float)));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c,  h_c.data(),  8*8*8*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out,  0, bytes));
    CHECK_CUDA(cudaMemset(d_out2, 0, bytes));

    // Launch config (tweak as desired)
    dim3 block(8, 8, 8);
    dim3 grid = ceilDiv3D(NX, NY, NZ, block);

    // Warm-up
    const int WARPMUP = 25;
    for (int r = 0; r < WARPMUP; ++r) {
        f3d_naive<<<grid, block>>>(d_in, d_out, d_c);
    }
    CHECK_CUDA(cudaGetLastError());
    for (int r = 0; r < WARPMUP; ++r) {
        f3d_bricks<<<grid, block>>>(d_in, d_out2, d_c);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing helpers
    const int REPS = 100;
    cudaEvent_t start, stop, start2, stop2;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&start2));
    CHECK_CUDA(cudaEventCreate(&stop2));

    // Time f3d_naive
    CHECK_CUDA(cudaEventRecord(start));
    for (int r = 0; r < REPS; ++r) {
        f3d_naive<<<grid, block>>>(d_in, d_out, d_c);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_naive = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_naive, start, stop));

    // Time f3d_bricks
    CHECK_CUDA(cudaEventRecord(start2));
    for (int r = 0; r < REPS; ++r) {
        f3d_bricks<<<grid, block>>>(d_in, d_out2, d_c);
    }
    CHECK_CUDA(cudaEventRecord(stop2));
    CHECK_CUDA(cudaEventSynchronize(stop2));
    float ms_bricks = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_bricks, start2, stop2));

    // Report
    printf("Timing over %d repetitions:\n", REPS);
    printf("  f3d_naive : %.3f ms total (%.3f ms / launch)\n", ms_naive, ms_naive / REPS);
    printf("  f3d_bricks: %.3f ms total (%.3f ms / launch)\n", ms_bricks, ms_bricks / REPS);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start2));
    CHECK_CUDA(cudaEventDestroy(stop2));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_out2));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}