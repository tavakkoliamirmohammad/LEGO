from lego import *

from lego.lego_c import *

a, i, j, k, radius, i_diff, j_diff, k_diff, bx, by, bz = symbols(
    'a i j k radius i_diff j_diff k_diff bx by bz', integer=True, positive=True)

N = 384
B = 8
normal = OrderBy(Row(N, N, N)).TileBy([N//B, N//B, N//B], [B, B, B])
bricks = OrderBy(Row(N // B, N//B, N//B), Row(B, B, B)
                 ).TileBy([N//B, N//B, N//B], [B, B, B])
const = OrderBy(Row(8, 8, 8)).TileBy([8, 8, 8])


def get_computation_indices(l):
    return l[bx, by, bz, i+a, j, k], l[bx, by, bz, i, j+a, k], l[bx, by, bz, i, j, k+a], l[bx, by, bz, i, j, k-a], l[bx, by, bz, i, j-a, k], l[bx, by, bz, i-a, j, k]

normal_indices = get_computation_indices(normal)
normal_in0_idx = normal_indices[0]
normal_in1_idx = normal_indices[1]
normal_in2_idx = normal_indices[2]
normal_in3_idx = normal_indices[3]
normal_in4_idx = normal_indices[4]
normal_in5_idx = normal_indices[5]

bricks_indices = get_computation_indices(bricks)
bricks_in0_idx = bricks_indices[0]
bricks_in1_idx = bricks_indices[1]
bricks_in2_idx = bricks_indices[2]
bricks_in3_idx = bricks_indices[3]
bricks_in4_idx = bricks_indices[4]
bricks_in5_idx = bricks_indices[5]

normal_out_idx = normal[bx, by, bz, i, j, k]
bricks_out_idx = bricks[bx, by, bz, i, j, k]

c_printer = LEGOCCodePrinter()
def _p(expr):
    return c_printer.doprint(sp.simplify(expr))

template = f"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do {{ \\
    cudaError_t err__ = (x); \\
    if (err__ != cudaSuccess) {{ \\
        fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \\
        std::exit(1); \\
    }} \\
}} while(0)
#endif

#ifdef RD_RADIUS
    #define SIZE RD_RADIUS
#else
    #define SIZE 1   // default radius -> 7x7x7 window
#endif

__global__ void laplacian_naive(float *in, float *out, float *c) {{
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
    if (gx < radius || gx >= {N} - radius ||
        gy < radius || gy >= {N} - radius ||
        gz < radius || gz >= {N} - radius) {{
        return;   // or write a clamped/ghost value here if you prefer
    }}

    float base = in[{_p(normal_out_idx)}] * c[0];

    #pragma unroll
    for (int a = 1; a <= radius; a++) {{
         base += c[a] * (
            in[{_p(normal_in0_idx)}] + in[{_p(normal_in1_idx)}] + in[{_p(normal_in2_idx)}] +
            in[{_p(normal_in3_idx)}] + in[{_p(normal_in4_idx)}] + in[{_p(normal_in5_idx)}]);
    }}

    out[{_p(normal_out_idx)}] = base;
}}


__global__ void laplacian_bricks(float *in, float *out, float *c) {{
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
    if (gx < radius || gx >= {N} - radius ||
        gy < radius || gy >= {N} - radius ||
        gz < radius || gz >= {N} - radius) {{
        return;   // or write a clamped/ghost value here if you prefer
    }}

    float base = in[{_p(bricks_out_idx)}] * c[0];

    #pragma unroll
    for (int a = 1; a <= radius; a++) {{
         base += c[a] * (
            in[{_p(bricks_in0_idx)}] + in[{_p(bricks_in1_idx)}] + in[{_p(bricks_in2_idx)}] +
            in[{_p(bricks_in3_idx)}] + in[{_p(bricks_in4_idx)}] + in[{_p(bricks_in5_idx)}]);
    }}

    out[{_p(bricks_out_idx)}] = base;
}}

// ---------------------------
// Helper: ceilDiv
// ---------------------------
static inline dim3 ceilDiv3D(int nx, int ny, int nz, dim3 block) {{
    return dim3((nx + block.x - 1) / block.x,
                (ny + block.y - 1) / block.y,
                (nz + block.z - 1) / block.z);
}}

// ---------------------------
// Main program
// ---------------------------
int main(int argc, char** argv) {{
    // Problem size (can override via CLI)
    int NX = {N}, NY = {N}, NZ = {N};

    const size_t n_size = static_cast<size_t>(NX) * NY * NZ;
    const size_t bytes = n_size * sizeof(float);

    printf("Dims: NX=%d NY=%d NZ=%d (total elements=%zu) | radius=%d | c is 8x8x8\\n",
           NX, NY, NZ, n_size, SIZE);

    // Host allocations
    std::vector<float> h_in(n_size), h_out(n_size, 0.0f), h_out2(n_size, 0.0f);
    std::vector<float> h_c(8*8*8, 0.0f);

    // Initialize input with randoms and c with something deterministic
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t t = 0; t < n_size; ++t) h_in[t] = dist(rng);

    // Fill c[0..6] along each dim meaningfully; leave the last slice as zero padding.
    // Ex: simple separable-like mask where center is bigger
    auto cidx = [](int a,int b,int c){{ return a*64 + b*8 + c; }};
    for (int a = 0; a < 7; ++a)
        for (int b = 0; b < 7; ++b)
            for (int c = 0; c < 7; ++c) {{
                float da = (a - SIZE), db = (b - SIZE), dc = (c - SIZE);
                float d2 = da*da + db*db + dc*dc;
                h_c[cidx(a,b,c)] = 1.0f / (1.0f + d2); // center largest, decays with distance
            }}
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

    const int WARPMUP = 25;
    // Warm-up
    for (int r = 0; r < WARPMUP; ++r) {{
        laplacian_naive<<<grid, block>>>(d_in, d_out, d_c);
    }}
    CHECK_CUDA(cudaGetLastError());
    for (int r = 0; r < WARPMUP; ++r) {{
        laplacian_bricks<<<grid, block>>>(d_in, d_out2, d_c);
    }}
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
    for (int r = 0; r < REPS; ++r) {{
        laplacian_naive<<<grid, block>>>(d_in, d_out, d_c);
    }}
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_naive = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_naive, start, stop));

    // Time f3d_bricks
    CHECK_CUDA(cudaEventRecord(start2));
    for (int r = 0; r < REPS; ++r) {{
        laplacian_bricks<<<grid, block>>>(d_in, d_out2, d_c);
    }}
    CHECK_CUDA(cudaEventRecord(stop2));
    CHECK_CUDA(cudaEventSynchronize(stop2));
    float ms_bricks = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_bricks, start2, stop2));

    // Report
    printf("Timing over %d repetitions:\\n", REPS);
    printf("  laplacian_naive : %.3f ms total (%.3f ms / launch)\\n", ms_naive, ms_naive / REPS);
    printf("  laplacian_bricks: %.3f ms total (%.3f ms / launch)\\n", ms_bricks, ms_bricks / REPS);

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
}}
"""
print(template)


