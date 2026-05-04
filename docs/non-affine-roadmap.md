# Non-affine vectorization roadmap

> Patterns where modern clang -O3 (clang 20.x) auto-vectorizer scalarizes
> entirely on x86 + AVX-512. Each is a real workload; each has a clean
> AVX-512-friendly lowering through upstream MLIR vector dialect ops.

The empirical probe at `/tmp/clang_probe/` measured 14 non-affine patterns.
Clang vectorizes affine + standard gather/scatter/predicated cases (k01–03,
k06–07, k10), but **scalarizes** the patterns below.

## Targets

| # | Pattern | Workload | Upstream op | Hardware backing |
|---|---------|----------|-------------|------------------|
| **1** | **Compaction** `B[k++] = A[i] if cond[i]` | stream filter, sparse-to-dense | `vector.compressstore` | x86 `vpcompressps`, ARM SVE `compact + st1` |
| **2** | **Histogram (atomic-like)** `count[bin[i]]++` | image processing, particle binning, sort | `vector.gather` + `vector.scatter` + serialization on conflict | x86 `vpconflictd` (BMI/AVX-512 CD), or scalar fallback per conflict-class |
| **3** | **Argmin / argmax** `(val,idx) reduction` | ML inference, search | `vector.reduction` with paired index | tree-shuffle reduce |
| **4** | **Prefix scan** `B[i] = sum(A[0..i])` | sorting, parallel allocation | `vector.shuffle` tree | upcasts to LLVM scan intrinsic on x86 |

## Architecture principle (per user directive)

Use **upstream MLIR vector dialect ops first** (`vector.compressstore`,
`vector.gather`, `vector.scatter`, `vector.reduction`, `vector.shuffle`).
Lower hardware-specific intrinsics (`vpconflictd`, BMI2 PDEP/PEXT) **only
as a last resort**, and only when the pattern is provably architecture-bound
and upstream cannot emit a competitive sequence.

This buys cross-target portability: the same kernel compiles to AVX-512,
ARM SVE/SVE2, and (eventually) GPU.

## Compaction (target #1) — design

### Source pattern (cpu_dsl)
```python
@cpu_kernel
def compact(A: Buffer[N], cond: Buffer[N], B: Buffer[N], cnt: Buffer<i64>(1)):
    k = 0
    for i in range(N):
        if cond[i] > 0.0:
            B[k] = A[i]
            k = k + 1
    cnt[0] = k
```

### Recognizer (LegoVectorize.cpp)
Detect at admission time:
- `scf.for` has exactly one iter_arg, of `index` type
- Body shape: `load(A[i])`, `load(cond[i])`, `cmp`, `scf.if(p) -> index { store(B,k,v); yield k+1 } else { yield k }`
- The address chains for A[i], cond[i] are unit-stride; B[k] is data-dependent

When matched, set `compactionLoop = true` and bypass the per-access dispatcher.

### Emission (new helper `emitCompactLoop`)
Strip-mine by L=16 (lanes for f32 on AVX-512) or L=8 (f64). Emit:
```mlir
scf.for %i = 0 to N step L iter_args(%k = 0) {
  %vec_v   = vector.transfer_read A[%i] : vector<LxT>
  %vec_c   = vector.transfer_read cond[%i] : vector<LxT>
  %mask    = arith.cmpf ogt, %vec_c, %zero : vector<LxI1>
  vector.compressstore B[%k], %mask, %vec_v
  %popcnt  = math.ctpop ( vector.bitcast %mask to iL )
  %k_new   = %k + popcnt
  scf.yield %k_new
}
```

Tail loop handles the (N mod L) remainder via the original scalar body
(emitTailBody — already exists).

### Estimated cost
~250 LOC in LegoVectorize.cpp + 50 LOC FileCheck test + 70 LOC Python candidate +
50 LOC C baseline. **Half a day of focused work.**

## Histogram (target #2) — design

### Source pattern
```python
@cpu_kernel
def histogram(bin: Buffer<i64>(N), count: Buffer<i32>(K)):
    for i in range(N):
        count[bin[i]] = count[bin[i]] + 1
```

### Why clang scalarizes
The implicit memory dependence on `count[bin[i]]` (read-modify-write) means
two lanes with the same `bin[i]` race. Clang's auto-vec rejects vectorizing
the entire loop because of the conflict possibility.

### Vectorized emission (upstream-first)
```mlir
scf.for %i = 0 to N step L {
  %vec_b   = vector.transfer_read bin[%i] : vector<LxI64>
  // Gather current counts for these bins
  %vec_c   = vector.gather count, %vec_b, %trueMask : vector<LxI32>
  %vec_inc = arith.constant dense<1> : vector<LxI32>
  %vec_n   = arith.addi %vec_c, %vec_inc
  // Need to detect duplicates in vec_b before scatter, OR serialize via
  // conflict-detection. Two strategies:
  //
  // (A) UPSTREAM: emit an inner serialization loop over duplicates using
  //     vector.broadcast + arith.cmpi + count_active_lanes. Slower but
  //     portable.
  //
  // (B) HARDWARE FALLBACK: emit llvm.x86.avx512.conflict.d (if available)
  //     to compute conflict mask, then per-class scatter. Faster on x86.
  vector.scatter count, %vec_b, %trueMask, %vec_n : vector<LxI64>, ...
}
```

Strategy A first (portable). Add B as a code-quality enhancement when
target = x86 AVX-512-CD.

### Estimated cost
~400 LOC. Conflict-detection logic (strategy A) is the bulk. **One day.**

## Argmin / argmax (target #3) — design

### Source pattern
```python
@cpu_kernel
def argmin(A: Buffer[N], result: Buffer<i32>(1)):
    m = A[0]
    idx = 0
    for i in range(1, N):
        if A[i] < m:
            m = A[i]
            idx = i
    result[0] = idx
```

### Why clang scalarizes
Two-output loop-carried dependence (m + idx) coupled by a comparison.

### Vectorized emission
Carry two vector iter_args: `vec_m` (current minimums per lane) and
`vec_idx` (corresponding indices).
```mlir
scf.for %i = 0 to N step L iter_args(%vec_m, %vec_idx) {
  %vec_v   = vector.transfer_read A[%i] : vector<LxF32>
  %vec_iv  = vector.constant_mask + index broadcast : vector<LxIndex>
  %lt      = arith.cmpf olt, %vec_v, %vec_m
  %new_m   = arith.select %lt, %vec_v, %vec_m
  %new_idx = arith.select %lt, %vec_iv, %vec_idx
  scf.yield %new_m, %new_idx
}
// Final: tree-reduce vec_m and vec_idx jointly
%min_val = vector.reduction <minf>, %vec_m
// Find index of min_val in vec_m; pick corresponding lane of vec_idx
```

The cross-lane "find lane of min" requires scalar fallback or `vector.reduction`
with index-tracking semantics (not directly supported upstream).

Two emission strategies:
1. **Pure upstream**: tree-shuffle reduce that updates (val, idx) pair at each
   level. ~50 ops for L=16 but pure vector dialect.
2. **Hardware fallback**: `llvm.intr.x86.avx512.{,m,p}.reduce.{min,max}` for
   the val side, then `vector.extract` of matching idx via `llvm.intr.x86.avx512.cmpps`.

### Estimated cost
~300 LOC. **One day.**

## Prefix scan (target #4) — design

### Source pattern
```python
@cpu_kernel
def scan(A: Buffer[N], B: Buffer[N]):
    acc = 0.0
    for i in range(N):
        B[i] = acc
        acc = acc + A[i]
```

### Why clang scalarizes
Loop-carried scalar `acc` with no early termination — classic prefix-sum.
Clang's vectorizer doesn't recognize the parallel-prefix pattern.

### Vectorized emission (upstream)
Hillis-Steele scan within each chunk; carry the chunk sum to next iteration.
```mlir
scf.for %i = 0 to N step L iter_args(%acc) {
  %vec_v = vector.transfer_read A[%i] : vector<LxF32>
  // In-vector exclusive scan via shuffle ladder
  %s1 = vector.shuffle %vec_v, %zero, [shift-by-1] : vector<LxF32>
  %v1 = arith.addf %vec_v, %s1
  %s2 = vector.shuffle %v1, %zero, [shift-by-2] : vector<LxF32>
  %v2 = arith.addf %v1, %s2
  // ... log2(L) levels ...
  // Add carry from previous iteration
  %vec_acc = vector.broadcast %acc
  %v_total = arith.addf %vec_total, %vec_acc
  vector.transfer_write B[%i], %v_total
  // Carry = acc + sum(vec_v)
  %sum = vector.reduction <add>, %vec_v
  %new_acc = arith.addf %acc, %sum
  scf.yield %new_acc
}
```

### Estimated cost
~350 LOC. **One day.**

## Sequencing recommendation

1. **Compaction first** (this PR): smallest, demonstrates the architecture.
2. **Histogram next**: highest paper-grade impact (canonical clang-miss).
3. **Argmin** and **scan**: parallel-ish work, each one day.

Total: ~3-4 days of focused work to ship all four.
