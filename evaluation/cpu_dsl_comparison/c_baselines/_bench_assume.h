/* _bench_assume.h — let clang treat N as a compile-time constant.
 *
 * The C kernels take ``int N`` as a runtime parameter so a single binary
 * can serve multiple candidate sizes via argv[1].  Under that calling
 * convention the LLVM auto-vectoriser cannot prove identities that
 * depend on N's value (e.g., that ``(i & 0x5555) | ((i >> 1 & 0x5555) << 1)``
 * is the identity permutation when N <= 65536), so kernels keep gathers
 * that LEGO's frontend — which bakes N as a Python compile-time literal —
 * collapses to unit-stride loads.
 *
 * To get a fair apples-to-apples baseline we compile a *_clang_const_<N>
 * variant per candidate with -D__BENCH_N=<N>.  ``BENCH_ASSUME_N(N)`` then
 * tells clang the parameter equals that constant, replicating LEGO's
 * compile-time-N visibility without changing the function signature.
 */
#pragma once

/* Two ways to enable the assume:
 *   1) -DBENCH_USE_DEFAULT_N   — assume runtime N equals the .c file's own
 *                                 DEFAULT_N macro.  Picks the value LEGO
 *                                 hard-codes in the matching candidate when
 *                                 they line up (the common case).
 *   2) -D__BENCH_N=<value>     — assume runtime N equals <value>.  Used for
 *                                 candidates whose N differs from the
 *                                 source's DEFAULT_N (e.g., gemm.c with
 *                                 N=512 from candidate 12).
 *
 * Macro expansion happens at the call site inside each kernel function,
 * which is *after* the .c file's own DEFAULT_N is defined — so referencing
 * DEFAULT_N here is fine even though _bench_assume.h is included before
 * DEFAULT_N's definition.
 */
#ifdef __BENCH_N
#define BENCH_ASSUME_N(n) __builtin_assume((n) == __BENCH_N)
#elif defined(BENCH_USE_DEFAULT_N)
#define BENCH_ASSUME_N(n) __builtin_assume((n) == DEFAULT_N)
#else
#define BENCH_ASSUME_N(n) ((void)0)
#endif
