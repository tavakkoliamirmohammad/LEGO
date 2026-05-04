# c_baselines/Makefile — build C reference kernels in two flavors.
#
# Two baseline variants:
#   _O3      : gcc -O3 only (CASTLE-aligned: matches PolyBench reference build)
#   _agg     : gcc -O3 -march=native -mavx512f -ffast-math (aggressive: state-of-the-art)
#
# Each binary accepts an optional N argument and prints:
#   {"kernel":"<name>","N":<N>,"c_baseline_ms_per_call":<ms>}
#
# Usage:
#   make all          # build both O3 and aggressive GCC variants + Clang
#   make O3           # build only O3-only variants
#   make aggressive   # build only O3+march=native variants
#   make clang        # build Clang aggressive variants
#   make clean        # remove all binaries

CC_GCC   = gcc
CC_CLANG = clang

# CASTLE-aligned baseline: -O3 only (no march, no fast-math)
# This matches how PolyBench reference baselines are built.
CFLAGS_O3        = -O3 -Wall -Wno-unused-result

# Aggressive baseline: -O3 + AVX-512 + ffast-math (best possible compiler output)
CFLAGS_AGGRESSIVE = -O3 -march=native -mavx512f -ffast-math -Wall -Wno-unused-result

# Clang aggressive (for three-way comparison)
CFLAGS_CLANG = -O3 -march=native -mavx512f -ffast-math -Wall -Wno-unused-result

# Original 8 candidates that existed before
SRCS_ORIG = saxpy.c vecadd.c gemm.c stencil_3pt.c brick_within_cell.c \
            mixed_precision.c col_major.c self_update.c morton.c

# New candidates for the remaining 34:
SRCS_NEW = fma_1M.c fma_64k.c stride2_16k.c stride4_16k.c \
           stencil_3d7pt.c stencil_3d13pt.c stencil_2d5pt.c stencil_2d5pt_30x30.c \
           morton_fma_64k.c dgemm.c \
           spmv_indirect.c predicated_fma.c stride7.c

ALL_SRCS = $(SRCS_ORIG) $(SRCS_NEW)

# Derived binary names
O3_BINS        = $(ALL_SRCS:.c=_O3)
AGG_BINS       = $(ALL_SRCS:.c=_agg)
CLANG_BINS     = $(ALL_SRCS:.c=_clang)

# Legacy GCC bins (original aggressive flags, no suffix — keep for backward compat)
GCC_BINS   = $(SRCS_ORIG:.c=)
CLANG_BINS_ORIG = $(SRCS_ORIG:.c=_clang)

all: $(GCC_BINS) $(CLANG_BINS_ORIG) $(O3_BINS) $(AGG_BINS) $(CLANG_BINS)

O3: $(O3_BINS)

aggressive: $(AGG_BINS)

clang_all: $(CLANG_BINS)

# Legacy GCC binaries (aggressive flags, original names — backward compat)
%: %.c
	$(CC_GCC) $(CFLAGS_AGGRESSIVE) -o $@ $<

# Clang binaries (legacy, name with _clang suffix)
%_clang: %.c
	$(CC_CLANG) $(CFLAGS_CLANG) -o $@ $<

# O3-only variants (CASTLE-aligned)
%_O3: %.c
	$(CC_GCC) $(CFLAGS_O3) -o $@ $<

# Aggressive variants
%_agg: %.c
	$(CC_GCC) $(CFLAGS_AGGRESSIVE) -o $@ $<

# Run all O3 baselines with default sizes
run_O3: $(O3_BINS)
	@for b in $(O3_BINS); do ./$$b; done

# Run all aggressive baselines
run_agg: $(AGG_BINS)
	@for b in $(AGG_BINS); do ./$$b; done

# Run original GCC baselines
run: $(GCC_BINS)
	@for b in $(GCC_BINS); do ./$$b; done

clean:
	rm -f $(GCC_BINS) $(CLANG_BINS_ORIG) $(O3_BINS) $(AGG_BINS) $(CLANG_BINS)

.PHONY: all O3 aggressive clang_all run_O3 run_agg run clean
