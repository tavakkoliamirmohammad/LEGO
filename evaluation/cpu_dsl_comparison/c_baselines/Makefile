# c_baselines/Makefile — build C reference kernels in three flavours.
#
#   _O3         : gcc -O3 (CASTLE-aligned, conservative reference)
#   _clang      : clang-20 -O3 -march=native -mavx512f -ffast-math
#                 (max-aggressive clang)
#   _clang_const: same as _clang but no -ffast-math, plus -DBENCH_USE_DEFAULT_N
#                 so the kernel calls __builtin_assume(N == DEFAULT_N).  This
#                 is the apples-to-apples baseline against LEGO's JIT — LEGO
#                 bakes N as a Python compile-time literal, so for a fair
#                 comparison clang needs the same compile-time-N visibility.
#
# Each binary accepts an optional N argument and prints:
#   {"kernel":"<name>","N":<N>,"c_baseline_ms_per_call":<ms>}

CC_GCC   = gcc
# Pin clang to the latest stable on CHPC (clang-20.1.8) so dashboard
# numbers stay reproducible across runs.
CC_CLANG = clang-20

CFLAGS_O3          = -O3 -Wall -Wno-unused-result
CFLAGS_CLANG       = -O3 -march=native -mavx512f -ffast-math \
                     -Wall -Wno-unused-result
CFLAGS_CLANG_CONST = -O3 -march=native -mavx512f -DBENCH_USE_DEFAULT_N \
                     -Wall -Wno-unused-result

# Curated source set — one .c per kept candidate (see ../run_all.py).
ALL_SRCS = saxpy.c gemm.c stencil_3pt.c morton.c mixed_precision.c \
           brick_within_cell.c spmv_indirect.c predicated_fma.c \
           scatter_compute.c multi_reduce.c

O3_BINS          = $(ALL_SRCS:.c=_O3)
CLANG_BINS       = $(ALL_SRCS:.c=_clang)
CLANG_CONST_BINS = $(ALL_SRCS:.c=_clang_const)

all: $(O3_BINS) $(CLANG_BINS) $(CLANG_CONST_BINS)

O3:          $(O3_BINS)
clang_all:   $(CLANG_BINS)
clang_const_all: $(CLANG_CONST_BINS)

%_O3: %.c
	$(CC_GCC) $(CFLAGS_O3) -o $@ $<

%_clang: %.c
	$(CC_CLANG) $(CFLAGS_CLANG) -o $@ $<

%_clang_const: %.c _bench_assume.h
	$(CC_CLANG) $(CFLAGS_CLANG_CONST) -o $@ $<

clean:
	rm -f $(O3_BINS) $(CLANG_BINS) $(CLANG_CONST_BINS)

.PHONY: all O3 clang_all clang_const_all clean
