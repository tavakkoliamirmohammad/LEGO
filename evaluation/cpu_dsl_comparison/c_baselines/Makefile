# c_baselines/Makefile — build C-O3-march=native reference kernels.
#
# Each binary accepts an optional N argument (element count) and prints a
# single JSON line on stdout:
#   {"kernel":"<name>","N":<N>,"c_baseline_ms_per_call":<ms>}
#
# Usage:
#   make all          # build all GCC + Clang binaries
#   make gcc          # build only GCC binaries
#   make clang        # build only Clang binaries
#   make clean        # remove all binaries
#   make run          # run all GCC binaries with default N and print results

CC_GCC   = gcc
CC_CLANG = clang
CFLAGS_GCC   = -O3 -march=native -mavx512f -ffast-math -Wall -Wno-unused-result
CFLAGS_CLANG = -O3 -march=native -mavx512f -ffast-math -Wall -Wno-unused-result

SRCS = saxpy.c vecadd.c gemm.c stencil_3pt.c brick_within_cell.c \
       mixed_precision.c col_major.c self_update.c morton.c

GCC_BINS   = $(SRCS:.c=)
CLANG_BINS = $(SRCS:.c=_clang)

all: $(GCC_BINS) $(CLANG_BINS)

gcc: $(GCC_BINS)

clang: $(CLANG_BINS)

# GCC binaries (original — name without suffix)
%: %.c
	$(CC_GCC) $(CFLAGS_GCC) -o $@ $<

# Clang binaries (name with _clang suffix)
%_clang: %.c
	$(CC_CLANG) $(CFLAGS_CLANG) -o $@ $<

# Run all GCC baselines with default sizes and print JSON results
run: $(GCC_BINS)
	@for b in $(GCC_BINS); do ./$$b; done

# Run all Clang baselines
run_clang: $(CLANG_BINS)
	@for b in $(CLANG_BINS); do ./$$b; done

clean:
	rm -f $(GCC_BINS) $(CLANG_BINS)

.PHONY: all gcc clang run run_clang clean
