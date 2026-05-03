# c_baselines/Makefile — build C-O3-march=native reference kernels.
#
# Each binary accepts an optional N argument (element count) and prints a
# single JSON line on stdout:
#   {"kernel":"<name>","N":<N>,"c_baseline_ms_per_call":<ms>}
#
# Usage:
#   make all          # build all binaries
#   make clean        # remove all binaries
#   make run          # run all binaries with default N and print results

CC     = gcc
CFLAGS = -O3 -march=native -mavx512f -ffast-math -Wall -Wno-unused-result

SRCS = saxpy.c vecadd.c gemm.c stencil_3pt.c brick_within_cell.c \
       mixed_precision.c col_major.c self_update.c morton.c
BINS = $(SRCS:.c=)

all: $(BINS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

# Run all baselines with default sizes and print JSON results
run: $(BINS)
	@for b in $(BINS); do ./$$b; done

clean:
	rm -f $(BINS)

.PHONY: all run clean
