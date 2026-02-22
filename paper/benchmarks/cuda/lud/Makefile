CC = g++
CC_FLAGS = -g -O3 -Wall

RUNS   ?= 100
WARMUP ?= 25

REPORT_DIR ?= reports

.PHONY: clean \
        run-orig-32 run-orig-16 run-lego-16 run-lego-8 run-lego-4 run-lego-2 \
        ncu-run-orig-32 ncu-run-orig-16 ncu-run-lego-16 ncu-run-lego-8 ncu-run-lego-4 ncu-run-lego-2 \
        ncu-run-all

# ==========================
# Build targets
# ==========================

lud-orig-16: lud.cu lud_kernel.cu common.h Makefile
	nvcc -O3 -use_fast_math -lm -DRD_WG_SIZE_0=16 -o lud-orig lud.cu

lud-orig-32: lud.cu lud_kernel.cu common.h Makefile
	nvcc -O3 -use_fast_math -lm -DRD_WG_SIZE_0=32 -o lud-orig lud.cu

lud-lego-2: lud.cu lud_kernel.cu common.h Makefile
	nvcc -O3 -use_fast_math -lm -DLEGO_R2=1 -DRD_WG_SIZE_0=64 -o lud-lego lud.cu

lud-lego-4: lud.cu lud_kernel.cu common.h Makefile
	nvcc -O3 -use_fast_math -lm -DLEGO_R4=1 -DRD_WG_SIZE_0=64 -o lud-lego lud.cu

lud-lego-8: lud.cu lud_kernel.cu common.h Makefile
	nvcc -O3 -use_fast_math -lm -DLEGO_R8=1 -DRD_WG_SIZE_0=64 -o lud-lego lud.cu

lud-lego-16: lud.cu lud_kernel.cu common.h Makefile
	nvcc -O3 -use_fast_math -lm -DLEGO_R16=1 -DRD_WG_SIZE_0=64 -o lud-lego lud.cu

clean:
	rm -f lud-lego lud-orig
	rm -rf $(REPORT_DIR)

# ==========================
# Plain run targets (no NCU)
# ==========================

run-orig-32: lud-orig-32
	./lud-orig -s 4096  -r $(RUNS) -w $(WARMUP)
	./lud-orig -s 8192  -r $(RUNS) -w $(WARMUP)
	./lud-orig -s 16384 -r $(RUNS) -w $(WARMUP)
	./lud-orig -s 32768 -r $(RUNS) -w $(WARMUP)

run-orig-16: lud-orig-16
	./lud-orig -s 4096  -r $(RUNS) -w $(WARMUP)
	./lud-orig -s 8192  -r $(RUNS) -w $(WARMUP)
	./lud-orig -s 16384 -r $(RUNS) -w $(WARMUP)
	./lud-orig -s 32768 -r $(RUNS) -w $(WARMUP)

run-lego-16: lud-lego-16
	./lud-lego -s 4096  -r $(RUNS) -w $(WARMUP)
	./lud-lego -s 8192  -r $(RUNS) -w $(WARMUP)
	./lud-lego -s 16384 -r $(RUNS) -w $(WARMUP)
	./lud-lego -s 32768 -r $(RUNS) -w $(WARMUP)

run-lego-8: lud-lego-8
	./lud-lego -s 4096  -r $(RUNS) -w $(WARMUP)
	./lud-lego -s 8192  -r $(RUNS) -w $(WARMUP)
	./lud-lego -s 16384 -r $(RUNS) -w $(WARMUP)
	./lud-lego -s 32768 -r $(RUNS) -w $(WARMUP)

run-lego-4: lud-lego-4
	./lud-lego -s 4096  -r $(RUNS) -w $(WARMUP)
	./lud-lego -s 8192  -r $(RUNS) -w $(WARMUP)
	./lud-lego -s 16384 -r $(RUNS) -w $(WARMUP)
	./lud-lego -s 32768 -r $(RUNS) -w $(WARMUP)

run-lego-2: lud-lego-2
	./lud-lego -s 4096  -r $(RUNS) -w $(WARMUP)
	./lud-lego -s 8192  -r $(RUNS) -w $(WARMUP)
	./lud-lego -s 16384 -r $(RUNS) -w $(WARMUP)
	./lud-lego -s 32768 -r $(RUNS) -w $(WARMUP)

# ==========================
# NCU Roofline-only run targets (explicit, no SIZE var)
#   - Profiles only lud_internal* kernels
#   - Profiles only the first matching launch (--launch-count 1)
#   - Writes: reports/ncu-<target>-s<SIZE>.ncu-rep
# ==========================

NCU_BASE = ncu --set roofline -f --target-processes all \
           -k regex:"^lud_internal.*" \
           --launch-skip 0 --launch-count 1

ncu-run-orig-32: lud-orig-32
	@mkdir -p $(REPORT_DIR)
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-orig-32-s4096.ncu-rep   ./lud-orig -s 4096  -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-orig-32-s8192.ncu-rep   ./lud-orig -s 8192  -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-orig-32-s16384.ncu-rep  ./lud-orig -s 16384 -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-orig-32-s32768.ncu-rep  ./lud-orig -s 32768 -r 1 -w 0

ncu-run-orig-16: lud-orig-16
	@mkdir -p $(REPORT_DIR)
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-orig-16-s4096.ncu-rep   ./lud-orig -s 4096  -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-orig-16-s8192.ncu-rep   ./lud-orig -s 8192  -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-orig-16-s16384.ncu-rep  ./lud-orig -s 16384 -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-orig-16-s32768.ncu-rep  ./lud-orig -s 32768 -r 1 -w 0

ncu-run-lego-16: lud-lego-16
	@mkdir -p $(REPORT_DIR)
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-16-s4096.ncu-rep   ./lud-lego -s 4096  -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-16-s8192.ncu-rep   ./lud-lego -s 8192  -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-16-s16384.ncu-rep  ./lud-lego -s 16384 -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-16-s32768.ncu-rep  ./lud-lego -s 32768 -r 1 -w 0

ncu-run-lego-8: lud-lego-8
	@mkdir -p $(REPORT_DIR)
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-8-s4096.ncu-rep    ./lud-lego -s 4096  -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-8-s8192.ncu-rep    ./lud-lego -s 8192  -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-8-s16384.ncu-rep   ./lud-lego -s 16384 -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-8-s32768.ncu-rep   ./lud-lego -s 32768 -r 1 -w 0

ncu-run-lego-4: lud-lego-4
	@mkdir -p $(REPORT_DIR)
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-4-s4096.ncu-rep    ./lud-lego -s 4096  -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-4-s8192.ncu-rep    ./lud-lego -s 8192  -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-4-s16384.ncu-rep   ./lud-lego -s 16384 -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-4-s32768.ncu-rep   ./lud-lego -s 32768 -r 1 -w 0

ncu-run-lego-2: lud-lego-2
	@mkdir -p $(REPORT_DIR)
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-2-s4096.ncu-rep    ./lud-lego -s 4096  -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-2-s8192.ncu-rep    ./lud-lego -s 8192  -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-2-s16384.ncu-rep   ./lud-lego -s 16384 -r 1 -w 0
	$(NCU_BASE) -o $(REPORT_DIR)/ncu-run-lego-2-s32768.ncu-rep   ./lud-lego -s 32768 -r 1 -w 0

# Convenience aggregate
ncu-run-all: ncu-run-orig-32 ncu-run-orig-16 ncu-run-lego-16 ncu-run-lego-8 ncu-run-lego-4 ncu-run-lego-2
	@echo "NCU runs completed. Reports in $(REPORT_DIR)/"

# --- Export all roofline tables from .ncu-rep to .csv
# SECT ?= SpeedOfLight_RooflineChart
SECT ?= SpeedOfLight_RooflineChart

ncu-export-roofline:
	@for rep in $(REPORT_DIR)/*.ncu-rep; do \
		out=$${rep%.ncu-rep}.roofline.csv; \
		echo "Exporting $$rep -> $$out (section=$(SECT))"; \
		ncu --import $$rep --section $(SECT) \
		    --csv --print-details=all > $$out; \
	done
