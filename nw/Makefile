#SIZE=32768
SIZE=16384
#SIZE=8192
#SIZE=4096
#SIZE=2048
#SIZE=1024
REPORT_DIR ?= reports

NCU_BASE = ncu --set roofline -f --target-processes all \
           --launch-skip 0 --launch-count 1

original: needle.cu needle.h needle_kernel.cu timing.h
	nvcc -O3 -DRD_WG_SIZE_0_0=64 -DOPTIM=0 -o needle needle.cu

wpadding: needle.cu needle.h needle_kernel.cu timing.h
	nvcc -O3 -DRD_WG_SIZE_0_0=64 -DOPTIM=1 -o needle needle.cu

antidiag: needle.cu needle.h needle_kernel.cu timing.h
	nvcc -O3 -DRD_WG_SIZE_0_0=64 -DOPTIM=2 -o needle needle.cu

run: needle
	./needle $(SIZE) 10

run-ncu-original: original
	@mkdir -p $(REPORT_DIR)
	$(NCU_BASE) -k regex:"^needle_cuda_shared_1" -o $(REPORT_DIR)/ncu-run-original-k1-s${SIZE}.ncu-rep ./needle $(SIZE) 10
	$(NCU_BASE) -k regex:"^needle_cuda_shared_2" -o $(REPORT_DIR)/ncu-run-original-k2-s${SIZE}.ncu-rep ./needle $(SIZE) 10


run-ncu-antidiag: antidiag
	@mkdir -p $(REPORT_DIR)
	$(NCU_BASE) -k regex:"^needle_cuda_shared_1" -o $(REPORT_DIR)/ncu-run-antidiag-k1-s${SIZE}.ncu-rep ./needle $(SIZE) 10
	$(NCU_BASE) -k regex:"^needle_cuda_shared_2" -o $(REPORT_DIR)/ncu-run-antidiag-k2-s${SIZE}.ncu-rep ./needle $(SIZE) 10

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

default: antidiag run

clean:
	rm -rf needle $(REPORT_DIR)
	
## Use:
##  $ make original
##  $ make wpadding
##  $ make antidiag
## to compile: the original nw implementation, 
##             the padding optimization
##             the antiadigonal-permutation optimization
## Then use:
##   $ make run
## to run 10 instances of nw (since the runtimes may vary in my experience)
##
## On an A100, using SIZE=32768, RD_WG_SIZE_0_0=64 Cosmin obeserves:
##   original: > 72.2
##   wpadding: > 34.4
##   antidiag: > 36.5
