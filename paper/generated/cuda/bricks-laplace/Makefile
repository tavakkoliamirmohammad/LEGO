CC = g++
CC_FLAGS = -g -O3 -Wall

bricks-r1: main.cu Makefile
	nvcc -O3 -use_fast_math -lm -DRD_RADIUS=1 -o bricks-f3d main.cu

bricks-r2: main.cu Makefile
	nvcc -O3 -use_fast_math -lm -DRD_RADIUS=2 -o bricks-f3d main.cu

bricks-r3: main.cu Makefile
	nvcc -O3 -use_fast_math -lm -DRD_RADIUS=3 -o bricks-f3d main.cu

bricks-r4: main.cu Makefile
	nvcc -O3 -use_fast_math -lm -DRD_RADIUS=4 -o bricks-f3d main.cu

run-bricks-r1: bricks-r1
	./bricks-f3d

run-bricks-r2: bricks-r2
	./bricks-f3d

run-bricks-r3: bricks-r3
	./bricks-f3d

run-bricks-r4: bricks-r4
	./bricks-f3d

NCU_BASE = ncu --set roofline -f --target-processes all \
           --launch-skip 0 --launch-count 1
REPORT_DIR ?= ../bricks_reports

ncu-run-bricks-r1: bricks-r1
	@mkdir -p $(REPORT_DIR)
	$(NCU_BASE) -k regex:"^laplacian_naive" \-o $(REPORT_DIR)/ncu-run-star-naive-r1.ncu-rep ./bricks-f3d
	$(NCU_BASE) -k regex:"^laplacian_bricks" \-o $(REPORT_DIR)/ncu-run-star-bricks-r1.ncu-rep ./bricks-f3d

ncu-run-bricks-r2: bricks-r2
	@mkdir -p $(REPORT_DIR)
	$(NCU_BASE) -k regex:"^laplacian_naive" \-o $(REPORT_DIR)/ncu-run-star-naive-r2.ncu-rep ./bricks-f3d
	$(NCU_BASE) -k regex:"^laplacian_bricks" \-o $(REPORT_DIR)/ncu-run-star-bricks-r2.ncu-rep ./bricks-f3d

ncu-run-bricks-r3: bricks-r3
	@mkdir -p $(REPORT_DIR)
	$(NCU_BASE) -k regex:"^laplacian_naive" \-o $(REPORT_DIR)/ncu-run-star-naive-r3.ncu-rep ./bricks-f3d
	$(NCU_BASE) -k regex:"^laplacian_bricks" \-o $(REPORT_DIR)/ncu-run-star-bricks-r3.ncu-rep ./bricks-f3d

ncu-run-bricks-r4: bricks-r4
	@mkdir -p $(REPORT_DIR)
	$(NCU_BASE) -k regex:"^laplacian_naive" \-o $(REPORT_DIR)/ncu-run-star-naive-r4.ncu-rep ./bricks-f3d
	$(NCU_BASE) -k regex:"^laplacian_bricks" \-o $(REPORT_DIR)/ncu-run-star-bricks-r4.ncu-rep ./bricks-f3d
	
SECT ?= SpeedOfLight_RooflineChart

ncu-export-roofline:
	@for rep in $(REPORT_DIR)/*.ncu-rep; do \
		out=$${rep%.ncu-rep}.roofline.csv; \
		echo "Exporting $$rep -> $$out (section=$(SECT))"; \
		ncu --import $$rep --section $(SECT) \
		    --csv --print-details=all > $$out; \
	done


clean:
	rm -f bricks-f3d
