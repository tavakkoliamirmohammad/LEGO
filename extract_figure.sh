mkdir -p figures

cd figures

mkdir -p fig_11 fig_12 fig_13 tab_v
cp ../plots/cuda_nw.pdf fig_12/cuda_nw.pdf
cp ../plots/lud.pdf fig_12/lud.pdf
cp ../plots/bricks.pdf fig_12/stencil.pdf

cp ../plots/charts/* fig_11/

cp ../benchmarks/cuda/lud/lud_roofline.png fig_13/lud_roofline.png
cp ../benchmarks/cuda/bricks_reports/stencil_roofline.png fig_13/stencil_roofline.png

cp ../plots/mlir_transpose.txt tab_v/mlir_transpose.txt