
echo "Running NCU for all Bricks Roofline configurations..."
cd ../bricks
echo "----"
make ncu-run-bricks-r1
echo "----"
make ncu-run-bricks-r2

cd ../bricks-laplace
echo "----"
make ncu-run-bricks-r1
echo "----"
make ncu-run-bricks-r2
echo "----"
make ncu-run-bricks-r3
echo "----"
make ncu-run-bricks-r4
echo "Exporting NCU data..."
make ncu-export-roofline
echo "Generating roofline plot..."
cd ../bricks_reports
python3 plot_roofline.py

