export PYTHONPATH=../..:$PYTHONPATH
cd "$(dirname "$0")"
cd nw
echo "----"
make ncu-export-roofline
cd ..

cd lud
echo "----"
make ncu-export-roofline
cd ..

cd bricks
echo "----"
make ncu-export-roofline
cd ..

cd bricks-laplace
echo "----"
make ncu-export-roofline
cd ..

echo "Generating roofline data for LUD..."
cd lud
bash run_ncu.sh
cd ..

echo "Generating roofline data for Stencil..."
cd bricks_reports
bash gen_bricks_roofline.bash
cd ../..