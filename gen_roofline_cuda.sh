
echo "Generating roofline data for LUD..."
cd lud
bash run_ncu.sh
cd ..

echo "Generating roofline data for Stencil..."
cd bricks_reports
bash gen_bricks_roofline.bash
cd ..