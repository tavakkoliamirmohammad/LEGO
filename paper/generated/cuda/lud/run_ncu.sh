echo "Running NCU for all LUD configurations..."
make ncu-run-all
echo "Exporting NCU data..."
make ncu-export-roofline
echo "Generating roofline plot..."
python3 plot_roofline.py