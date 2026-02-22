echo "Creting virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "Installing required packages..."
pip install -r requirements.txt
echo "Installing Triton and dependencies..."
pip3 install triton==3.2.0 torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
echo "Setup complete."