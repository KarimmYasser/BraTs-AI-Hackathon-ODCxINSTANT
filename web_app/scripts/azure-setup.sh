#!/bin/bash
# Azure VM Quick Setup Script
# Run this after SSHing into your Azure VM

set -e

echo "=== Updating System ==="
sudo apt update && sudo apt upgrade -y

echo "=== Installing NVIDIA Drivers ==="
sudo apt install -y nvidia-driver-535

echo "=== Installing Python ==="
sudo apt install -y python3-pip python3-venv git curl

echo "=== Setting up Application ==="
cd ~

# Create app directory
mkdir -p brain-tumor-app
cd brain-tumor-app

echo "=== Creating Virtual Environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installing PyTorch with CUDA ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing Requirements ==="
pip install fastapi uvicorn python-multipart nibabel numpy pillow pydantic

echo "=== Setup Complete ==="
echo ""
echo "IMPORTANT: You need to reboot for NVIDIA drivers to load!"
echo "After reboot:"
echo "  1. SSH back in"
echo "  2. cd brain-tumor-app && source venv/bin/activate"
echo "  3. Copy your app files or clone from git"
echo "  4. Run: python -m uvicorn app:app --host 0.0.0.0 --port 8000"
echo ""
echo "Run 'sudo reboot' now to complete setup."
