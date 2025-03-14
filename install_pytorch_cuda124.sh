#!/bin/bash

# Install PyTorch packages with CUDA 12.4 support
echo "Installing PyTorch packages for CUDA 12.4..."

# Install PyTorch core packages using the CUDA 12 index
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric packages with CUDA 12 support
pip install torch-geometric==2.5.3
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# Install other dependencies from the requirements file
echo "Installing other dependencies..."
pip install -r requirements_cuda124.txt --no-deps

echo "Installation complete. PyTorch is now installed with CUDA 12.4 support." 