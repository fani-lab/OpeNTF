#!/bin/bash

# Script to update existing nmtenv2 conda environment with CUDA 12.4 compatible PyTorch

echo "Updating nmtenv2 environment with PyTorch for CUDA 12.4..."

# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nmtenv2

# First, remove existing PyTorch packages to avoid conflicts
conda remove -y --force-remove pytorch torchvision torchaudio pytorch-cuda cuda-toolkit pytorch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv || true
pip uninstall -y torch torchvision torchaudio torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv || true

# Install PyTorch with CUDA 12.x support via conda
echo "Installing PyTorch packages via conda..."
conda install -y pytorch=2.2.0 torchvision=0.17.0 torchaudio=2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify CUDA is detected by PyTorch
echo "Verifying PyTorch CUDA installation..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('PyTorch version:', torch.__version__)"

# Install PyTorch Geometric packages
echo "Installing PyTorch Geometric packages..."
pip install torch-geometric==2.5.3
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

echo "Updating other dependencies from requirements file..."

# Create a temporary requirements file without problematic packages
echo "# Temporary requirements file without conflicting packages" > requirements_temp.txt
grep -v "psutil==" requirements_cuda124.txt >> requirements_temp.txt

# Install packages from the modified requirements file, but skip PyTorch packages
pip install -r requirements_temp.txt --no-deps --ignore-installed torch torchvision torchaudio torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

# Clean up
rm requirements_temp.txt

echo "Installation complete! Your nmtenv2 environment now has PyTorch compatible with CUDA 12.4"
echo ""
echo "Note: Some dependency conflicts were avoided by skipping problematic packages (e.g., psutil)."
echo "These conflicts are typically not critical for the main functionality."
echo ""
echo "To test if CUDA is working correctly, run the following command:"
echo "python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"" 