#!/bin/bash

# This script creates a conda environment named nmtenv using mamba as the package manager.
# Note: The PATH is modified only for the current session

# Create and set CERT_PATH
export CERT_PATH="/etc/ssl/certs/ca-certificates.crt"
export REQUESTS_CA_BUNDLE="$CERT_PATH"
export SSL_CERT_FILE="$CERT_PATH"

# Check if mamba is installed
if ! command -v mamba &> /dev/null; then
    echo "Mamba is not installed. Please install Mamba in your base conda environment, e.g., 'conda install mamba -n base -c conda-forge'."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Using Python version: $PYTHON_VERSION"

# Remove existing environment if it exists
if mamba env list | grep -q "nmtenv"; then
    echo "Removing existing conda environment..."
    mamba env remove -n nmtenv
fi

# Create new conda environment with Python 3
echo "Creating conda environment..."
mamba create -y -n nmtenv python=3.8.10

# Activate conda environment (using conda activation)
eval "$(conda shell.bash hook)"
conda activate nmtenv

# Verify we're using Python 3
python --version

# Install basic requirements using conda-forge
echo "Installing basic requirements..."
mamba install -y -c conda-forge \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    matplotlib \
    gensim

# Install PyTorch ecosystem with CUDA 11.1
echo "Installing PyTorch ecosystem..."
mamba install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.1 -c pytorch

# Install OpenNMT-py and pytrec_eval using pip (not available in conda)
echo "Installing OpenNMT-py and pytrec_eval..."
pip install OpenNMT-py==3.0.4 pytrec-eval-terrier==0.5.2

# Install PyTorch Geometric and dependencies
echo "Installing PyTorch Geometric ecosystem..."
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.6.0 torch-spline-conv==1.2.1 \
    -f https://data.pyg.org/whl/torch-1.10.1+cu111.html

pip install torch-geometric

echo "Setup complete! Conda environment is activated."
echo "To deactivate the conda environment, run: conda deactivate"
echo "To activate again later, run: conda activate nmtenv" 