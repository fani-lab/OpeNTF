#!/bin/bash

# This script installs dependencies directly without creating a virtual environment.

# Create and set CERT_PATH
export CERT_PATH="/etc/ssl/certs/ca-certificates.crt"
export REQUESTS_CA_BUNDLE="$CERT_PATH"
export SSL_CERT_FILE="$CERT_PATH"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Using Python version: $PYTHON_VERSION"

# First ensure pip3 is installed
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip3..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
fi

# Install virtualenv if not already installed  (This part might seem counterintuitive,
# but it's kept in case other scripts or processes expect it to be available,
# even if we're not creating a venv here. It won't activate a venv.)
echo "Installing virtualenv..."
python3 -m pip install --user virtualenv --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Add local bin to PATH for virtualenv (same reasoning as above)
export PATH=$HOME/.local/bin:$PATH


# Verify we're using Python 3
python --version

# Upgrade pip and setuptools with trusted hosts
echo "Upgrading pip and setuptools..."
python -m pip install --upgrade pip setuptools wheel --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Install basic requirements first
echo "Installing basic requirements..."
python -m pip install numpy pandas scikit-learn scipy matplotlib gensim --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Install OpenNMT-py and pytrec_eval
echo "Installing OpenNMT-py and pytrec_eval..."
python -m pip install OpenNMT-py==3.0.4 pytrec_eval --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Install PyTorch ecosystem
echo "Installing PyTorch ecosystem..."
python -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Install PyTorch Geometric and dependencies
echo "Installing PyTorch Geometric ecosystem..."
python -m pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.6.0 torch-spline-conv==1.2.1 \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -f https://data.pyg.org/whl/torch-1.10.1+cu111.html

python -m pip install torch-geometric --trusted-host pypi.org --trusted-host files.pythonhosted.org

echo "Setup complete! Dependencies installed globally." 