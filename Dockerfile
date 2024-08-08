FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN pip3 install --upgrade pip setuptools

# Install Python packages
RUN pip3 install \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    matplotlib \
    gensim

# Install PyTorch 2.3.0 with CUDA 11.8 support (wheels)
RUN pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Download and install torch-scatter, torch-sparse, torch-cluster, torch-spline-conv manually (wheels) for PyTorch 2.3.0
RUN pip3 install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
RUN pip3 install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
RUN pip3 install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
RUN pip3 install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.10.1+cu111.html

# Install OpenNMT-py
RUN pip3 install OpenNMT-py
RUN pip3 install pytrec_eval

# Set the working directory
WORKDIR /OpeNTF

# Copy your application code
COPY . /OpeNTF
