# Use my custom cuda_ubuntu image
FROM kmthang/cuda_ubuntu:cu11.1_ub20.04

# Add label to the image and version to the image
LABEL maintainer="thangk@uwindsor.ca"
LABEL version="1.0"

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
# RUN python3 -m venv /OpeNTF/venv

# Activate the virtual environment and install pip/setuptools in the venv
# RUN /OpeNTF/venv/bin/pip install --upgrade pip setuptools
RUN pip install --upgrade pip setuptools

# Install Python packages within the virtual environment
RUN pip install --root-user-action=ignore \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    matplotlib \
    gensim

# Install OpenNMT-py and pytrec_eval
RUN pip install --root-user-action=ignore OpenNMT-py==3.0.4
RUN pip install --root-user-action=ignore pytrec_eval

# Install torch, torchvision, and torchaudio
RUN pip install --root-user-action=ignore torch==1.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install --root-user-action=ignore torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install --root-user-action=ignore torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Install torch-scatter, torch-sparse, torch-cluster, torch-spline-conv, and torch-geometric in the virtual environment
RUN pip install --root-user-action=ignore torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
RUN pip install --root-user-action=ignore torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
RUN pip install --root-user-action=ignore torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
RUN pip install --root-user-action=ignore torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
RUN pip install --root-user-action=ignore torch-geometric

# Set the working directory
WORKDIR /OpeNTF

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
