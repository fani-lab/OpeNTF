# Team Formation 
Team Formation in Social Network refers to forming a team of individuals, based on their skills or expertise to accomplish the specified task.

1. [Setup](#1-setup)
2. [Quickstart](#2-quickstart)
3. [Features](#3-features)
4. [Acknowledgement](#4-acknowledgement)

## 1. Setup
We require ``Python >=3.7`` and packages listed in [``requirements.txt``](requirements.txt). We need to clone the codebase and install the required packages by:
```sh
git clone https://github.com/VaghehDashti/TeamFormation.git
cd TeamFormation
pip install -r requirements.txt
```
But a better way would be to create a new [``conda``](https://www.anaconda.com/products/individual) environment (by default ``TeamFormation``) and install the required packages by:
```sh
git clone https://github.com/VaghehDashti/TeamFormation.git
cd TeamFormation
conda env create -f environment.yml
```

For installation of specific version of a python package due to, e.g., ``CUDA`` versions compatibility, one can edit [``environment.yml``](environment.yml) or [``requirements.txt``](requirements.txt) like as follows:

```
# CUDA 10.1
torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
## 2. Quickstart:

## 3. Features:

## 4. Acknowledgement:

