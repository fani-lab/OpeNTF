# Team Formation 
Team Formation in Social Network refers to forming a team of individuals, based on their skills or expertise to accomplish the specified task.

1. [Setup](#1-setup)
2. [Quickstart](#2-quickstart)
3. [Features](#3-features)
4. [Acknowledgement](#4-acknowledgement)

## 1. Setup
You need to have ``Python >= 3.8`` and the following packages installed to be able to run the code:
```
torch==1.6.0
matplotlib==3.4.2
scipy==1.6.3
numpy==1.20.3
tqdm==4.60.0
scikit_learn==0.24.2
```
If you have [``conda``](https://www.anaconda.com/products/individual), you can easily install the required packages by:
```sh
git clone https://github.com/VaghehDashti/TeamFormation.git
cd TeamFormation
conda env create -f environment.yml
```

For installation of specific version of a python package due to, e.g., ``CUDA`` versions compatibility, one can edit [``environment.yml``](environment.yml) like as follows:

```
# CUDA 10.1
torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
## 2. Quickstart:
After installing the required packages, you can easily test neural models both with and without negative sampling with the default hyperparameters by running the ``main.py`` inside the ``src`` folder
```sh
python main.py
```
If you want to change the hyperparameters you can open ``src/mdl/param.py`` and change the values to whatever you want.
The default hyperparameters, which we used in our paper, are:
- d: 100 (size of the hidden layer)
- lr: 0.01 (learning rate)
- b: 5 (mini-batch size)
- e: 2 (number of epochs)
- ns: 5 (number of negative samples)

## 3. Features:

## 4. Acknowledgement:

