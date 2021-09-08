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
## 2. Quickstart

```sh
cd src
python main.py -data=data/raw/toy.json -model=dnn
```

The above run, load and preprocessed a tiny-size toy example dataset [``toy.json``](data/raw/toy.json) from [``dblp``]() followed by 5-fold train-evaluation on a training split and final test on test set for a simple feedforward neural model using default hyperparameters.

## 3. Features:
**Data Preprocessing**

Raw dataset, e.g., scholarly papers from [``dblp``]() or movies from [``imdb``], were assumed to be populated in [``data/raw``](data/raw). For the sake of integration test, a tiny-size toy example dataset [``toy.json``](data/raw/toy.json) from [``dblp``]() has been already available.

Raw data will be preprocessed into two main sparse matrices each row of which represents: 

>i) ``member_vecs``: occurrence (boolean) vector representation for members of a team, e.g., authors of a paper or crew members of a movie, 
>ii) ``skill_vecs``: occurrence (boolean) vector representation for required skills for a team, e.g., keywords of a paper or genre of a movie.

Also, indexes will be created to map the vector's indexes to members' names and skills' name, i.e., ``i2m``, ``m2i``, ``i2s``, ``s2i``.

The sparse matrixes and the indexes will be persisted in [``data/preprocessed/{name of dataset}``](data/preprocessed/) as pickles ``teams.pkl`` and ``indexes.pkl``. For example, the preprocessed data for our toy example are [``teams.pkl``](data/preprocessed/toy/teams.pkl) and [``indexes.pkl``](data/preprocessed/toy/indexes.pkl).

Please note that the preprocessing step will be executed once. Subsequent runs loads the persisted pickle files. In order to regenerate them, one should simply delete them. 

**Data Train-Test Split**

Likewise, please explain beriefly what happens in this step.

**Model Train-Eval-Test**

Likewise, please explain beriefly what happens in this step.

For each model, different phases of machine learning pipeline has been implemented in ** and will be triggered by cmd arguement inside the [``src/main.py``](src/main.py). For example, for our feedforward baseline, the pipeline has been implemented in [``src/dnn.py``](src/dnn.py). Models' hyperparameters such as learning rate (``lr``) or number of epochs (``e``) can be set in [``src/mdl/param.py``](src/mdl/param.py).

**Run**

Put input options for user to select the model, or combinations of models and also data sourse.

**Results**

Explain what the result mean and how to interpret. Also, put some figures

## 4. Acknowledgement:

