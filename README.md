# Neural Team Formation 
Teams are the primary vehicle for coordinating experts with diverse skills needed for a particular collaborative project, and Team Formation (TF) has firsthand effects on creating organizational performance. Social network analysis has been employed for TF by incorporating social ties and interpersonal collaboration features using measures such as degree and closeness centrality. Socially driven methods, however, face two significant challenges, esp., when currency (timeliness) is of the prime concern: ``i) temporality``: the expert's skills and her social attributes constantly changes over time. A successful collaboration of experts in a team years ago does not tailor a successful team now, ``ii) Complexity``: optimum teams are found by computationally expensive search over all the subgraphs of a large-scale collaboration graph (search-based.) 

## Objectives
We propose ``neural machine learning`` approaches to Team Formation. We will train neural models that would learn relationships among experts and their social attributes in vector space. Wherein, we consider all past (un)successful team compositions as training samples to predict future teams and the team's level of success. Therefore, we bring efficiency while enhancing efficacy due to the inherently iterative and online learning procedure in neural architectures. More importantly, we will address temporality by incorporating sequences on the neural architecture via recurrence, which yields a richer representation and more accurate team composition within time. We are the first who consider temporal graph neural networks for team formation and will provide major advances via ``1) time-sensitive`` and ``2) scalable`` incorporation of experts' attributes and skills. 

1. [Setup](#1-setup)
2. [Quickstart](#2-quickstart)
3. [Features](#3-features)
4. [Acknowledgement](#4-acknowledgement)

## 1. Setup
You need to have ``Python >= 3.8`` and the following packages, listed in [``requirements.txt``](requirements.txt), installed:
```
torch==1.6.0
matplotlib==3.4.2
scipy==1.6.3
numpy==1.20.3
tqdm==4.60.0
scikit_learn==0.24.2
pandas==1.3.3
```
To clone the codebase and install the required packages by ``pip``:
```sh
git clone https://github.com/Fani-Lab/neural_team_formation
cd TeamFormation
pip install -r requirements.txt
```
or by [``conda``](https://www.anaconda.com/products/individual):

```sh
git clone https://github.com/Fani-Lab/neural_team_formation
cd TeamFormation
conda env create -f environment.yml
```

For installation of specific version of a python package due to, e.g., ``CUDA`` versions compatibility, one can edit [``requirements.txt``](requirements.txt) or [``environment.yml``](environment.yml) like as follows:

```
# CUDA 10.1
torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
## 2. Quickstart

```sh
cd src
python main.py -data=data/raw/toy.json -model=dnn
```

The above run, load and preprocessed a tiny-size toy example dataset [``toy.json``](data/raw/toy.json) from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z) followed by 5-fold train-evaluation on a training split and final test on test set for a simple feedforward neural model using default hyperparameters.

## 3. Features:
**Data Preprocessing**

Raw dataset, e.g., scholarly papers from [AMiner](https://www.aminer.org/) 's citation network dataset of [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z) or movies from [``imdb``](https://datasets.imdbws.com/), were assumed to be populated in [``data/raw``](data/raw). For the sake of integration test, a tiny-size toy example dataset [``toy.json``](data/raw/toy.json) from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z) and [[``toy.title.basics.tsv``](data/raw/toy.title.basics.tsv), [``toy.title.principals.tsv``](data/raw/toy.title.principals.tsv), [``toy.name.basics.tsv``](data/raw/toy.name.basics.tsv)] from [``imdb``](https://datasets.imdbws.com/) have been already provided.

Raw data will be preprocessed into two main sparse matrices each row of which represents: 

>i) ``member_vecs``: occurrence (boolean) vector representation for members of a team, e.g., authors of a paper or crew members of a movie,
> 
>ii) ``skill_vecs``: occurrence (boolean) vector representation for required skills for a team, e.g., keywords of a paper or genre of a movie.

Also, indexes will be created to map the vector's indexes to members' names and skills' name, i.e., ``i2m``, ``m2i``, ``i2s``, ``s2i``.

The sparse matrixes and the indexes will be persisted in [``data/preprocessed/{name of dataset}``](data/preprocessed/) as pickles ``teams.pkl`` and ``indexes.pkl``. For example, the preprocessed data for our toy example are [``data/preprocessed/toy.json/teams.pkl``](data/preprocessed/toy.json/teams.pkl) and [``data/preprocessed/toy.json/indexes.pkl``](data/preprocessed/toy.json/indexes.pkl).

Please note that the preprocessing step will be executed once. Subsequent runs loads the persisted pickle files. In order to regenerate them, one should simply delete them. 

**Data Train-Test Split**

We randomly take 15% of the dataset for the test set, i.e., the model never sees these instances during training. You can change this parameter [sth](./src/dal/data_utils.py)

**Model Train-Eval-Test**

We use 5-fold validation and train a model on each fold and utilize the validation set of each fold to adjust the learning rate during training.

For each model, different phases of machine learning pipeline has been implemented in ** and will be triggered by cmd arguement inside the [``src/main.py``](src/main.py). For example, for our feedforward baseline, the pipeline has been implemented in [``src/dnn.py``](src/dnn.py). Models' hyperparameters such as learning rate (``lr``) or number of epochs (``e``) can be set in [``src/mdl/param.py``](src/mdl/param.py).

**Run**

Put input options for user to select the model, or combinations of models and also data sourse.

**Results**

Explain what the result mean and how to interpret. Also, put some figures

## 4. Acknowledgement:

