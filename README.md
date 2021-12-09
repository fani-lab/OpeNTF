# Neural Team Formation 
Teams are the primary vehicle for coordinating experts with diverse skills needed for a particular collaborative project, and Team Formation (TF) has firsthand effects on creating organizational performance. Social network analysis has been employed for TF by incorporating social ties and interpersonal collaboration features using measures such as degree and closeness centrality. Socially driven methods, however, face two significant challenges, esp., when currency (timeliness) is of the prime concern: ``i) temporality``: the expert's skills and her social attributes constantly changes over time. A successful collaboration of experts in a team years ago does not tailor a successful team now, ``ii) Complexity``: optimum teams are found by computationally expensive search over all the subgraphs of a large-scale collaboration graph (search-based.) 

## Objectives
We propose ``neural machine learning`` approaches to Team Formation. We will train neural models that would learn relationships among experts and their social attributes in vector space. Wherein, we consider all past (un)successful team compositions as training samples to predict future teams and the team's level of success. Therefore, we bring efficiency while enhancing efficacy due to the inherently iterative and online learning procedure in neural architectures. More importantly, we will address temporality by incorporating sequences on the neural architecture via recurrence, which yields a richer representation and more accurate team composition within time. We are the first who consider temporal graph neural networks for team formation and will provide major advances via ``1) time-sensitive`` and ``2) scalable`` incorporation of experts' attributes and skills. 

1. [Setup](#1-setup)
2. [Quickstart](#2-quickstart)
3. [Features](#3-features)
4. [Acknowledgement](#4-acknowledgement)

## 1. Setup
You need to have ``Python >= 3.8`` and install the following main packages, among others listed in [``requirements.txt``](requirements.txt):
```
torch==1.6.0
pytre_eval==0.5
gensim==3.8.3
```
By ``pip``, clone the codebase and install the required packages:
```sh
git clone https://github.com/Fani-Lab/neural_team_formation
cd neural_team_formation
pip install -r requirements.txt
```
By [``conda``](https://www.anaconda.com/products/individual):

```sh
git clone https://github.com/Fani-Lab/neural_team_formation
cd neural_team_formation
conda env create -f environment.yml
conda activate ntf
```

For installation of specific version of a python package due to, e.g., ``CUDA`` versions compatibility, one can edit [``requirements.txt``](requirements.txt) or [``environment.yml``](environment.yml) like as follows:

```
# CUDA 10.1
torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
## 2. Quickstart

```sh
cd src
python main.py -data=data/raw/dblp/toy.dblp.v12.json -domain=dblp -model=fnn
```

The above run, loads and preprocesses a tiny-size toy example dataset [``toy.dblp.v12.json``](data/raw/dblp/toy.dblp.v12.json) from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z) followed by 5-fold train-evaluation on a training split and final test on test set for a simple feedforward neural model using default hyperparameters from [``./src/param.py``](./src/param.py).

## 3. Features:
**Data Loading and Parallel Preprocessing**

Raw dataset, e.g., scholarly papers from [AMiner](https://www.aminer.org/) 's citation network dataset of [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z) or movies from [``imdb``](https://datasets.imdbws.com/), were assumed to be populated in [``data/raw``](data/raw). For the sake of integration test, a tiny-size toy example dataset [``toy.dblp.v12.json``](data/raw/dblp/toy.dblp.v12.json) from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z) and [[``toy.title.basics.tsv``](data/raw/imdb/toy.title.basics.tsv), [``toy.title.principals.tsv``](data/raw/imdb/toy.title.principals.tsv), [``toy.name.basics.tsv``](data/raw/imdb/toy.name.basics.tsv)] from [``imdb``](https://datasets.imdbws.com/) have been already provided.

Raw data will be preprocessed into two main ``sparse`` matrices each row of which represents: 

>i) ``member_vecs``: occurrence (boolean) vector representation for members of a team, e.g., authors of a paper or crew members of a movie,
> 
>ii) ``skill_vecs``: occurrence (boolean) vector representation for required skills for a team, e.g., keywords of a paper or genre of a movie.

Also, indexes will be created to map the vector's indexes to members' names and skills' name, i.e., ``i2c``, ``c2i``, ``i2s``, ``s2i``.

The sparse matrices and the indices will be persisted in [``data/preprocessed/{dblp,imdb,uspt}/{name of dataset}``](data/preprocessed/) as pickles ``teamsvecs.pkl`` and ``indexes.pkl``. For example, the preprocessed data for our dblp toy example are [``data/preprocessed/dblp/toy.dblp.v12.json/teams.pkl``](data/preprocessed/dblp/toy.dblp.v12.json/teams.pkl) and [``data/preprocessed/dblp/toy.dblp.v12.json/indexes.pkl``](data/preprocessed/dblp/toy.dblp.v12.json/indexes.pkl).

> Our pipeline benefits from parallel generation of sparse matrices for teams that significantly reduces the preprocessing time. For instance, 
> 1) it reduced 11 days to ** hours for the dblp.v12 dataset of ** papers.
> 2) it reduced ** days to ** hours for the imdb dataset of ** moview.
> 3) it reduced ** days to ** hours for the uspt dataset of ** patents.

Please note that the preprocessing step will be executed once. Subsequent runs loads the persisted pickle files. In order to regenerate them, one should simply delete them. 

**Data Train-Test Split**

We randomly take 15% of the dataset for the test set, i.e., the model never sees these instances during training. You can change ``train_test_split`` parameter in [``./src/param.py``](./src/param.py).

**Model Train-Validation Split**

We use n-fold cross-validation, that is, we train a model n times on (n-1) folds and utilize the ramaining fold as the validation set to adjust the learning rate during training. The number of folds is set by ``nfolds`` in [``./src/param.py``](./src/param.py).

**Model Architecture**

Each model has been defined in [``./src/mdl/``](./src/mdl/) and realized in a submain pipeline that can be executed for ``train``, ``test``, and ``eval`` steps. 
For example, for our feedforward baseline [``fnn``](./src/mdl/fnn.py), the pipeline has been implemented in [``./src/mdl/fnn.py``](src/mdl/fnn.py) which is realized in [``./src/fnn_main.py``](src/fnn_main.py).
Model's hyperparameters such as learning rate (``lr``) or number of epochs (``e``) can be set in [``./src/param.py``](src/param.py).

**Negative Sampling Strategies**

As known, employing unsuccessful teams convey complementary negative signals to the model to alleviate the long-tail problem. 
Most real-world training datasets in the team formation domain, however, do not have explicit unsuccessful teams (e.g., collections of rejected papers.) 
In the absence of unsuccessful training instances, we proposed negative sampling strategies based on the closed-world assumption where no currently known successful group of experts for the required skills is assumed to be unsuccessful. 
We study the effect of three different negative sampling distributions: two static negative sampling distributions , and an adaptive noise distribution:

1) Uniform distribution (``uniform``), where subsets of experts is randomly chosen with the same probability as unsuccessful teams from the uniform distribution over all subsets of experts.

2) Unigram distribution (``unigram``), where subsets of experts is chosen regarding their frequency in all previous successful teams. 
   Intuitively, teams of experts that have been more successful but for other skill subsets will be given a higher probability and chosen more frequently as a negative sample to dampen the effect of popularity bias.

3) Smoothed unigram distribution in each training minibatch (``unigram_b``), where we employed the add-1 or Laplace smoothing when computing the unigram distribution of the experts but in each training minibatch. 
   Minibatch stochastic gradient descent is the _de facto_ method for neural models where the data is splitted into batches of data, each of which sent to the model for partial calculation to speed up training while maintaining high accuracy. 

To include a negative sampling strategy, there are two paramters for a model to set in [``./src/param.py``](src/param.py):
- ``ns``: the negative sampling stratey which can be ``uniform``, ``unigram``, ``unigram_b`` or ``None``(no negative stampling).
- ``nns``: number of negative samples

**Run**

The pipeline accept three required input values:
1) ``-data``: the path to the raw datafile, e.g., ``-data=./../data/raw/dblp/dblp.v12.json``, or the main file of a dataset, e.g., ``-data=./../data/raw/imdb/title.basics.tsv``
2) ``-domain``: the domain of the raw datafile that could be ``dblp`` or ``imdb``, e.g., ``-domain=dblp``.
3) ``-model``: the baseline model, e.g., ``-model=fnn`` 

**Results**

We used [``pytrec_eval``](https://github.com/cvangysel/pytrec_eval) to evaluate the performance of models on test set as well as on their own train sets (should overfit) and validation sets.
We report the predictions, evaluation metrics on each test instance, and average on all test instances in ``./output/{dataset name}/{model name}/{model's running setting}/``. 
For example:
1) [``f0.test.pred``](./output/toy.dblp.v12.json/fnn/t30.s11.e12.l[100].lr0.1.b4096.e2/f0.test.pred) is the predictions of the trained model on [1,2,3,4] folds on each instance of the test set
2) [``f0.test.pred.eval.csv``](./output/toy.dblp.v12.json/fnn/t30.s11.e12.l[100].lr0.1.b4096.e2/f0.test.pred.eval.csv) is the evaluation metrics of the trained model on [1,2,3,4] folds on each instance of the test set
3) [``f0.test.pred.eval.mean.csv``](./output/toy.dblp.v12.json/fnn/t30.s11.e12.l[100].lr0.1.b4096.e2/f0.test.pred.eval.mean.csv) is the average of evaluation metrics of the trained model on [1,2,3,4] folds on all instance of the test set


TODO: Put result figures and explain them.

## 4. Acknowledgement:

