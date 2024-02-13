# ``OpeNTF``: An Open-Source Neural Team Formation Benchmark Library 
Team formation involves selecting a team of skillful experts who will, more likely than not, accomplish a task. Researchers have proposed a rich body of computational methods to automate the traditionally tedious and error-prone manual process. We previously released OpeNTF, an open-source framework hosting canonical neural models as the cutting-edge class of approaches, along with large-scale training datasets from varying domains. In this paper, we contribute OpeNTF, which extends the initial release in two prime directions. (1) The first of its kind in neural team formation, we integrated `debiasing reranking algorithms` to mitigate the `popularity` and `gender` disparities in the neural models’ team recommendations based on two alternative notions of fairness: equality of opportunity and demographic parity. (2) We further contribute a `curriculum learning` training strategy for neural models’ training to capture the difficulty level of experts to learn easy popular experts first, then moving toward hard nonpopular experts, as opposed to randomly shuffled training datasets. OpeNTF is a forward-looking effort to automate team formation via fairness-aware and time-sensitive methods. AI-ML-based solutions are increasingly impacting how resources are allocated to various groups in society, and ensuring fairness and time are systematically considered is key.

<table border=0>
<tr>
<td >

  
- [1. Setup](#1-setup)
- [2. Quickstart](#2-quickstart)
- [3. Features](#3-features)
  * [`Dynamic Loss-based Curriculum Learning`](#31-Dynamic-Loss-based-Curriculum-Learning)
  * [`Datasets and Parallel Preprocessing`](#32-datasets-and-parallel-preprocessing)
  * [`Model Architecture`](#35-model-architecture)
  * [`Run`](#37-run)
- [4. Results](#4-results)
- [5. Acknowledgement](#5-acknowledgement)



## 1. [Setup](https://colab.research.google.com/github/fani-lab/OpeNTF/blob/main/quickstart.ipynb)
You need to have ``Python >= 3.8`` and install the following main packages, among others listed in [``requirements.txt``](requirements.txt):
```
torch>=1.9.0
pytrec-eval-terrier==0.5.2
gensim==3.8.3
```
By ``pip``, clone the codebase and install the required packages:
```sh
git clone --recursive https://github.com/Fani-Lab/opentf
cd opentf
pip install -r requirements.txt
```
By [``conda``](https://www.anaconda.com/products/individual):

```sh
git clone --recursive https://github.com/Fani-Lab/opentf
cd opentf
conda env create -f environment.yml
conda activate opentf
```

For installation of specific version of a python package due to, e.g., ``CUDA`` versions compatibility, one can edit [``requirements.txt``](requirements.txt) or [``environment.yml``](environment.yml) like as follows:

```
# CUDA 10.1
torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
## 2. Quickstart [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fani-lab/Adila/blob/main/quickstart.ipynb)

```sh
cd src
python -u main.py -data ../data/raw/dblp/toy.dblp.v12.json -domain dblp -model fnn bnn -fairness det_greedy -attribute popularity
```

The above run, loads and preprocesses a tiny-size toy example dataset [``toy.dblp.v12.json``](data/raw/dblp/toy.dblp.v12.json) from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z) followed by _n_-fold train-evaluation on a training split and final test on the test set for ``feedforward`` and ``Bayesian`` neural models using default hyperparameters from [``./src/param.py``](./src/param.py). Then, the predictions will be passed through the `det_greedy` reranking fairness algorithm to mitigate popularity bias in teams with default `k_max`, `np_ratio` fromn [``./src/param.py``](./src/param.py).

```
python -u main.py -data ../data/raw/dblp/toy.dblp.v12.json -domain dblp -model tbnn tbnn_dt2v_emb
```

This script loads and preprocesses the same dataset [``toy.dblp.v12.json``](data/raw/dblp/toy.dblp.v12.json) from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z), takes the teams from the the last year as the test set and trains the ``Bayesian`` neural model following our proposed streaming training strategy as explained in ``3.2.2. Temporal Neural Team Formation`` with two different input representations _i_) sparse vector represntation and _ii_) temporal skill vector represntation using default hyperparameters from [``./src/param.py``](./src/param.py).

## 3. Features

#### **3.1. Dynamic Loss-based Curriculum Learning** 
inspired by the curriculum learning in education that adopts the human learning pace through gradual progress from _easy_ to _hard_ topics, we propose to leverage curriculum-based learning strategies that provide an order between experts from the easy popular experts to the hard nonpopular ones to overcome the neural models' performance drain caused by learning strategies that are disregardful of experts' different difficulty levels. In a curriculum, training samples are ordered based on their level of difficulty (complexity) either statically _prior_ to learning, or dynamically _during_ learning procedure. 
For instance, for team recommendation, a long-standing well-experienced expert who has been in many successful teams would be easier to learn from, compared to an early career expert. A dynamic curriculum identifies easy samples from difficult ones by considering the effect of the samples on the learning progress of a model during learning epochs. For instance, if the model's objective function produces large (small) loss values in several epochs for a sample expert, the model would consider the expert as difficult (easy). 

To include a Curriculum Learning strategy, there is a parameter for a model to set in [``./src/param.py``](src/param.py):
- ``CL``: the curriculum learning strategy which can be ``SL`` (non parametric curriculum), ``DP`` (parametric curriculum), ``normal``(no curriculum learning).

To have a better understanding of our proposed loss-based curriculum learning strategy, we compare the flow of calculating training loss using normal cross entropy and our proposed loss function in (parametric curriculum).
please note that &Phi; refers to the difficulty level of each expert (class).

<p align="center"><img src='./misc/CL flow 1.png.png' width="300" ></p>





#### **3.3. Datasets and Parallel Preprocessing**

Raw dataset, e.g., scholarly papers from AMiner's citation network dataset of [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z), movies from [``imdb``](https://datasets.imdbws.com/), or US patents from [``uspt``](https://patentsview.org/download/data-download-tables) were assumed to be populated in [``data/raw``](data/raw). For the sake of integration test, tiny-size toy example datasets [``toy.dblp.v12.json``](data/raw/dblp/toy.dblp.v12.json) from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z), [[``toy.title.basics.tsv``](data/raw/imdb/toy.title.basics.tsv), [``toy.title.principals.tsv``](data/raw/imdb/toy.title.principals.tsv), [``toy.name.basics.tsv``](data/raw/imdb/toy.name.basics.tsv)] from [``imdb``](https://datasets.imdbws.com/) and [``toy.patent.tsv``](data/preprocessed/uspt/toy.patent.tsv) have been already provided.

<p align="center"><img src='./src/cmn/dataset_hierarchy.png' width="300" ></p>

Raw data will be preprocessed into two main ``sparse`` matrices each row of which represents: 

>i) ``vecs['member']``: occurrence (boolean) vector representation for members of a team, e.g., authors of a paper or crew members of a movie,
> 
>ii) ``vecs['skill']``: occurrence (boolean) vector representation for required skills for a team, e.g., keywords of a paper or genre of a movie.

Also, indexes will be created to map the vector's indexes to members' names and skills' names, i.e., ``i2c``, ``c2i``, ``i2s``, ``s2i``.

The sparse matrices and the indices will be persisted in [``data/preprocessed/{dblp,imdb,uspt}/{name of dataset}``](data/preprocessed/) as pickles ``teamsvecs.pkl`` and ``indexes.pkl``. For example, the preprocessed data for our dblp toy example are [``data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl``](data/preprocessed/dblp/toy.dblp.v12.json/teams.pkl) and [``data/preprocessed/dblp/toy.dblp.v12.json/indexes.pkl``](data/preprocessed/dblp/toy.dblp.v12.json/indexes.pkl).

> Our pipeline benefits from parallel generation of sparse matrices for teams that significantly reduces the preprocessing time as shown below:
> 
> <p align="center"><img src="./data/speedup.jpg" width="200"><img src="./data/speedup_loglog.jpg" width="190"></p>


Please note that the preprocessing step will be executed once. Subsequent runs load the persisted pickle files. In order to regenerate them, one should simply delete them. 


#### **3.6. Model Architecture**

Each model has been defined in [``./src/mdl/``](./src/mdl/) under an inheritance hierarchy. They override abstract functions for ``train``, ``test``, ``eval``, and ``plot`` steps.

For example, for our feedforward baseline [``fnn``](./src/mdl/fnn.py), the model has been implemented in [``./src/mdl/fnn.py``](src/mdl/fnn.py). Model's hyperparameters such as the learning rate (``lr``) or the number of epochs (``e``) can be set in [``./src/param.py``](src/param.py).


<p align="center"><img src='./src/mdl/team_inheritance_hierarchy.png' width="550" ></p>
  
Currently, we support neural models:
1) Bayesian [``bnn``](./src/mdl/bnn.py) where model's parameter (weights) is assumed to be drawn from Gaussian (Normal) distribution and the task is to not to learn the weight but the mean (μ) and standard deviation (σ) of the distribution at each parameter.

<p align="center"><img src='./src/mdl/bnn.png' width="350" ></p>

2) non-Bayesian feedforward [``fnn``](./src/mdl/fnn.py) where the model's parameter (weights) is to be learnt.

The input to the models is the vector representations for (_temporal_) skills and the output is the vector representation for members. In another word, given the input skills, the models predict the members from the pool of candidates. We support three vector representations:

i) Sparse vector representation (occurrence or boolean vector): See preprocessing section above.

ii) Dense vector representation ([``team2vec``](src/mdl/team2vec/team2doc2vec.py)): Inspired by paragraph vectors by [Le and Mikolov](https://cs.stanford.edu/~quocle/paragraph_vector.pdf), we consider a team as a document and skills as the document words (``embtype == 'skill'``). Using distributed memory model, we map skills into a real-valued embedding space. Likewise and separately, we consider members as the document words and map members into real-valued vectors (``embtype == 'member'``). We also consider mapping skills and members into the same embedding space (``embtype == 'joint'``). Our embedding method benefits from [``gensim``](https://radimrehurek.com/gensim/) library.

3) In OpeNTF, The ``Nmt`` wrapper class is designed to make use of advanced transformer models and encoder-decoder models that include multiple ``LSTM`` or ``GRU`` cells, as well as various attention mechanisms. ``Nmt`` is responsible for preparing the necessary input and output elements and invokes the executables of ``opennmt-py`` by creating a new process using Python's ``subprocess`` module. Additionally, because the ``Nmt`` wrapper class inherits from ``Ntf``, these models can also take advantage of temporal training strategies through ``tNtf``.

#### **3.8. Run**

The pipeline accepts three required list of values:
1) ``-data``: list of path to the raw datafiles, e.g., ``-data ./../data/raw/dblp/dblp.v12.json``, or the main file of a dataset, e.g., ``-data ./../data/raw/imdb/title.basics.tsv``
2) ``-domain``: list of domains of the raw data files that could be ``dblp``, ``imdb``, or `uspt`; e.g., ``-domain dblp imdb``.
3) ``-model``: list of baseline models that could be ``fnn``, ``fnn_emb``, ``bnn``, ``bnn_emb``, ``random``; e.g., ``-model random fnn bnn ``.

Here is a brief explanation of the models:
- ``fnn``, ``bnn``, ``fnn_emb``, ``bnn_emb``: follows the standard machine learning training procedure.
  
## 4. Results

We used [``pytrec_eval``](https://github.com/cvangysel/pytrec_eval) to evaluate the performance of models on the test set as well as on their own train sets (should overfit) and validation sets. We report the predictions, evaluation metrics on each test instance, and average on all test instances in ``./output/{dataset name}/{model name}/{model's running setting}/``.  For example:

1) ``f0.test.pred`` is the predictions per test instance for a model which is trained folds [1,2,3,4] and validated on fold [0].
2) ``f0.test.pred.eval.csv`` is the values of evaluation metrics for the predictions per test instance
3) ``f0.test.pred.eval.mean.csv`` is the average of values for evaluation metrics over all test instances.
4) ``test.pred.eval.mean.csv`` is the average of values for evaluation metrics over all _n_ fold models.

**Benchmarks at Scale**

Full predictions of all models on test and training sets and the values of evaluation metrics are available in a rar file and will be delivered upon request! 




## 5. Acknowledgement:
We benefit from [``pytrec_eval``](https://github.com/cvangysel/pytrec_eval), [``gensim``](https://radimrehurek.com/gensim/), [Josh Feldman's blog](https://joshfeldman.net/WeightUncertainty/), and other libraries. We would like to thank the authors of these libraries and helpful resources.
  

