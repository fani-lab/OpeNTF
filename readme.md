# ``OpeNTF``: An Open-Source Neural Team Formation Benchmark Library 
Collaborative teams are the primary vehicle for coordinating experts with diverse skills needed for a particular collaborative project, and Team Formation (TF) has firsthand effects on creating organizational performance. Social network analysis has been employed for TF by incorporating social ties and interpersonal collaboration features using measures such as degree and closeness centrality. Socially driven methods, however, face two significant challenges, esp., when currency (timeliness) is of the prime concern: ``i) temporality``: the expert's skills and her social attributes constantly change over time. A successful collaboration of experts in a team years ago does not tailor a successful team now, ``ii) complexity``: optimum teams are found by computationally expensive search over all the subgraphs of a large-scale collaboration graph (search-based.) 

## [![license](https://img.shields.io/badge/Demo-youtu.be/nFLow5Oy9nw-critical?style=&logo=youtube)](https://www.youtube.com/watch?v=nFLow5Oy9nw)
We propose ``neural machine learning`` approaches to Team Formation. We train neural models that would learn relationships among experts and their social attributes in vector space. Wherein, we consider all past (un)successful team compositions as training samples to predict future teams and the team's level of success. Therefore, we bring efficiency while enhancing efficacy due to the inherently iterative and online learning procedure in neural architectures. More importantly, we will address temporality by incorporating sequences on the neural architecture via recurrence, which yields a richer representation and more accurate team composition within time. We are the first who consider temporal graph neural networks for team formation and will provide major advances via ``1) time-sensitive`` and ``2) scalable`` incorporation of experts' attributes and skills. 

1. [Setup](#1-setup)
2. [Quickstart](#2-quickstart)
3. [Features](#3-features)
4. [Results](#4-results)
5. [Acknowledgement](#5-acknowledgement)
6. [License](#6-license)
7. [Citation](#7-citation)
8. [Awards](#8-awards)

## 1. [Setup](https://colab.research.google.com/github/fani-lab/OpeNTF/blob/main/quickstart.ipynb)
You need to have ``Python >= 3.8`` and install the following main packages, among others listed in [``requirements.txt``](requirements.txt):
```
torch>=1.9.0
pytrec-eval-terrier==0.5.2
gensim==3.8.3
```
By ``pip``, clone the codebase and install required packages:
```sh
git clone https://github.com/Fani-Lab/opentf
cd opentf
pip install -r requirements.txt
```
By [``conda``](https://www.anaconda.com/products/individual):

```sh
git clone https://github.com/Fani-Lab/opentf
cd opentf
conda env create -f environment.yml
conda activate opentf
```

For installation of specific version of a python package due to, e.g., ``CUDA`` versions compatibility, one can edit [``requirements.txt``](requirements.txt) or [``environment.yml``](environment.yml) like as follows:

```
# CUDA 10.1
torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
## 2. Quickstart [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fani-lab/OpeNTF/blob/main/quickstart.ipynb)

```sh
cd src
python -u main.py -data ../data/raw/dblp/toy.dblp.v12.json -domain dblp -model fnn bnn
```

The above run, loads and preprocesses a tiny-size toy example dataset [``toy.dblp.v12.json``](data/raw/dblp/toy.dblp.v12.json) from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z) followed by _n_-fold train-evaluation on a training split and final test on the test set for ``feedforward`` and ``Bayesian`` neural models using default hyperparameters from [``./src/param.py``](./src/param.py).

```
python -u main.py -data ../data/raw/dblp/toy.dblp.v12.json -domain dblp -model tbnn tbnn_dt2v_emb
```

This script loads and preprocesses the same dataset [``toy.dblp.v12.json``](data/raw/dblp/toy.dblp.v12.json) from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z), takes the teams from the the last year as the test set and trains the ``Bayesian`` neural model following our proposed streaming training strategy as explained in ``3.2.2. Temporal Neural Team Formation`` with two different input representations _i_) sparse vector represntation and _ii_) temporal skill vector represntation using default hyperparameters from [``./src/param.py``](./src/param.py).

## 3. Features
**3.1. Datasets and Parallel Preprocessing**

Raw dataset, e.g., scholarly papers from AMiner's citation network dataset of [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z), movies from [``imdb``](https://datasets.imdbws.com/), or US patents from [``uspt``](https://patentsview.org/download/data-download-tables) were assumed to be populated in [``data/raw``](data/raw). For the sake of integration test, tiny-size toy example datasets [``toy.dblp.v12.json``](data/raw/dblp/toy.dblp.v12.json) from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z), [[``toy.title.basics.tsv``](data/raw/imdb/toy.title.basics.tsv), [``toy.title.principals.tsv``](data/raw/imdb/toy.title.principals.tsv), [``toy.name.basics.tsv``](data/raw/imdb/toy.name.basics.tsv)] from [``imdb``](https://datasets.imdbws.com/) and [``toy.patent.tsv``](data/preprocessed/uspt/toy.patent.tsv) have been already provided.

<p align="center"><img src='./src/cmn/team_inheritance_hierarchy.png' width="250" ></p>

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

**3.2. [Train][Validation][Test] Split**

We use _n_-fold cross-validation, that is, we train a model _n_ times on _(n-1)_ folds and utilize the remaining fold as the validation set to adjust the learning rate during training. The number of folds is set by ``nfolds`` in [``./src/param.py``](./src/param.py). In total, we have _n_ models each of which will be evaluated on the test set.

At each run, we store ids of instances in train-validation folds and test set in [``./data/preprocessed/{dblp,imdb,uspt}/{name of dataset}/splits.json``](data/preprocessed/) like in [``./data/preprocessed/dblp/toy.dblp.v12.json/splits.json``](./data/preprocessed/dblp/toy.dblp.v12.json/splits.json)

**3.2.1. Non-Temporal Neural Team Formation**

We randomly take ``85%`` of the dataset for the train-validation set and ``15%`` as the test set, i.e., the model never sees these instances during training or model tuning. You can change ``train_test_split`` parameter in [``./src/param.py``](./src/param.py).

**3.2.2. Temporal Neural Team Prediction**

Previous works in team formation presumed that teams follow the i.i.d property and hence when training their models they followed the bag of teams approach, where they train and validate their models on a shuffled dataset of teams. Moreover, they were interpolative and did not try to predict _future_ successful teams. In this work, we aim at extrapolating and predicting _future_ teams of experts. We sort the teams by time intervals and train a neural model incrementally  through the ordered collection of teams in [C<sub>0</sub>, ..C<sub>t</sub>, ..C<sub>T</sub>]. As can be seen in Figure below, after random initialization of skills’ and experts’ embeddings at t=0, we start training the model on the teams in the first time interval C<sub>0</sub> for a number of epochs, then we continue with training  on the second time interval C<sub>1</sub> using the learned embeddings from the previous time interval and so forth until we finish the training on the last training time interval C<sub>t=T</sub>. We believe that using this approach, will help the model understand how experts’ skills and collaborative ties evolve through time and the final embeddings are their optimum representation in the latent space to predict _future_ successful teams at time interval C<sub>t=T+1</sub>.

<p align="center"><img src='./src/mdl/tntf.png' width="700"></p>

**3.3. Model Architecture**

Each model has been defined in [``./src/mdl/``](./src/mdl/) under an inheritance hierarchy. They override abstract functions for ``train``, ``test``, ``eval``, and ``plot`` steps. 
For example, for our feedforward baseline [``fnn``](./src/mdl/fnn.py), the model has been implemented in [``./src/mdl/fnn.py``](src/mdl/fnn.py). Model's hyperparameters such as the learning rate (``lr``) or the number of epochs (``e``) can be set in [``./src/param.py``](src/param.py).

<p align="center"><img src='./src/mdl/model_inheritance_hierarchy.png' width="350" ></p>
  
Currently, we support neural models:
1) Bayesian [``bnn``](./src/mdl/bnn.py) where model's parameter (weights) is assumed to be drawn from Gaussian (Normal) distribution and the task is to not to learn the weight but the mean (μ) and standard deviation (σ) of the distribution at each parameter.

<p align="center"><img src='./src/mdl/bnn.png' width="350" ></p>

2) non-Bayesian feedforward [``fnn``](./src/mdl/fnn.py) where the model's parameter (weights) is to be learnt.

The input to the models is the vector representations for (_temporal_) skills and the output is the vector representation for members. In another word, given the input skills, the models predict the members from the pool of candidates. We support three vector representations:

1) Sparse vector representation (occurrence or boolean vector): See preprocessing section above.
2) Dense vector representation ([``team2vec``](./src/mdl/team2vec.py)): Inspired by paragraph vectors by [Le and Mikolov](https://cs.stanford.edu/~quocle/paragraph_vector.pdf), we consider a team as a document and skills as the document words (``embtype == 'skill'``). Using distributed memory model, we map skills into a real-valued embedding space. Likewise and separately, we consider members as the document words and map members into real-valued vectors (``embtype == 'member'``). We also consider mapping skills and members into the same embedding space (``embtype == 'joint'``). Our embedding method benefits from [``gensim``](https://radimrehurek.com/gensim/) library.
3) Temporal skill vector represntation ([``team2vec``](./src/mdl/team2vec.py)): Inspired by [Hamilton et al.](https://aclanthology.org/P16-1141/), we also incorporate time information into the underlying neural model besides utilizing our proposed streaming training strategy. We used the distributed memory model of Doc2Vec to generate the real-valued joint embeddings of the subset of skills and time intervals, where the skills and time intervals are the words of the document (``embtype == 'dt2v'``).

**3.4. Negative Sampling Strategies**

As known, employing ``unsuccessful`` teams convey complementary negative signals to the model to alleviate the long-tail problem. Most real-world training datasets in the team formation domain, however, do not have explicit unsuccessful teams (e.g., collections of rejected papers.) In the absence of unsuccessful training instances, we proposed negative sampling strategies based on the ``closed-world`` assumption where no currently known successful group of experts for the required skills is assumed to be unsuccessful.  We study the effect of ``three`` different negative sampling strategies: two based on static distributions, and one based on adaptive noise distribution:

1) Uniform distribution (``uniform``), where subsets of experts are randomly chosen with the ``same probability`` as unsuccessful teams from the uniform distribution over all subsets of experts.

2) Unigram distribution (``unigram``), where subsets of experts are chosen regarding ``their frequency`` in all previous successful teams. Intuitively, teams of experts that have been more successful but for other skill subsets will be given a higher probability and chosen more frequently as a negative sample to dampen the effect of popularity bias.

3) Smoothed unigram distribution in each training minibatch (``unigram_b``), where we employed the ``add-1 or Laplace smoothing`` when computing the unigram distribution of the experts but in each training minibatch. Minibatch stochastic gradient descent is the _de facto_ method for neural models where the data is split into batches of data, each of which is sent to the model for the partial calculation to speed up training while maintaining high accuracy. 

To include a negative sampling strategy, there are two parameters for a model to set in [``./src/param.py``](src/param.py):
- ``ns``: the negative sampling strategy which can be ``uniform``, ``unigram``, ``unigram_b`` or ``None``(no negative sampling).
- ``nns``: number of negative samples

**3.5. Run**

The pipeline accepts three required list of values:
1) ``-data``: list of path to the raw datafiles, e.g., ``-data ./../data/raw/dblp/dblp.v12.json``, or the main file of a dataset, e.g., ``-data ./../data/raw/imdb/title.basics.tsv``
2) ``-domain``: list of domains of the raw data files that could be ``dblp``, ``imdb``, or `uspt`; e.g., ``-domain dblp imdb``.
3) ``-model``: list of baseline models that could be ``fnn``, ``fnn_emb``, ``bnn``, ``bnn_emb``, ``tfnn``, ``tfnn_emb``, ``tfnn_dt2v_emb``, ``tbnn``, ``tbnn_emb``, ``tbnn_dt2v_emb``, ``random``; e.g., ``-model random fnn bnn tfnn tbnn tfnn_dt2v_emb tbnn_dt2v_emb``.

Here is a brief explanation of the models:
- ``fnn``, ``bnn``, ``fnn_emb``, ``bnn_emb``: follows the standard machine learning training procedure.
- ``tfnn``, ``tbnn``, ``tfnn_emb``, ``tbnn_emb``: follows our proposed streaming training strategy without adding temporal information to the input of the models.
- ``tfnn_dt2v_emb``, ``tbnn_dt2v_emb``: follows our proposed streaming training strategy and employs temporal skills as input of the models.

## 4. Results

We used [``pytrec_eval``](https://github.com/cvangysel/pytrec_eval) to evaluate the performance of models on the test set as well as on their own train sets (should overfit) and validation sets. We report the predictions, evaluation metrics on each test instance, and average on all test instances in ``./output/{dataset name}/{model name}/{model's running setting}/``.  For example:

1) ``f0.test.pred`` is the predictions per test instance for a model which is trained folds [1,2,3,4] and validated on fold [0].
2) ``f0.test.pred.eval.csv`` is the values of evaluation metrics for the predictions per test instance
3) ``f0.test.pred.eval.mean.csv`` is the average of values for evaluation metrics over all test instances.
4) ``test.pred.eval.mean.csv`` is the average of values for evaluation metrics over all _n_ fold models.

**Benchmarks at Scale**

**1. Non-Temporal Neural Team Formation**

||min. #member's team: 75, min team size: 3, epochs: 20, learning rate: 0.1, hidden layer: [1, 100d], minibatch: 4096, #negative samples: 3|
|--------|------|
|Datasets|[dblp.v12](https://originalstatic.aminer.cn/misc/dblp.v12.7z), [imdb](https://imdb.com/interfaces/), [uspt](https://patentsview.org/download/data-download-tables) (running ...)|
|Metrics|recall@2,5,10, map@2,5,10, ndcg@2,5,10, p@2,5,10, auc|
|Baselines|{fnn,bnn}×{sparse, emb}×{none, uniform, unigram, unigram_b}|
|Results|[``./output/dblp.v12.json.filtered.mt75.ts3/``](./output/dblp.v12.json.filtered.mt75.ts3/), [``./output/title.basics.tsv.filtered.mt75.ts3/``](./output/title.basics.tsv.filtered.mt75.ts3/)|

<p align="center">
<img src='https://user-images.githubusercontent.com/8619934/154041216-c80cccfb-70a2-4831-8781-cdb4718fb00e.png' >
<img src='https://user-images.githubusercontent.com/8619934/154041087-e4d99b1e-eb6b-456a-837b-840e4bd5090a.png' >

Full predictions of all models on test and training sets and the values of evaluation metrics, per instance and average, are available in a rar file of size ``74.8GB`` and will be delivered upon request! 

**2. Temporal Neural Team Prediction**

We kick-started our experiments based on the best results from the non-temporal neural team formation experiments.

||min. #member's team: 75, min team size: 3, epochs: 20, learning rate: 0.1, hidden layer: [1, 128d], minibatch: 128, #negative samples: 3|
|--------|------|
|Datasets|[dblp.v12](https://originalstatic.aminer.cn/misc/dblp.v12.7z), [imdb](https://imdb.com/interfaces/), [uspt](https://patentsview.org/download/data-download-tables)|
|Metrics|recall@2,5,10, map@2,5,10, ndcg@2,5,10, p@2,5,10, auc|
|Baselines|{bnn, tbnn}×{sparse, emb, dt2v_emb}×{unigram_b},{[rrn](https://dl.acm.org/doi/10.1145/3018661.3018689)}|
|Results|[``./output/dblp.v12.json.filtered.mt75.ts3/``](./output/dblp.v12.json.filtered.mt75.ts3/), [``./output/title.basics.tsv.filtered.mt75.ts3/``](./output/title.basics.tsv.filtered.mt75.ts3/), [``./output/patent.tsv.filtered.mt75.ts3/``](./output/patent.tsv.filtered.mt75.ts3/)|

<p align="center"><img src='./output/ecir_results.PNG'></p>

Full predictions of all models on test and training sets and the values of evaluation metrics are available in a rar file and will be delivered upon request! 

## 5. Acknowledgement:
We benefit from [``pytrec_eval``](https://github.com/cvangysel/pytrec_eval), [``gensim``](https://radimrehurek.com/gensim/), [Josh Feldman's blog](https://joshfeldman.net/WeightUncertainty/), and other libraries. We would like to thank the authors of these libraries and helpful resources.
  
## 6. License:
©2024. This work is licensed under a [CC BY-NC-SA 4.0](license.txt) license.

## 7. Citation:
```
@inproceedings{DBLP:conf/ecir/FaniBDS24,
  author       = {Hossein Fani and
                  Reza Barzegar and
                  Arman Dashti and
                  Mahdis Saeedi},
  title        = {A Streaming Approach to Neural Team Formation Training},
  booktitle    = {Advances in Information Retrieval - 46th European Conference on Information Retrieval, {ECIR} 2024, Glasgow, UK, March 24-28, 2024, Proceedings, Part {I}},
  series       = {Lecture Notes in Computer Science},
  volume       = {14608},
  pages        = {325--340},
  publisher    = {Springer},
  year         = {2024},
  url          = {https://doi.org/10.1007/978-3-031-56027-9\_20},
  doi          = {10.1007/978-3-031-56027-9\_20},
  timestamp    = {Wed, 24 Apr 2024 14:55:54 +0200},
  biburl       = {https://dblp.org/rec/conf/ecir/FaniBDS24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```
@inproceedings{DBLP:conf/cikm/DashtiSF22,
  author    = {Arman Dashti and Saeed Samet and Hossein Fani},
  title     = {Effective Neural Team Formation via Negative Samples},
  booktitle = {Proceedings of the 31st {ACM} International Conference on Information {\&} Knowledge Management, Atlanta, GA, USA, October 17-21, 2022},
  pages     = {3908--3912},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3511808.3557590},
  doi       = {10.1145/3511808.3557590},
  biburl    = {https://dblp.org/rec/conf/cikm/DashtiSF22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
  ```
  ```
@inproceedings{DBLP:conf/cikm/DashtiSPF22,
  author    = {Arman Dashti and Karan Saxena and Dhwani Patel and Hossein Fani},
  title     = {OpeNTF: {A} Benchmark Library for Neural Team Formation},
  booktitle = {Proceedings of the 31st {ACM} International Conference on Information {\&} Knowledge Management, Atlanta, GA, USA, October 17-21, 2022},
  pages     = {3913--3917},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3511808.3557526},
  doi       = {10.1145/3511808.3557526},
  biburl    = {https://dblp.org/rec/conf/cikm/DashtiSPF22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## 8. Awards:

> [CAD$300, Gold medalist, UWill Discover, 2022](https://scholar.uwindsor.ca/uwilldiscover/2022/2022Day3/30/)

> CAD$300, Best Research, Demo Day, School of Computer Science, University of Windsor, 2022. 
