# Source Code 
This folder contains our implementation of neural team formation, including all the steps of the pipeline from loading the raw datasets, generating sparse matrix, building a neural model, training it, and evaluating it on classification and IR metrics.

1) [``main.py``](./main.py), the main entry point to the pipeline that provides end-to-end train-test-eval benchmark on different baselines and datasets
2) [``__config__.yaml``](./__config__.yaml), the main settings for preprocessing data, and train, test, and evaluation strategies
3) [``./cmn/``](./cmn), you can find the abstract class definitions for teams and members as well as the inheritance hierarchy for different domains, including:
   1) [``team.py``](./cmn/team.py) is the definition for the `Team` as the abstract parent class for a team with members and skills as two mandatory attributes 
   2) [``publication.py``](./cmn/publication.py) is the definition for the `Publication` class, where publications are teams, authors are the members, and keywords are the skills, like in [``data/dblp``](data/dblp)
   3) [``movie.py``](./cmn/movie.py) is the definition for the `Movie` class, where movies are teams, cast'n crews are members, and genres and subgenres are skills like in [``data/imdb``](data/imdb)
   4) [``patent.py``](./cmn/patent.py) is the definition for the `Patent` class, where patents are teams, inventors are members, and patents' class/subclasses are skills like in [``data/uspt``](data/uspt)
   5) [``repository.py``](./cmn/repository.py) is the definition for the `Repository` class, where software repos in github are teams, software developers are members, and programming languages are skills like in [``data/gith``](data/gith)
  
<p align="center"><img src='../docs/datasets_class_diagram.png' width="500" ></p>
    
4) [``./mdl/``](./mdl), we have implemented the neural models here:
   1) [``ntf.py``](./mdl/ntf.py), the parent abstract class definition of a team formation model, to be overriden/realized by machine learning models 
   2) [``rnd.py``](./mdl/Rnd.py), a `random` model that map `1-hot` or `dense` vector of `skills` to a random `1-hot` vector, as a minimum baseline
   3) [``fnn.py``](./mdl/fnn.py) and [``bnn.py``](./mdl/bnn.py), the feedforward non-variational and varational (Bayesian) neural multilabel classifiers that map `1-hot` or `dense` vector of `skills` to `1-hot` vector of `members` 
   4) [``nmt.py``](./mdl/nmt.py), the wrapper class over [`OpenNMT`](https://github.com/OpenNMT/OpenNMT-py) for `translative` models like `seq-2-seq` and `transformers` models, where the required subset of `skills` is mapped to the optimum subset of `members`
   5) [``tntf.py``](./mdl/tntf.py), the wrapper class over [``ntf.py``](./mdl/ntf.py) for streaming/temporal training of a model

<p align="center"><img src='../docs/models_class_diagram.png' width="500" ></p>

5) [``./eval/``](./eval), we report evaluation of models based on quantitative metrics (classification and ranking metrics) as well as qualitative metrics here.
