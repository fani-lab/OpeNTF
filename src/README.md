# Source Code 
This folder contains our implementation of neural team formation, including all the steps of the pipeline from loading the raw datasets, generating sparse matrix, building a neural model, training it, and evaluating it on classification and IR metrics.

1) [``cmn``](./cmn), you can find the abstract class definitions for teams and members as well as inheritance hierarchy for different domains, including:
   1) [``team.py``](./cmn/team.py) class is the parent class for a team 
   2) [``publication.py``](./cmn/publication.py) (research papers as teams and authors as members)
   3) [``movie.py``](./cmn/movie.py) imdb (movies as teams, and cast and crews as members)
   4) [``patent.py``](./cmn/patent.py) (patents as teams and inventors as members)
    
2) [``mdl``](./mdl), we have implemented the neural models here, e.g., [``fnn.py``](./mdl/fnn.py) and a custom dataset class in [``cds.py``](./mdl/cds.py) that will be used to split the data into mini-batches.
3) [``eval``](./eval), we report and plot evaluation of models based on classification metrics and IR metrics here.
4) [``main.py``](./main.py), the main entry point to the benchmark pipeline that realizes all steps related to ``train`` a neural model and plotting the loss, ``test`` on the unseen test set, and ``eval`` based on classification and IR metrics through `polymorphism`.
5) [``param.py``](./param.py), the settings for filtering the data, the number of processes, size of buckets for parallel generation of the sparse matrix, hyperparameters for the model, are all in this file.
6) [``main.py``](./main.py), is the main entry point to the pipeline that provides end-to-end train-test-eval benchmark on different baselines and datasets. 
