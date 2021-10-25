# Source Folder
This folder contains our implementation of neural team formation, including all the steps of the pipeline from loading the raw datasets, generating sparse matrix, building a neural model, training it, and evaluating it on classification and IR metrics.

In [cmn](/src/cmn), you can find the class definitions for teams and experts. 
The [team](/src/cmn/team.py) class is the parent class for both [publication](/src/cmn/publication.py) and [movie](/src/cmn/movie.py) classes.
In [team](/src/cmn/team.py), you can find functions for parallel generation of the sparse matrix, retrieving and plotting the stats of a dataset, and returning the unigram distribution of a dataset.
The [member](/src/cmn/member.py) class is the parent class for both [author](/src/cmn/author.py) and [castncrew](/src/cmn/castncrew.py) classes.
For the dblp dataset, we implemented the [publication](/src/cmn/publication.py) class, which contains the function for loading the raw dataset, transforming the data to [publication](/src/cmn/publication.py) and [author](/src/cmn/author.py) objects, and pickling them.
For the imdb dataset, we implemented the [movie](/src/cmn/movie.py) class, which contains the function for loading the raw dataset, transforming the data to [movie](/src/cmn/movie.py) and [castncrew](/src/cmn/castncrew.py) objects, and pickling them.

In [dal](/src/dal), we have implemented functions for splitting the dataset to train/validation/test sets, and measuring a model's perfromance on both classification metrics such as ROC and AUC, and IR metrics such as precision at k, recall at k, ndcg at k, and map at k.

In [mdl](/src/mdl), we have implemented the neural model in [nn.py](/src/mdl/nn.py) and a custom dataset class in [custom_dataset.py](/src/mdl/custom_dataset.py) that will be used to split the data into mini-batches.

In [misc](/src/misc), we have implemented a function that transforms our sparse matrix into a format that would work for Rad et al.'s variational Bayesian neural network.

In [pipeline.py](/src/pipeline.py), you can find the functions for training a neural model with and without negative sampling, plotting the loss for training and validaiton sets, evaluating the model on training and validation sets, and evaluating the model on the unseen test set.

In [param.py](/src/param.py), the settings for filtering the data and the number of processes and size of buckets for parallel generation of the sparse matrix can be found. Also, you can change the hyperparameters for the model in this file.

[main.py](/src/main.py), is the executable program that connects all different parts of our model. 