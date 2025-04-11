# OUTPUT Folder
This folder contains various classes and functions to save a sample
output of the VAE including diagrams, models, and predictions. The 
diagrams folder contains a saved sample metric performance diagram. The
Models folder contains various t2v generated sample models. The 
prediction folder contains the predicted indices made by the VAE model.

### Diagrams Folder
The [Diagrams](/teamFormationLibrary/output/diagrams) folder 
contains sample diagrams of the evaluation performance of the model.
The measures that were computed include: Recall, MRR, MAP, and NDCG. The
graph shows these measures over various top-k experts.

### Models Folder
The [Models](/teamFormationLibrary/output/Models) folder contains various 
sample T2V models that were developed through the Embedding stage of the 
library pipeline. These include: members2vec, team2vec, and team models. 
The model stores the weights as embedded vectors. These models are used during 
the remaining stages of the pipeline as vector representations.

### Predictions Folder
The [Predictions](/teamFormationLibrary/output/predictions) folder contains 
a sample predicted result of the model in a csv format. This file is called 
'S_VAE_O_output'. The indices in this file include the true indices and 
the predicted indices based on the top-k value chosen during previous stages
of the process. An example of a row in this file is shown in the image below:
<p align="center">
  <img width="500" height="20" src="https://i.ibb.co/G9tJDff/s.png">
</p>

The folder also contains a file for the sample correlation predictions, 
which is what we used as a baseline to compute the correlation. This file is 
called 'correlation_baseline_output'.