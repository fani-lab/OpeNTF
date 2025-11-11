# Pipeline Outputs

The preprocessed version of datasets and models' footprint like checkpoints, prediction files, and evaluation results, are all here:

1. `./output/{domain}/`, the specific subfolder for a domain like [`./dblp/`](./dblp/)

2. `./output/{domain}/{dataset}`, the folder for a dataset in a domain like [`./dblp/toy.dblp.v12.json`](./dblp/toy.dblp.v12.json)
  > if data filtering enabled in [`../src/__config__.yaml#L38`](../src/__config__.yaml#L38), the folder name includes `min_nteam` and `min_team_size` like `./output/{domain}/{dataset}.mt{min_nteam}.ts{min_team_size}` in [`./dblp/toy.dblp.v12.json.mt10.ts2`](./dblp/toy.dblp.v12.json.mt10.ts2)

For each dataset, preprocessed files are created such that all models will be using same source of data including:
    
  - `teams.pkl`, the object version of team instances in the raw dataset
  - `indexes.pkl`, the mapping between ids and names in raw dataset and the rowid in the preprocessed matrix version 
  - `teamsvecs.pkl`, the sparse vector representation of teams, each row of which is a team (T) with 1-hot vectors of skills (S) and members/experts (E) of size |T|×|S + E|
  - `skillcoverage.pkl`, a matrix representing skill coverage of each member/expert of size |E|×|S|
  - `{graph structure}.{edge merge}.graph.pkl`, if a `gnn` model is used for either skill embedding or end-to-end team recommendation, a graph will be created based on [`../src/mdl/emb/__config__.yaml#L27`](../src/mdl/emb/__config__.yaml#L27) 

Our pipeline benefits from the parallel generation of sparse matrices for teams, which significantly reduces the preprocessing time as shown below:

<p align="center"><img src="../docs/speedup.jpg" width="200"><img src="../docs/speedup_loglog.jpg" width="190"></p>

3. `splits.f{nfolds}.r{train_test_ratio}`, the subfolder that contains the models' footprint for train-test splits based on `train_test_ratio` and `nfolds` cross-validation during training, like [`./dblp/toy.dblp.v12.json/splits.f3.r0.85`](./dblp/toy.dblp.v12.json/splits.f3.r0.85)

``` 
.
├── teams.pkl
├── indexes.pkl
├── teamsvecs.pkl
├── skillcoverage.pkl
├── stm.mean.graph.pkl
├── splits.f3.r0.85.pkl                             #contains the splits information about train folds and test splits
├── splits.f3.r0.85
    ├── {model}.b{batch size}.e{max epoch}.ns{negative sample}.lr{learning rate}.es{early stopping}.h[{layers}]...
    │   ├── f0.e0.pt 
    │   ├── f0.e9.pt
    │   ├── f0.pt                                   #last checkpoint at early stopping
    │   ├── f0.test.e0.pred                         #per epoch inferences 
    │   ├── f0.test.e9.pred
    │   ├── f0.test.pred                            #model's final inferences  
    │   ├── f0.test.e0.pred.eval.per_instance.csv   #per epoch inferences' evaluations per each team instance in fold 0
    │   ├── f0.test.e0.pred.eval.mean.csv           #per epoch inferences' evaluations average over team instance in fold 0
    │   ├── f0.test.e9.pred.eval.mean.csv
    │   ├── f0.test.e9.pred.eval.per_instance.csv
    │   ├── f0.test.pred.eval.mean.csv              #model's final inferences' evaluations average over team instance in fold 0
    │   ├── f0.test.pred.eval.per_instance.csv
    │   ├── test.pred.eval.per_instance_mean.csv    #model's final inferences' evaluations per instance average over all folds
    │   ├── test.pred.eval.mean.csv                 #model's final inferences' evaluations average over team instance over all folds
    │   └── logs4tboard                             #tensorboard visualization data
    │       └── run_1759698340
    │           └── events.out.tfevents.1759698340.Dakho-Mac-Pro.local
    ├── {gnn model}.{hyperparameters}
        ├── {transfer model}.{hyperparameters}      #if the gnn model is used for skill embeddings to be fed to fnn, bnn, etc
        ├── f0.e0.pt
        ├── f0.pt
        ├── f0.test.e0.pred
        ├── f0.test.pred
        ├── f0.test.e0.pred.eval.mean.csv
        ├── f0.test.e0.pred.eval.per_instance.csv
        ├── f0.test.pred.eval.mean.csv
        ├── f0.test.pred.eval.per_instance.csv
        ├── test.pred.eval.mean.csv
        ├── test.pred.eval.per_instance_mean.csv
        └── logs4tboard
            └── run_1759698865
                └── events.out.tfevents.1759698865.Dakho-Mac-Pro.local
```
