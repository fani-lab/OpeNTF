You can find raw datasets in [raw](./raw) and the preprocessed data including teams pickles and the sparse matrices in [preprocessed](./preprocessed).
In [preprocessed/{domain}/{dataset name}](./preprocessed), you can find:
1) ```teams.pkl```
2) ```indexes.pkl```
3) ```teamsvecs.pkl```

which are the teams, indices, and sparse matrix pickles respectively.

If you have used ``filter=1`` when running the main file, the folder name would have ```.filtered.mt{min_nteam}.ts{min_team_size}``` at the end of its name. 
This means that we filter out those members that have participated in less than ``min_nteam`` as well as those teams that have less than ``min_team_size`` members. 
The filtering settings can be set by the ``filter`` parameter at [``./../src/param.py``](./../src/param.py)

Our pipeline benefits from parallel generation of sparse matrices for teams that significantly reduces the preprocessing time as shown below:

<p align="center"><img src="./speedup.jpg" width="400"><img src="./speedup_loglog.jpg" width="370"></p>
