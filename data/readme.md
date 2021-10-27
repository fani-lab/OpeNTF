# Data Folder
You can find raw datasets in [raw](/data/raw) and the preprocessed data including teams pickles and the sparse matrices in [preprocessed](/data/preprocessed).
In [preprocessed/{name_of_dataset}/{title_of_dataset}](/data/preprocessed), you can find ```teams.pkl```, ```indexes.pkl```, and ```teamsvecs.pkl```, which are the teams, indices, and sparse matrix pickles respectively.
If you have used filter=1 when running the main file, the folder name would have ```.filtered.mt{min_nteam}.ts{min_team_size}``` at the end of its name.