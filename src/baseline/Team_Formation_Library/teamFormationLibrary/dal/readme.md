# Data Access Layer Folder
This folder contains various sub-components of the Data Access Layer,
mainly the Data Loader and the Embedding components. The Data Loader
enables the loading of various files from local folders (i.e.user 
dataset, t2v model, train/test indices).

Example of usage:
```python
DAL = DataAccessLayer(database_name, database_path, embeddings_save_path) #creates an instance of the DAL
DAL.generate_embeddings() #generates embeddings for a DAL instance
```
