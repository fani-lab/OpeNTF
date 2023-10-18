import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

def load_dataset():
    # Load a graph dataset
    dataset = KarateClub()
    return dataset

def explore_dataset(dataset):

    # Access the first graph in the dataset
    data = dataset[0]
    print(f'Type of the class data : {type(data)}')

    # Print some information about the dataset
    # print(f"Dataset name: {dataset.name}")
    print(f"Number of graphs in the dataset: {len(dataset)}")
    print(f"Number of nodes in the graph: {data.num_nodes}")
    print(f"Number of edges in the graph: {data.num_edges}")
    print(f"Node features dimension: {data.num_features}")
    # print(f"Node labels dimension: {data.num_classes}")

    # Check if the dataset is undirected or directed
    print(f"Is the graph directed? {data.is_directed()}")

if __name__ == "__main__":
    dataset = load_dataset()
    explore_dataset(dataset)
