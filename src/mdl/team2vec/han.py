import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import HANConv, to_hetero
from torch_geometric.data import Data,HeteroData

# HAN applies attention on metapath based neighbors of nodes. So we need to add metapaths to
# the existing heterodata object (data.HeteroData) to make this information available

class HAN(torch.nn.Module):
  def __init__(self, hidden_channels, metadata, heads=2):
    super().__init__()
    self.conv1 = HANConv(hidden_channels, hidden_channels, metadata = metadata, heads = heads)
    self.conv2 = HANConv(hidden_channels, hidden_channels, metadata = metadata, heads = heads)
    self.conv3 = HANConv(hidden_channels, hidden_channels, metadata = metadata, heads = heads)
    self.lin = Linear(hidden_channels, hidden_channels)

  def forward(self, x, edge_index):
      x = self.conv1(x, edge_index)
      x = self.conv2(x, edge_index)
      x = self.conv3(x, edge_index)
      for node_type in x.keys():
          x[node_type] = self.lin(x[node_type])
      return x
