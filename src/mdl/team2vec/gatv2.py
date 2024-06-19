import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATv2Conv, to_hetero
from torch_geometric.data import Data,HeteroData

class GATv2(torch.nn.Module):
  def __init__(self, hidden_channels, heads=2, add_self_loops = False):
    super().__init__()
    self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=add_self_loops, heads = heads)
    self.conv2 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=add_self_loops, heads = heads)
    self.conv3 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=add_self_loops, heads = heads)
    self.conv4 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=add_self_loops, heads=1)

  def forward(self, x, edge_index):
      x = F.relu(self.conv1(x, edge_index))
      x = F.relu(self.conv2(x, edge_index))
      x = F.relu(self.conv3(x, edge_index))
      x = self.conv4(x, edge_index)
      return x
