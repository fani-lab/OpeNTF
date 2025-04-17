from torch import nn
from torch import Tensor

class Decoder(nn.Module):
    def forward(self, source_node_emb, target_node_emb, edge_label_index) -> Tensor:
        # Convert node embeddings to edge-level representations:
        # (e.g. : shapes -> edge_feat_source (4, 16), edge_feat_target (4, 16))
        # the number of rows correspond to total number of labeled_edges in the seed_edge_type, in this case, 4
        edge_feat_source = source_node_emb[edge_label_index[0]]
        edge_feat_target = target_node_emb[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge in the edge_label_index
        return (edge_feat_source * edge_feat_target).sum(dim=-1)