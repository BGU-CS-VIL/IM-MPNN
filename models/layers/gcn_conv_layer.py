# Adapted from tuned-LRGB: https://github.com/toenshoff/LRGB
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F


class GCNConvLayer(nn.Module):
    """Graph Isomorphism Network with Edge features (GINE) layer.
    """
    def __init__(self, dim_in, dim_out, dropout, residual, act=nn.ReLU):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        self.act = act()
        self.model = pyg_nn.GCNConv(dim_in, dim_out, bias=True)

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x, batch.edge_index)
        batch.x = self.act(batch.x)
        batch.x = F.dropout(batch.x, self.dropout, training=self.training)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch
