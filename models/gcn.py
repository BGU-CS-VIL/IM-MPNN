import torch.nn as nn
import torch.nn.functional as F
from .layers.gcn_conv_layer import GCNConvLayer


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.2,
        residual: bool = True,
        use_mlp_enc_dec: bool = True,
        act=nn.ReLU,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.use_mlp_enc_dec = use_mlp_enc_dec
        
        self.convs = nn.ModuleList()

        if use_mlp_enc_dec:
            self.enc = nn.Linear(in_channels, hidden_channels)
            self.dec = nn.Linear(hidden_channels, out_channels)
            in_channels = hidden_channels
            mid_layers = num_layers
        else:
            self.enc = nn.Identity()
            self.dec = nn.Identity()
            mid_layers = num_layers - 2
        
        assert mid_layers >= 0, "num_layers too low"

        if not use_mlp_enc_dec:
            self.convs.append(GCNConvLayer(in_channels, hidden_channels, dropout, residual=False, act=act))
        
        for _ in range(mid_layers):
            self.convs.append(GCNConvLayer(hidden_channels, hidden_channels, dropout, residual, act=act))
        
        if not use_mlp_enc_dec:
            self.convs.append(GCNConvLayer(hidden_channels, out_channels, dropout=0.0, residual=False, act=nn.Identity))

    def forward(self, batch):
        """
        Forward pass for node-level classification on one (or a mini-batch of) graphs.
        """
        
        batch.x = self.enc(batch.x)

        # Go through all layers except the last one with ReLU (hidden layers)
        for conv in self.convs:
            batch = conv(batch)

        batch.x = self.dec(batch.x)

        return batch.x