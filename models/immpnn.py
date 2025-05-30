import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_scatter import scatter

from .layers.gcn_conv_layer import GCNConvLayer
# from .layers.gatedgcn_layer import GatedGCNLayer
# from .layers.gine_conv_layer import GINEConvLayer


class IMMPNN(torch.nn.Module):
    """
    Interleaved Multi-scale Message Passing Neural Network (IM-MPNN)
    
    A graph neural network that processes information at multiple scales simultaneously,
    with interleaved message passing between different scales. This architecture allows
    for capturing both local and global structures in graph data.
    
    The network consists of:
    1. An encoder to project input features to hidden dimension
    2. Multiple multiscale graph convolution layers
    3. Interleave MLPs to facilitate information flow between scales
    4. A classifier to produce final predictions
    
    Features from different scales are combined using hierarchical pooling and unpooling
    operations based on the graclus clustering algorithm.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, scales=3, dropout=0.2, residual=True, conv_type='gcnconv'):
        """
        Args:
            in_channels (int): Dimension of input node features
            hidden_channels (int): Dimension of hidden node features
            out_channels (int): Dimension of output features/number of classes
            num_layers (int): Number of message passing layers
            scales (int, optional): Number of scales to use. Defaults to 3.
            dropout (float, optional): Dropout probability. Defaults to 0.2.
            residual (bool, optional): Whether to use residual connections. Defaults to True.
            conv_type (str, optional): Type of graph convolution to use. Defaults to 'gcnconv'.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.scales = scales
        self.dropout = dropout
        self.residual = residual
        self.conv_type = conv_type

        self.encoder = torch.nn.Linear(self.in_channels, self.hidden_channels)

        self.multiscale_convs = torch.nn.ModuleList([
            MultiscaleGraphConvLayer(self.hidden_channels, self.hidden_channels, self.scales, self.conv_type, self.dropout, self.residual)
            for _ in range(self.num_layers)
        ])

        self.interleave_mlps = torch.nn.ModuleList([
            InterleaveMLP(self.hidden_channels, self.scales)
            for _ in range(self.num_layers - 1)
        ])

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_channels * (self.scales + 1), self.hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_channels, self.out_channels),
        )

    def create_multiscale(self, batch):
        batches_in_scales = [batch]
        for _ in range(self.scales):
            curr_batch = batches_in_scales[-1]
            cluster = pyg_nn.graclus(curr_batch.edge_index, num_nodes=curr_batch.x.shape[0])
            cluster, _ = pyg_nn.pool.consecutive.consecutive_cluster(cluster)
            pooled_batch = pyg_nn.pool.avg_pool(cluster, curr_batch)
            pooled_batch.cluster = cluster
            batches_in_scales.append(pooled_batch)
        return batches_in_scales

    def concat_scales(self, batches_in_scales):
        for i in range(self.scales, 0, -1):
            higher_scale_x = F.embedding(batches_in_scales[i].cluster, batches_in_scales[i].x)
            batches_in_scales[i-1].x = torch.cat([batches_in_scales[i-1].x, higher_scale_x], dim=1)
        return batches_in_scales[0]

    def forward(self, batch):
        batch.x = self.encoder(batch.x)

        batches_in_scales = self.create_multiscale(batch)

        for conv_layer, mlp in zip(self.multiscale_convs[:-1], self.interleave_mlps):
            batches_in_scales = mlp(conv_layer(batches_in_scales))

        batches_in_scales = self.multiscale_convs[-1](batches_in_scales)

        batch = self.concat_scales(batches_in_scales)
        return self.classifier(batch.x)


class MultiscaleGraphConvLayer(torch.nn.Module):
    """
    Applies graph convolution operations at multiple scales.
    
    This layer contains a separate graph convolution for each scale in the
    multiscale representation of the graph.
    """
    def __init__(self, in_channels, out_channels, scales, conv_type, dropout=0.0, residual=False):
        """
        Args:
            in_channels (int): Dimension of input node features
            out_channels (int): Dimension of output node features
            scales (int): Number of scales in the multiscale representation
            conv_type (str): Type of graph convolution to use
                Options: 'gcnconv', 'gatedgcnconv', 'gineconv'
            dropout (float, optional): Dropout probability applied in each layer. Defaults to 0.0.
            residual (bool, optional): Whether to use residual connections in convolutions. 
                Defaults to False.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.dropout = dropout
        self.residual = residual

        conv_model = self.build_conv_model(conv_type)

        self.layers = torch.nn.ModuleList([
            conv_model(in_channels, out_channels, dropout=self.dropout, residual=self.residual)
            for _ in range(self.scales + 1)
        ])

    def build_conv_model(self, model_type):
        if model_type == 'gcnconv':
            return GCNConvLayer
        # elif model_type == 'gatedgcnconv':
        #     return GatedGCNLayer
        # elif model_type == 'gineconv':
        #     return GINEConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batches_in_scales):
        assert len(batches_in_scales) == self.scales + 1, "Number of scales should be equal to the number of batches in scales + 1"

        for i, curr_batch in enumerate(batches_in_scales):
                batches_in_scales[i] = self.layers[i](curr_batch)
        
        return batches_in_scales


class InterleaveMLP(torch.nn.Module):
    """
    Interleave MLP module for facilitating information flow between different scales.
    
    This module allows bidirectional information exchange between adjacent scales:
    1. Bottom-up: Information flows from finer scales to coarser scales
    2. Top-down: Information flows from coarser scales to finer scales
    
    Each scale integrates information from both its adjacent finer scale
    (through move_up_mlps) and its adjacent coarser scale (through move_down_mlps).
    This interleaving of information helps to create scale-aware node representations
    that capture both local and global graph structures.
    """
    def __init__(self, channels, scales, bias=False):
        """
        Args:
            channels (int): Number of feature channels at each scale
            scales (int): Number of scales (excluding the original graph)
            bias (bool, optional): Whether to include bias terms in the linear 
                transformations. Defaults to False.
        """
        super().__init__()
        self.channels = channels
        self.scales = scales
        self.bias = bias
        
        self.fine_to_coarse_mlps = torch.nn.ModuleList([
            pyg_nn.dense.Linear(channels, channels, bias=self.bias)
            for _ in range(self.scales)
        ])
        self.coarse_to_fine_mlps = torch.nn.ModuleList([
            pyg_nn.dense.Linear(channels, channels, bias=self.bias)
            for _ in range(self.scales)
        ])

    def forward(self, batches_in_scales):
        assert len(batches_in_scales) == self.scales + 1, "Number of scales should match the number of batches in scales + 1"

        for i, curr_batch in enumerate(batches_in_scales):
            # for each node q in scale s, we interleave its features with the features of 
            # its fine nodes i and j in scale s-1
            # and coarse node (p,q) in scale s+1.

            # fine nodes
            # 0.5 * (W_{l2h}(x_i) + W_{l2h}(x_j))
            lower_scale_y = (
                self.fine_to_coarse_mlps[i - 1](
                    scatter(
                        batches_in_scales[i - 1].x,
                        curr_batch.cluster,
                        dim=0,
                        reduce='mean'
                    )
                ) if i > 0 else 0
            )

            # coarse node
            # W_{h2l}(x_{(p,q)})
            higher_scale_y = (
                F.embedding(
                    batches_in_scales[i + 1].cluster,
                    self.coarse_to_fine_mlps[i](batches_in_scales[i + 1].x)
                ) if i < self.scales else 0
            )

            curr_batch.x = curr_batch.x + lower_scale_y + higher_scale_y

        return batches_in_scales
