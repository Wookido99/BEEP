import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool

from typing import List, Optional, Union
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.nn.models import JumpingKnowledge
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN


class GCN(torch.nn.Module):
    def __init__(self, in_dim, num_classes, num_layers, hidden, dropout=0.5, pool_type='mean', use_jk=False, jk_mode='cat'):
        super(GCN, self).__init__()
        num_features = in_dim
        self.conv1 = GCNConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        
        self.use_jk = use_jk
        if use_jk:
            self.jk = JumpingKnowledge(jk_mode)

        lin_in_dim = num_layers * hidden if use_jk and jk_mode == 'cat' else hidden
        self.lin1 = Linear(lin_in_dim, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.dropout = dropout

        if pool_type == 'mean':
            self.pool = global_mean_pool
        elif pool_type == 'sum':
            self.pool = global_add_pool
        elif pool_type == 'max':
            self.pool = global_max_pool

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.use_jk:
            self.jk.reset_parameters()
    
    def forward(self, data, edge_weights=None):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            h: graph emb
            x: graph pred
            node_emb: node emb
        """
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        
        if edge_weights is None:
            if 'edge_weights' in data:
                edge_weight = data.edge_weights.float()
            else:
                edge_weight = None
        else: 
            edge_weight = edge_weights.float()
            
        x = F.relu(self.conv1(x, edge_index, edge_weight = edge_weight))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight = edge_weight))
            xs += [x]
        node_emb = x
        if self.use_jk:
            x = self.jk(xs)
        h = self.pool(x, batch)

        x = F.relu(self.lin1(h))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return h, x, node_emb
        # return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
  