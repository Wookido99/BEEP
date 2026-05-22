import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import BatchNorm

class GuidedExplainer(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout=0.2):
        super(GuidedExplainer, self).__init__()
        
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels) 
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        h1 = F.relu(self.gcn1(x, edge_index))
        h2 = F.relu(self.gcn2(h1, edge_index))
        h_cat = torch.cat([h1, h2], dim=-1)
        
        src, dst = edge_index
        edge_feat = torch.cat([h_cat[src], h_cat[dst]], dim=-1)

        logits = self.edge_mlp(edge_feat)
        prob = F.softmax(logits, dim=-1)[:,1]
        
        return prob


class GuidedExplainerGIN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout=0.2):
        super(GuidedExplainerGIN, self).__init__()

        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gin1 = GINConv(mlp1)

        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gin2 = GINConv(mlp2)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels) 
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h1 = F.relu(self.gin1(x, edge_index))
        h2 = F.relu(self.gin2(h1, edge_index))
        h_cat = torch.cat([h1, h2], dim=-1)

        src, dst = edge_index
        edge_feat = torch.cat([h_cat[src], h_cat[dst]], dim=-1)

        logits = self.edge_mlp(edge_feat)
        prob = F.softmax(logits, dim=-1)[:,1]

        return prob