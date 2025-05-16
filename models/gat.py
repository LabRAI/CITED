import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.name = self.__class__.__name__
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gat1 = GATConv(in_feats, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, out_feats)

    def reset_parameters(self):
        print('GAT parameters reset')
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        embedding = x

        logits = self.classifier(embedding)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}


class GATVar(GAT):
    def __init__(self, in_feats, out_feats, hidden_dim, heads=8, dropout=0.6, arch_diff=False):
        if arch_diff:
            super(GAT, self).__init__()
            self.in_feats = in_feats
            self.out_feats = out_feats
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.gat1 = GATConv(in_feats, hidden_dim, heads=heads, dropout=dropout)
            self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=4, dropout=dropout)
            self.gat3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=dropout)
            self.classifier = nn.Linear(hidden_dim, out_feats)
            self.name = 'GATVar_ArchDiff'
        else:
            super().__init__(in_feats, out_feats, hidden_dim, heads, dropout)
            self.name = 'GATVar'

    def reset_parameters(self):
        print(f'{self.name} parameters reset')
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()
        self.classifier.reset_parameters()
        if hasattr(self, 'gat3'):
            self.gat3.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        if hasattr(self, 'gat3'):
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.gat3(x, edge_index)
        embedding = x
        logits = self.classifier(x)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}
