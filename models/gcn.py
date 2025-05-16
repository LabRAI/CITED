import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.name = self.__class__.__name__
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gcn1 = GCNConv(in_feats, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_feats)

    def reset_parameters(self):
        print('GCN model parameters reset')
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)

        embedding = x

        logits = self.classifier(embedding)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}


class GCNVar(GCN):
    def __init__(self, in_feats, out_feats, hidden_dim, dropout=0.5, arch_diff=False):
        if arch_diff:
            super(GCN, self).__init__()
            self.name = 'GCNVar_ArchDiff'
            self.dropout = dropout
            self.gcn1 = GCNConv(in_feats, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
            self.gcn3 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, out_feats)
        else:
            super(GCNVar, self).__init__(in_feats, out_feats, hidden_dim, dropout)
            self.name = 'GCNVar'

    def reset_parameters(self):
        print(f'{self.name} parameters reset')
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        if hasattr(self, 'gcn3'):
            self.gcn3.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        if hasattr(self, 'gcn3'):
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.gcn3(x, edge_index)

        embedding = x
        logits = self.classifier(embedding)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}
