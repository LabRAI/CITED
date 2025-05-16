import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.name = self.__class__.__name__
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.sage1 = SAGEConv(in_feats, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_feats)

    def reset_parameters(self):
        print('GraphSAGE model parameters reset')
        self.sage1.reset_parameters()
        self.sage2.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)

        embedding = x
        logits = self.classifier(embedding)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}


class GraphSAGEVar(GraphSAGE):
    def __init__(self, in_feats, out_feats, hidden_dim, dropout=0.5, arch_diff=False):
        if arch_diff:
            super(GraphSAGE, self).__init__()
            self.name = 'GraphSAGEVar_ArchDiff'
            self.dropout = dropout
            self.sage1 = SAGEConv(in_feats, hidden_dim)
            self.sage2 = SAGEConv(hidden_dim, hidden_dim)
            self.sage3 = SAGEConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, out_feats)
        else:
            super(GraphSAGEVar, self).__init__(in_feats, out_feats, hidden_dim, dropout)
            self.name = 'GraphSAGEVar'

    def reset_parameters(self):
        print(f'{self.name} parameters reset')
        self.sage1.reset_parameters()
        self.sage2.reset_parameters()
        if hasattr(self, 'sage3'):
            self.sage3.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        if hasattr(self, 'sage3'):
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.sage3(x, edge_index)

        embedding = x
        logits = self.classifier(embedding)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}
