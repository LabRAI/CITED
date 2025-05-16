import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FAConv


class FAGCN(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim, dropout=0.5):
        super(FAGCN, self).__init__()
        self.name = self.__class__.__name__
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.input_proj = nn.Linear(in_feats, hidden_dim)
        self.faconv1 = FAConv(hidden_dim)
        self.faconv2 = FAConv(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_feats)

    def reset_parameters(self):
        print('FAGCN model parameters reset')
        self.input_proj.reset_parameters()
        self.faconv1.reset_parameters()
        self.faconv2.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x_0 = x.clone()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.faconv1(x, x_0, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.faconv2(x, x_0, edge_index)

        embedding = x
        logits = self.classifier(embedding)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}


class FAGCNVar(FAGCN):
    def __init__(self, in_feats, out_feats, hidden_dim, dropout=0.5, arch_diff=False):
        if arch_diff:
            super(FAGCN, self).__init__()
            self.name = 'FAGCNVar_ArchDiff'
            self.dropout = dropout
            self.input_proj = nn.Linear(in_feats, hidden_dim)
            self.faconv1 = FAConv(hidden_dim)
            self.faconv2 = FAConv(hidden_dim)
            self.faconv3 = FAConv(hidden_dim)
            self.classifier = nn.Linear(hidden_dim, out_feats)
        else:
            super(FAGCNVar, self).__init__(in_feats, out_feats, hidden_dim, dropout)
            self.name = 'FAGCNVar'

    def reset_parameters(self):
        print(f'{self.name} parameters reset')
        self.input_proj.reset_parameters()
        self.faconv1.reset_parameters()
        self.faconv2.reset_parameters()
        if hasattr(self, 'faconv3'):
            self.faconv3.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x = self.input_proj(x)        # [N, in_feats] -> [N, hidden_dim]
        x_0 = x.clone()

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.faconv1(x, x_0, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.faconv2(x, x_0, edge_index)

        if hasattr(self, 'faconv3'):
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.faconv3(x, x_0, edge_index)

        embedding = x
        logits = self.classifier(embedding)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}
