import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv


class GCN2(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_dim, dropout=0.5, alpha=0.1, theta=0.5, layer_num=2):
        super(GCN2, self).__init__()
        self.name = self.__class__.__name__
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.alpha = alpha
        self.theta = theta
        self.layer_num = layer_num

        self.initial_proj = nn.Linear(in_feats, hidden_dim)
        self.convs = nn.ModuleList([
            GCN2Conv(hidden_dim, alpha=alpha, theta=theta, layer=i + 1) for i in range(layer_num)
        ])
        self.classifier = nn.Linear(hidden_dim, out_feats)

    def reset_parameters(self):
        print('GCN2 model parameters reset')
        self.initial_proj.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x0 = F.relu(self.initial_proj(x))

        x_tmp = x0
        for conv in self.convs:
            x_tmp = F.dropout(x_tmp, p=self.dropout, training=self.training)
            x_tmp = F.relu(conv(x_tmp, x0, edge_index))

        embedding = x_tmp
        logits = self.classifier(embedding)
        soft_label = F.softmax(logits, dim=1)
        hard_label = soft_label.argmax(dim=1)

        return logits, {'embedding': embedding, 'soft_label': soft_label, 'hard_label': hard_label}


class GCN2Var(GCN2):
    def __init__(self, in_feats, out_feats, hidden_dim, dropout=0.5,
                 alpha=0.1, theta=0.5, layer_num=2, arch_diff=False):
        if arch_diff:
            super(GCN2, self).__init__()
            self.name = 'GCN2Var_ArchDiff'
            self.in_feats = in_feats
            self.out_feats = out_feats
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.alpha = alpha
            self.theta = theta
            self.layer_num = layer_num + 1

            self.initial_proj = nn.Linear(in_feats, hidden_dim)
            self.convs = nn.ModuleList([
                GCN2Conv(hidden_dim, alpha=alpha, theta=theta, layer=i + 1)
                for i in range(self.layer_num)
            ])
            self.classifier = nn.Linear(hidden_dim, out_feats)
        else:
            super(GCN2Var, self).__init__(in_feats, out_feats, hidden_dim,
                                          dropout, alpha, theta, layer_num)
            self.name = 'GCN2Var'

    def reset_parameters(self):
        print(f'{self.name} parameters reset')
        self.initial_proj.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()