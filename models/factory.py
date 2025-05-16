import torch

from models.fagcn import FAGCN
from models.gat import GAT
from models.gcn import GCN
from models.gcn2 import GCN2
from models.graphsage import GraphSAGE


def get_model_by_name(model_name, data, hidden_dim):
    in_feats = data.num_features
    out_feats = data.num_classes

    model_name = model_name.lower()

    if model_name == 'gcn':
        return GCN(in_feats=in_feats, out_feats=out_feats, hidden_dim=hidden_dim)
    elif model_name == 'gat':
        return GAT(in_feats=in_feats, out_feats=out_feats, hidden_dim=hidden_dim)
    elif model_name == 'graphsage':
        return GraphSAGE(in_feats=in_feats, out_feats=out_feats, hidden_dim=hidden_dim)
    elif model_name == 'gcn2':
        return GCN2(in_feats=in_feats, out_feats=out_feats, hidden_dim=hidden_dim)
    elif model_name == 'fagcn':
        return FAGCN(in_feats=in_feats, out_feats=out_feats, hidden_dim=hidden_dim)
    else:
        raise ValueError(f'Unknown model name: {model_name}')


def generate_model_variants(base_model, n_variants=100):
    in_feats = base_model.in_feats
    out_feats = base_model.out_feats
    hidden_dim = base_model.hidden_dim
    dropout = base_model.dropout

    per_model_count = n_variants // 5

    models = []

    for _ in range(per_model_count):
        models.append(GCN(in_feats, out_feats, hidden_dim, dropout))
        # models.append(GCNVar(in_feats, out_feats, hidden_dim, dropout, arch_diff=True))
        models.append(GAT(in_feats, out_feats, hidden_dim, heads=8, dropout=dropout))
        # models.append(GATVar(in_feats, out_feats, hidden_dim, heads=8, dropout=dropout, arch_diff=True))
        models.append(GraphSAGE(in_feats, out_feats, hidden_dim, dropout))
        # models.append(GraphSAGEVar(in_feats, out_feats, hidden_dim, dropout, arch_diff=True))
        models.append(GCN2(in_feats, out_feats, hidden_dim, dropout=dropout))
        # models.append(GCN2Var(in_feats, out_feats, hidden_dim, dropout=dropout, arch_diff=True))
        models.append(FAGCN(in_feats, out_feats, hidden_dim, dropout))
        # models.append(FAGCNVar(in_feats, out_feats, hidden_dim, dropout, arch_diff=True))

    for model in models:
        model.reset_parameters()

    return models


def get_model_hash(model):
    import hashlib
    hash_md5 = hashlib.md5()
    with torch.no_grad():
        for param in model.parameters():
            hash_md5.update(param.detach().cpu().numpy().tobytes())
    return hash_md5.hexdigest()


if __name__ == '__main__':
    from time import time

    in_feats = 1433
    out_feats = 7
    hidden_dim = 64
    dropout = 0.5

    target_model = GCN(in_feats, out_feats, hidden_dim, dropout)
    t0 = time()
    m = generate_model_variants(target_model, n_variants=100)
    t1 = time()
    print('gen time: ', t1 - t0)
    print(len(m))
    print(m[0])
    print(m[-1])
