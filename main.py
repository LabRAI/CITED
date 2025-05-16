from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from models.gcn import GCN
from pipline.cited import CITED
from utils.dataset import CustomDataset

torch.manual_seed(0)
matplotlib.use('Agg')


def viz_emb_with_boundary(embeddings: torch.Tensor, labels: torch.Tensor, boundary_indices: torch.Tensor,
                          mask: Optional[torch.Tensor], save_path: str):
    """
    Visualize node embeddings with t-SNE, highlight boundary nodes.

    Args:
        embeddings (torch.Tensor): Node embeddings, shape [num_nodes, dim].
        labels (torch.Tensor): Node labels, shape [num_nodes].
        boundary_indices (torch.Tensor): Indices of boundary nodes.
    """
    embeddings = embeddings.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    boundary_indices = boundary_indices.cpu().detach().numpy()

    embeddings_2d = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000).fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    num_classes = labels.max().item() + 1

    if mask is None:
        for cls in range(num_classes):
            idx = (labels == cls)
            plt.scatter(
                embeddings_2d[idx, 0],
                embeddings_2d[idx, 1],
                s=10,
                alpha=0.7,
                label=f'Class {cls}'
            )
    else:
        for cls in range(num_classes):
            idx = (labels == cls) & mask.cpu().numpy()
            plt.scatter(
                embeddings_2d[idx, 0],
                embeddings_2d[idx, 1],
                s=10,
                alpha=0.7,
                label=f'Class {cls}'
            )

    # Highlight boundary nodes
    plt.scatter(
        embeddings_2d[boundary_indices, 0],
        embeddings_2d[boundary_indices, 1],
        facecolors='none',
        edgecolors='black',
        s=80,
        linewidths=1.5,
        label='Boundary Nodes'
    )

    plt.legend(markerscale=2, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE of Node Embeddings with Boundary Nodes')
    plt.tight_layout()
    plt.savefig(save_path)


def run_target_pipeline():
    from pipline.target import TargetPipeline
    from models.fagcn import FAGCN
    ds_name = 'cora'
    device = 'cuda:0'
    dataset = CustomDataset(ds_name)
    data = dataset.get()
    dataset.stats()
    # target_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    target_model = FAGCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    target_pipeline = TargetPipeline(target_model, data, device=device)
    target_pipeline.run(3)


def run_gen_cited():
    dataset = CustomDataset('cora')
    data = dataset.get()
    dataset.stats()
    model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    cited = CITED(model, data)
    sig_node_index, sig_area_index, sig_area_threshold = cited.signature(boundary_ratio=0.1, area_ratio=0.5)
    print('node index: ', sig_node_index.shape, 'area index: ', sig_area_index.shape, 'area threshold: ',
          sig_area_threshold)
    train_index = data.train_mask.nonzero(as_tuple=True)[0]
    mask = ~torch.isin(train_index, sig_area_index)
    rest_index = train_index[mask]
    num_query = int(0.5 * len(train_index))
    perm = torch.randperm(len(train_index))[:num_query]
    random_index = train_index[perm]
    query_index = sample_query_index(sig_area_index, rest_index, sig_ratio=0.5, total_ratio=0.5)
    print(
        f'train index: {len(train_index)}, area index: {len(sig_area_index)}, rest index: {len(rest_index)}, random index: {len(random_index)}, designed query: {len(query_index)}')
    _, O = cited.model(cited.data.x, cited.data.edge_index)  # inference once
    viz_emb_with_boundary(O['embedding'], O['hard_label'], sig_node_index, mask=data.train_mask,
                          save_path='cora_rest.png')


def run_attack_pipeline():
    from pipline.attack import GNNStealingPipeline
    dataset = CustomDataset('cora')
    data = dataset.get()
    dataset.stats()
    # init model
    model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    # defense
    cited = CITED(model, data)
    sig_node_index, sig_area_index, sig_area_threshold = cited.signature(boundary_ratio=0.1, area_ratio=0.5)
    print('node index: ', sig_node_index.shape, 'area index: ', sig_area_index.shape, 'area threshold: ',
          sig_area_threshold)
    train_index = data.train_mask.nonzero(as_tuple=True)[0]
    mask = ~torch.isin(train_index, sig_area_index)
    rest_index = train_index[mask]
    num_query = int(0.5 * len(train_index))
    perm = torch.randperm(len(train_index))[:num_query]
    random_index = train_index[perm]
    query_index = sample_query_index(sig_area_index, rest_index, sig_ratio=0.5, total_ratio=0.5)
    print(
        f'train index: {len(train_index)}, area index: {len(sig_area_index)}, rest index: {len(rest_index)}, random index: {len(random_index)}, designed query: {len(query_index)}')
    # attack
    attack_pipeline = GNNStealingPipeline(model, data, level='hard_label')
    attack_pipeline.attack()  # change this to validate boundary importance
    attack_pipeline.inference_surrogate()


def count_class_distribution(index_tensor, labels, num_classes=None):
    """
    index_tensor: Tensor of indices to sample from labels
    labels: Tensor of ground truth labels for all nodes
    num_classes: total number of classes (optional)
    """
    subset_labels = labels[index_tensor]
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1
    count = torch.bincount(subset_labels, minlength=num_classes)
    return count


def sample_query_index(area_index, rest_index, sig_ratio, total_ratio=0.5):
    total_size = area_index.numel() + rest_index.numel()
    query_size = int(total_size * total_ratio)

    area_sample_size = int(query_size * sig_ratio)
    rest_sample_size = query_size - area_sample_size

    area_sample_size = min(area_sample_size, area_index.numel())
    rest_sample_size = min(rest_sample_size, rest_index.numel())

    area_query = area_index[torch.randperm(area_index.numel())[:area_sample_size]]
    rest_query = rest_index[torch.randperm(rest_index.numel())[:rest_sample_size]]

    query_index = torch.cat([area_query, rest_query])
    return query_index


def run_independent():
    from pipline.factory import IndependentFactory
    dataset = CustomDataset('cora')
    data = dataset.get()
    dataset.stats()
    model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    independent_factory = IndependentFactory(model, data, variant_num=2)
    independent_factory.train_independent()


def run_surrogate():
    from pipline.factory import AttackFactory
    dataset = CustomDataset('cora')
    data = dataset.get()
    dataset.stats()
    model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    attack_factory = AttackFactory(model, data, level='hard_label', variant_num=10)
    attack_factory.train_surrogate()


def run_cited():
    from pipline.factory import IndependentDataset
    from pipline.target import TargetPipeline
    from pipline.attack import GNNStealingPipeline
    from pipline.cited import CITEDOVPipeline
    ds_name = 'cora'
    defense_name = 'CITED'
    dataset = CustomDataset(ds_name)
    data = dataset.get()
    dataset.stats()
    # init model
    model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    model_surrogate = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    model_independent = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    # defense
    cited = CITED(model, data)
    cited_data = cited.signature(cited_boundary_ratio=0.1, cited_signature_ratio=0.15)
    cited.finetune_signature(cited_data)
    # train surrogate
    attack_pipeline = GNNStealingPipeline(model, data, defense_name=defense_name, lr=0.001, weight_decay=1e-5,
                                          level='label')
    attack_pipeline.attack(query_ratio=0.5, conf_threshold=0.2)  # change this to validate boundary importance
    # train independent
    dataset = IndependentDataset(ds_name)
    data_independent = dataset.generate(num_class_samples=50, seed=1234)
    independent_pipline = TargetPipeline(model, data_independent)
    independent_pipline.independent_once(model_independent)
    ov = CITEDOVPipeline(model, cited_data, defense_name, [attack_pipeline.surrogate_model], [independent_pipline.independent_model])
    ov.verify(level='label')


def run_cited_factory():
    from pipline.factory import IndependentFactory, AttackFactory
    from pipline.cited import CITEDOVPipeline
    device = 'cuda:2'
    ds_name = 'cora'
    defense_name = 'CITED'
    lr = 0.001
    weight_decay = 1e-5
    variant_num = 5
    query_ratio = 0.5
    threshold = 0.94
    level = 'label'
    cited_boundary_ratio = 0.1
    cited_signature_ratio = 0.5
    dataset = CustomDataset(ds_name)
    data = dataset.get()
    dataset.stats()
    model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    # run cited (just save defense model)
    cited = CITED(model, data, device=device)
    cited_data = cited.signature(cited_boundary_ratio=0.1, cited_signature_ratio=0.15)
    # train independent models
    independent_factory = IndependentFactory(model, ds_name, variant_num=variant_num, device=device)
    independent_factory.train_independent(fixed_seed=42, lr=lr, weight_decay=weight_decay, epochs=200)
    # train surrogate models
    attack_factory = AttackFactory(model, data, defense_name, level=level, variant_num=variant_num,
                                   device=device)  # embedding, hard_label
    attack_factory.train_surrogate(query_ratio=query_ratio, conf_threshold=threshold, lr=lr, weight_decay=weight_decay,
                                   fixed_seed=42)
    # ownership verification
    ov = CITEDOVPipeline(model, cited_data, defense_name, independent_factory.independent_models,
                         attack_factory.surrogate_models, device=device)
    ov.verify(cited_boundary_ratio, cited_signature_ratio, level=level,
              plot_path=f'GCN_{ds_name}_{level}.png')  # embedding, label


def test_randomwm():
    from pipline.defense import RandomWMPipeline
    from pipline.attack import GNNStealingPipeline
    from pipline.independent import IndependentPipeline
    from pipline.verification import WMOVPipeline
    from utils.dataset import IndependentDataset
    ds_name = 'cora'
    defense_name = 'RandomWM'
    device = 'cuda:2'
    dataset = CustomDataset(ds_name)
    data = dataset.get()
    dataset.stats()

    target_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    pipe = RandomWMPipeline(target_model, data, device=device)

    wm_data = pipe.embed_watermark_trigger(num_nodes=10, edge_prob=0.1, p_feat=0.1)
    pipe.finetune_on_watermarked_data(wm_data, epochs=50, lr=0.001, weight_decay=1e-5)
    print('wm data: ', wm_data)
    print('wm mask: ', wm_data.wm_mask.sum())

    defense_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    attack_pipe = GNNStealingPipeline(defense_model, data, defense_name='RandomWM', lr=0.001, weight_decay=1e-5,
                                      level='label')
    attack_pipe.attack(conf_threshold=0.94)
    attack_pipe.inference_surrogate()

    independent_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    independent_ds = IndependentDataset(ds_name)
    independent_data = independent_ds.generate(num_class_samples=50, seed=1234)
    independent_pipe = IndependentPipeline(independent_model, independent_data)
    independent_pipe.independent_once(independent_model)
    independent_model = independent_pipe.independent_model

    wmov = WMOVPipeline(defense_model, wm_data, defense_name, [independent_pipe.independent_model],
                        [attack_pipe.surrogate_model])
    wmov.verify(level='label', plot_path=f'GCN_{ds_name}_{defense_name}.png')


def test_randomwm_factory():
    from pipline.defense import RandomWMPipeline
    from pipline.factory import IndependentFactory, AttackFactory
    from pipline.verification import WMOVPipeline
    ds_name = 'cora'
    defense_name = 'RandomWM'
    lr = 0.001
    weight_decay = 1e-5
    variant_num = 5
    query_ratio = 0.5
    threshold = 0.94
    level = 'label'
    device = 'cpu'

    dataset = CustomDataset(ds_name)
    data = dataset.get()
    dataset.stats()

    # train defense model
    print('[Defense]Start defense')
    target_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    defense_pipe = RandomWMPipeline(target_model, data, device=device)
    wm_data = defense_pipe.embed_watermark_trigger(num_nodes=10, edge_prob=0.1, p_feat=0.1)
    defense_pipe.finetune_on_watermarked_data(wm_data, epochs=50, lr=0.001, weight_decay=1e-5)
    # train independent models
    print('[Independent]train independent models')
    defense_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    independent_factory = IndependentFactory(defense_model, dataset_name=ds_name, variant_num=variant_num,
                                             device=device)
    independent_factory.train_independent(fixed_seed=42, lr=lr, weight_decay=weight_decay, epochs=200)
    # train surrogate models
    print('[Surrogate]train surrogate models')
    surrogate_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    attack_factory = AttackFactory(surrogate_model, data, defense_name, level=level, variant_num=variant_num,
                                   device=device)
    attack_factory.train_surrogate(query_ratio=query_ratio, conf_threshold=threshold, lr=lr, weight_decay=weight_decay,
                                   fixed_seed=42)
    # ownership verification
    print('[Verification]ownership verification')
    ov = WMOVPipeline(target_model, wm_data, defense_name,
                      independent_factory.independent_models,
                      attack_factory.surrogate_models,
                      device=device)
    ov.verify(level=level, plot_path=f'GCN_{ds_name}_{level}_{defense_name}.png')


def test_backdoorwm():
    from pipline.defense import BackdoorWMPipeline
    from pipline.attack import GNNStealingPipeline
    from pipline.independent import IndependentPipeline
    from pipline.verification import WMOVPipeline
    from utils.dataset import IndependentDataset
    ds_name = 'cora'
    conf_threshold = 0.94
    defense_name = 'BackdoorWM'
    device = 'cpu'
    dataset = CustomDataset(ds_name)
    data = dataset.get()
    dataset.stats()

    target_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    defense_pipe = BackdoorWMPipeline(target_model, data, device=device)

    bd_data = defense_pipe.embed_backdoor(backdoor_ratio=0.15, backdoor_len=20)
    defense_pipe.finetune_on_backdoor_data(bd_data, epochs=50, lr=0.001, weight_decay=1e-5)
    print('backdoor data: ', bd_data)
    print('backdoor mask: ', bd_data.wm_mask.sum())

    defense_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    attack_pipe = GNNStealingPipeline(defense_model, data, defense_name, lr=0.001, weight_decay=1e-5,
                                      level='label')
    attack_pipe.attack(query_ratio=0.5, conf_threshold=conf_threshold)
    attack_pipe.inference_surrogate()

    independent_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    independent_ds = IndependentDataset(ds_name)
    independent_data = independent_ds.generate(num_class_samples=50, seed=1234)
    independent_pipe = IndependentPipeline(independent_model, independent_data)
    independent_pipe.independent_once(independent_model)
    independent_model = independent_pipe.independent_model

    wmov = WMOVPipeline(defense_model, bd_data, defense_name, [independent_pipe.independent_model],
                        [attack_pipe.surrogate_model])
    wmov.verify(level='label', plot_path=f'GCN_{ds_name}_{defense_name}.png')


def test_backdoorwm_factory():
    from pipline.defense import BackdoorWMPipeline
    from pipline.factory import IndependentFactory, AttackFactory
    from pipline.verification import WMOVPipeline
    ds_name = 'citeseer'
    defense_name = 'BackdoorWM'
    lr = 0.001
    weight_decay = 1e-5
    variant_num = 10
    query_ratio = 0.5
    threshold = 0.7
    level = 'label'
    device = 'cuda:0'

    dataset = CustomDataset(ds_name)
    data = dataset.get()
    dataset.stats()

    # train defense model
    print('[Defense]Start defense')
    target_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    defense_pipe = BackdoorWMPipeline(target_model, data, device=device)
    wm_data = defense_pipe.embed_backdoor(backdoor_ratio=0.1, backdoor_len=20)
    defense_pipe.finetune_on_backdoor_data(wm_data, epochs=50, lr=0.001, weight_decay=1e-5)
    # train independent models
    print('[Independent]train independent models')
    defense_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    independent_factory = IndependentFactory(defense_model, dataset_name=ds_name, variant_num=variant_num,
                                             device=device)
    independent_factory.train_independent(fixed_seed=42, lr=lr, weight_decay=weight_decay, epochs=200)
    # train surrogate models
    print('[Surrogate]train surrogate models')
    surrogate_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    attack_factory = AttackFactory(surrogate_model, data, defense_name, level=level, variant_num=variant_num,
                                   device=device)
    attack_factory.train_surrogate(query_ratio=query_ratio, conf_threshold=threshold, lr=lr, weight_decay=weight_decay,
                                   fixed_seed=42)
    # ownership verification
    print('[Verification]ownership verification')
    ov = WMOVPipeline(target_model, wm_data, defense_name,
                      independent_factory.independent_models,
                      attack_factory.surrogate_models,
                      device=device)
    ov.verify(level=level, plot_path=f'GCN_{ds_name}_{level}_{defense_name}.png')


def test_survivewm():
    from pipline.defense import SurviveWMPipeline
    from pipline.attack import GNNStealingPipeline
    from pipline.independent import IndependentPipeline
    from pipline.verification import WMOVPipeline
    from utils.dataset import IndependentDataset
    ds_name = 'cora'
    defense_name = 'SurviveWM'
    device = 'cpu'
    dataset = CustomDataset(ds_name)
    data = dataset.get()
    dataset.stats()

    target_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    defense_pipe = SurviveWMPipeline(target_model, data, device=device)

    wm_data = defense_pipe.embed_wm(wm_nodes=10, edge_prob=0.3)
    defense_pipe.finetune_on_wm_data(wm_data, epochs=50, lr=0.001, weight_decay=1e-5)
    print('backdoor data: ', wm_data)
    print('backdoor mask: ', wm_data.wm_mask.sum())

    defense_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    attack_pipe = GNNStealingPipeline(defense_model, data, defense_name, lr=0.001, weight_decay=1e-5,
                                      level='label')
    attack_pipe.attack(conf_threshold=0.94)
    attack_pipe.inference_surrogate()

    independent_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    independent_ds = IndependentDataset(ds_name)
    independent_data = independent_ds.generate(num_class_samples=50, seed=1234)
    independent_pipe = IndependentPipeline(independent_model, independent_data)
    independent_pipe.independent_once(independent_model)
    independent_model = independent_pipe.independent_model

    wmov = WMOVPipeline(defense_model, wm_data, defense_name, [independent_pipe.independent_model],
                        [attack_pipe.surrogate_model])
    wmov.verify(level='label', plot_path=f'GCN_{ds_name}_{defense_name}.png')


def test_survivewm_factory():
    from pipline.defense import SurviveWMPipeline
    from pipline.factory import IndependentFactory, AttackFactory
    from pipline.verification import WMOVPipeline
    ds_name = 'citeseer'
    defense_name = 'SurviveWM'
    lr = 0.001
    weight_decay = 1e-5
    variant_num = 10
    threshold = 0.7
    query_ratio = 0.5
    level = 'label'
    device = 'cuda:2'

    dataset = CustomDataset(ds_name)
    data = dataset.get()
    dataset.stats()

    # train defense model
    print('[Defense]Start defense')
    target_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    defense_pipe = SurviveWMPipeline(target_model, data, device=device)
    wm_data = defense_pipe.embed_wm(wm_nodes=10, edge_prob=0.3)
    defense_pipe.finetune_on_wm_data(wm_data, epochs=50, lr=0.001, weight_decay=1e-5)
    # train independent models
    print('[Independent]train independent models')
    defense_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    independent_factory = IndependentFactory(defense_model, dataset_name=ds_name, variant_num=variant_num,
                                             device=device)
    independent_factory.train_independent(fixed_seed=42, lr=lr, weight_decay=weight_decay, epochs=200)
    # train surrogate models
    print('[Surrogate]train surrogate models')
    surrogate_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    attack_factory = AttackFactory(surrogate_model, data, defense_name, level=level, variant_num=variant_num,
                                   device=device)
    attack_factory.train_surrogate(query_ratio=query_ratio, conf_threshold=threshold, lr=lr, weight_decay=weight_decay,
                                   fixed_seed=42)
    # ownership verification
    print('[Verification]ownership verification')
    ov = WMOVPipeline(target_model, wm_data, defense_name,
                      independent_factory.independent_models,
                      attack_factory.surrogate_models,
                      device=device)
    ov.verify(level=level, plot_path=f'GCN_{ds_name}_{level}_{defense_name}.png')


def test_defense_model():
    import os
    ds_name = 'cora'
    dataset = CustomDataset(ds_name)
    data = dataset.get()
    dataset.stats()

    model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    target_path = './output/defense'
    model_path = os.path.join(target_path, f'{data.name}_{model.name}.pth')
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    model(data.x, data.edge_index)
    logits, O = model(data.x, data.edge_index)
    pred = logits[data.test_mask].argmax(dim=1).cpu()
    y = data.y[data.test_mask]
    acc = (pred == y).sum() / data.test_mask.sum()
    print(acc)


if __name__ == "__main__":
    # run_target_pipeline()
    # run_gen_cited()
    # run_attack_pipeline()
    # run_ov()
    # run_once()
    # test_backdoorwm()
    run_cited()
