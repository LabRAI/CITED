# config.py

defense_configs = {
    'SurviveWM': {
        'defense_name': 'SurviveWM',
        'survive_node_num': 10,
        'survive_edge_prob': 0.5,  # 0.5
        'finetune_epochs': 10,  # need change
    },
    'RandomWM': {
        'defense_name': 'RandomWM',
        'random_node_num': 10,
        'random_edge_prob': 0.3,  # 0.1-0.3
        'random_feat_ratio': 0.1,
        'finetune_epochs': 10,  # need change
    },
    'BackdoorWM': {
        'defense_name': 'BackdoorWM',
        'backdoor_ratio': 0.05,
        'backdoor_len': 20,  # 20-40
        'finetune_epochs': 10,  # need change
    },
    'CITED': {
        'defense_name': 'CITED',
        'cited_boundary_ratio': 0.05,
        'cited_signature_ratio': 0.1,
        'finetune_epochs': 50,  # need change
    }
}

data_configs = {
    'cora': {
        'ds_name': 'cora',
        'threshold': 0.2,
    },
    'citeseer': {
        'ds_name': 'citeseer',
        'threshold': 0.2,
    },
    'pubmed': {
        'ds_name': 'pubmed',
        'threshold': 0.2,
    },
    'amazon-photo': {
        'ds_name': 'amazon-photo',
        'threshold': 0.2,
    },
    'amazon-computers': {
        'ds_name': 'amazon-computers',
        'threshold': 0.2,
    },
    'coauthor-cs': {
        'ds_name': 'coauthor-cs',
        'threshold': 0.2,
    },
    'coauthor-physics': {
        'ds_name': 'coauthor-physics',
        'threshold': 0.2,
    }
}

default_config = {
    # exp setting
    'level': 'label',
    'variant_num': 5,
    # model setting
    'model_name': 'gcn',
    'hidden_dim': 128,
    # train setting
    'train_epochs': 200,
    'finetune_epochs': 50,
    'lr': 0.001,
    'weight_decay': 1e-5,
    # misc
    'fixed_seed': 42,
    'device': 'cuda:2',
}


def build_config(base_config, defense_name, data_name):
    defense_config = defense_configs.get(defense_name)
    data_config = data_configs.get(data_name)

    if defense_config is None or data_config is None:
        raise ValueError(f"Invalid defense or data name: {defense_name}, {data_name}")

    base_config.update(defense_config)
    base_config.update(data_config)

    return base_config
