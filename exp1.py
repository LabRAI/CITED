from models.factory import get_model_by_name
from pipline.cited import CITED
from pipline.defense import RandomWMPipeline, BackdoorWMPipeline, SurviveWMPipeline
from utils.dataset import CustomDataset, OriginDataset


def run_once_randomwm(config, trial_id=0):
    print(f'\n========== Trial {trial_id + 1} ==========')

    dataset = CustomDataset(config['ds_name'])
    data = dataset.get()
    dataset.stats()

    print('[Defense] Start defense')
    target_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    defense_pipe = RandomWMPipeline(target_model, data, device=config['device'])
    wm_data = defense_pipe.embed_watermark_trigger(random_node_num=config['random_node_num'],
                                                   random_edge_prob=config['random_edge_prob'],
                                                   random_feat_ratio=config['random_feat_ratio'])
    defense_pipe.finetune_on_watermarked_data(wm_data, epochs=config['finetune_epochs'], lr=config['lr'],
                                              weight_decay=config['weight_decay'])


def run_once_backdoorwm(config, trial_id=0):
    print(f'\n========== Trial {trial_id + 1} ==========')

    dataset = CustomDataset(config['ds_name'])
    data = dataset.get()
    dataset.stats()

    print('[Defense] Start defense')
    target_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    defense_pipe = BackdoorWMPipeline(target_model, data, device=config['device'])
    wm_data = defense_pipe.embed_backdoor(backdoor_ratio=config['backdoor_ratio'], backdoor_len=config['backdoor_len'])
    print('wm num: ', wm_data.wm_mask.sum(), 'train num: ', wm_data.train_mask.sum())
    defense_pipe.finetune_on_backdoor_data(wm_data, epochs=config['finetune_epochs'], lr=config['lr'],
                                           weight_decay=config['weight_decay'])


def run_once_surviveWM(config, trial_id=0):
    print(f'\n========== Trial {trial_id + 1} ==========')

    dataset = CustomDataset(config['ds_name'])
    data = dataset.get()
    dataset.stats()

    print('[Defense] Start defense')
    target_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    defense_pipe = SurviveWMPipeline(target_model, data, device=config['device'])
    wm_data = defense_pipe.embed_wm(survive_node_num=config['survive_node_num'],
                                    survive_edge_prob=config['survive_edge_prob'], )
    print('wm num: ', wm_data.wm_mask.sum(), 'train num: ', wm_data.train_mask.sum())
    defense_pipe.finetune_on_wm_data(wm_data, epochs=config['finetune_epochs'], lr=config['lr'],
                                     weight_decay=config['weight_decay'])


def run_once_cited(config, trial_id=0):
    print(f'\n========== Trial {trial_id + 1} ==========')

    dataset = CustomDataset(config['ds_name'])
    data = dataset.get()
    dataset.stats()

    print('[Defense] Start defense')
    target_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    defense_pipe = CITED(target_model, data, device=config['device'])
    wm_data = defense_pipe.signature(cited_boundary_ratio=config['cited_boundary_ratio'],
                                     cited_signature_ratio=config['cited_signature_ratio'])
    defense_pipe.finetune_signature(wm_data, epochs=config['finetune_epochs'], lr=config['lr'],
                                    weight_decay=config['weight_decay'])


def run_target_pipeline(model_name, ds_name):
    from pipline.target import TargetPipeline
    from models.fagcn import FAGCN
    from models.gat import GAT
    from models.factory import get_model_by_name
    device = 'cuda:0'
    dataset = OriginDataset(ds_name)
    data = dataset.get()
    dataset.stats()
    # target_model = GCN(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    target_model = get_model_by_name(model_name, data, hidden_dim=128)
    target_model = GAT(in_feats=data.num_features, out_feats=data.num_classes, hidden_dim=128)
    target_pipeline = TargetPipeline(target_model, data, device=device, lr=0.001, weight_decay=1e-5)
    target_pipeline.run(3)


if __name__ == '__main__':
    config = {
        # exp setting
        'level': 'label',
        'variant_num': 15,
        # model setting
        'model_name': 'gat',
        'hidden_dim': 128,
        # train setting
        'train_epochs': 200,
        'lr': 0.001,
        'weight_decay': 1e-5,
        # attack setting
        'query_ratio': 0.5,
        # misc
        'fixed_seed': 42,
        'device': 'cuda:0',
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        'ds_name': 'pubmed',
        'finetune_epochs': 10,
        # Defense RandomWM
        'random_node_num': 100,
        'random_edge_prob': 0.5,
        'random_feat_ratio': 0.1,
        # Defense BackdoorWM
        'backdoor_ratio': 0.2,
        'backdoor_len': 40,  # 20-40
        # Defense SurviveWM
        'survive_node_num': 300,
        'survive_edge_prob': 0.5,  # 0.5
        # Defense CITED
        'cited_boundary_ratio': 0.1,
        'cited_signature_ratio': 0.5,
    }

    run_once_randomwm(config)
    # run_once_backdoorwm(config)
    # run_once_surviveWM(config)
    # run_once_cited(config)

    model_name = 'gat'
    ds_name = 'pubmed'
    # run_target_pipeline(model_name, ds_name)
