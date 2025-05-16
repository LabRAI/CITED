import numpy as np

from models.factory import get_model_by_name
from pipline.cited import CITED, CITEDOVPipeline
from pipline.defense import RandomWMPipeline, BackdoorWMPipeline, SurviveWMPipeline
from pipline.factory import AttackFactory, IndependentFactory
from pipline.verification import WMOVPipeline
from utils.dataset import CustomDataset


def load_threshold(model_name, ds_name, level):
    results_path = f'./results/Res_CITED_{model_name}_{ds_name}_{level}.npz'
    print(f'[Loaded] Result loaded from: {results_path}')
    data = np.load(results_path, allow_pickle=True)
    threshold = data['threshold']  # [n_trial]
    return threshold, np.mean(threshold)


def run_once_randomwm(config, th_mean, trial_id=0):
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

    print('[Independent] Train independent models')
    defense_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    independent_factory = IndependentFactory(defense_model, dataset_name=config['ds_name'],
                                             variant_num=config['variant_num'], device=config['device'])
    independent_factory.train_independent(fixed_seed=config['fixed_seed'], lr=config['lr'],
                                          weight_decay=config['weight_decay'], epochs=config['train_epochs'])

    print('[Surrogate] Train surrogate models')
    surrogate_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    attack_factory = AttackFactory(surrogate_model, data, config['defense_name'], level=config['level'],
                                   variant_num=config['variant_num'], device=config['device'])
    attack_factory.train_surrogate(query_ratio=config['query_ratio'], conf_threshold=config['threshold'],
                                   lr=config['lr'],
                                   weight_decay=config['weight_decay'], fixed_seed=config['fixed_seed'])

    print('[Verification] Ownership verification')
    ov = WMOVPipeline(target_model, wm_data, config['defense_name'],
                      independent_factory.independent_models,
                      attack_factory.surrogate_models,
                      device=config['device'])
    # calc acc
    acc = ov.accuracy(th_mean)
    return acc


def run_once_backdoorwm(config, th_mean, trial_id=0):
    print(f'\n========== Trial {trial_id + 1} ==========')

    dataset = CustomDataset(config['ds_name'])
    data = dataset.get()
    dataset.stats()

    print('[Defense] Start defense')
    target_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    defense_pipe = BackdoorWMPipeline(target_model, data, device=config['device'])
    wm_data = defense_pipe.embed_backdoor(backdoor_ratio=config['backdoor_ratio'], backdoor_len=config['backdoor_len'])
    defense_pipe.finetune_on_backdoor_data(wm_data, epochs=config['finetune_epochs'], lr=config['lr'],
                                           weight_decay=config['weight_decay'])

    print('[Independent] Train independent models')
    defense_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    independent_factory = IndependentFactory(defense_model, dataset_name=config['ds_name'],
                                             variant_num=config['variant_num'], device=config['device'])
    independent_factory.train_independent(fixed_seed=config['fixed_seed'], lr=config['lr'],
                                          weight_decay=config['weight_decay'], epochs=config['train_epochs'])

    print('[Surrogate] Train surrogate models')
    surrogate_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    attack_factory = AttackFactory(surrogate_model, data, config['defense_name'], level=config['level'],
                                   variant_num=config['variant_num'], device=config['device'])
    attack_factory.train_surrogate(query_ratio=config['query_ratio'], conf_threshold=config['threshold'],
                                   lr=config['lr'],
                                   weight_decay=config['weight_decay'], fixed_seed=config['fixed_seed'])

    print('[Verification] Ownership verification')
    ov = WMOVPipeline(target_model, wm_data, config['defense_name'],
                      independent_factory.independent_models,
                      attack_factory.surrogate_models,
                      device=config['device'])
    # calc acc
    acc = ov.accuracy(th_mean)
    return acc


def run_once_survivewm(config, th_mean, trial_id=0):
    print(f'\n========== Trial {trial_id + 1} ==========')

    dataset = CustomDataset(config['ds_name'])
    data = dataset.get()
    dataset.stats()

    print('[Defense] Start defense')
    target_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    defense_pipe = SurviveWMPipeline(target_model, data, device=config['device'])
    wm_data = defense_pipe.embed_wm(survive_node_num=config['survive_node_num'],
                                    survive_edge_prob=config['survive_edge_prob'], )
    defense_pipe.finetune_on_wm_data(wm_data, epochs=config['finetune_epochs'], lr=config['lr'],
                                     weight_decay=config['weight_decay'])

    print('[Independent] Train independent models')
    defense_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    independent_factory = IndependentFactory(defense_model, dataset_name=config['ds_name'],
                                             variant_num=config['variant_num'], device=config['device'])
    independent_factory.train_independent(fixed_seed=config['fixed_seed'], lr=config['lr'],
                                          weight_decay=config['weight_decay'], epochs=config['train_epochs'])

    print('[Surrogate] Train surrogate models')
    surrogate_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    attack_factory = AttackFactory(surrogate_model, data, config['defense_name'], level=config['level'],
                                   variant_num=config['variant_num'], device=config['device'])
    attack_factory.train_surrogate(query_ratio=config['query_ratio'], conf_threshold=config['threshold'],
                                   lr=config['lr'],
                                   weight_decay=config['weight_decay'], fixed_seed=config['fixed_seed'])

    print('[Verification] Ownership verification')
    ov = WMOVPipeline(target_model, wm_data, config['defense_name'],
                      independent_factory.independent_models,
                      attack_factory.surrogate_models,
                      device=config['device'])
    # calc acc
    acc = ov.accuracy(th_mean)
    return acc


def run_once_cited(config, th_mean, trial_id=0):
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

    print('[Independent] Train independent models')
    defense_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    independent_factory = IndependentFactory(defense_model, dataset_name=config['ds_name'],
                                             variant_num=config['variant_num'], device=config['device'])
    independent_factory.train_independent(fixed_seed=config['fixed_seed'], lr=config['lr'],
                                          weight_decay=config['weight_decay'], epochs=config['train_epochs'])

    print('[Surrogate] Train surrogate models')
    surrogate_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    attack_factory = AttackFactory(surrogate_model, data, config['defense_name'], level=config['level'],
                                   variant_num=config['variant_num'], device=config['device'])
    attack_factory.train_surrogate(query_ratio=config['query_ratio'], conf_threshold=config['threshold'],
                                   lr=config['lr'],
                                   weight_decay=config['weight_decay'], fixed_seed=config['fixed_seed'])

    print('[Verification] Ownership verification')
    ov = CITEDOVPipeline(target_model, wm_data, config['defense_name'],
                         independent_factory.independent_models,
                         attack_factory.surrogate_models,
                         device=config['device'])
    # calc acc
    acc = ov.accuracy(th_mean)
    return acc


def run_trials(config, th_list, th_mean, trial_num):
    dispatcher = {
        'SurviveWM': run_once_survivewm,
        'RandomWM': run_once_randomwm,
        'BackdoorWM': run_once_backdoorwm,
        'CITED': run_once_cited,
    }

    acc_list = []

    for trial in range(trial_num):
        acc = dispatcher[config['defense_name']](config, th_mean, trial_id=trial)
        acc_list.append(acc)

    save_path = f'./results/Res_exp3_{config["defense_name"]}_{config["model_name"]}_{config["ds_name"]}_{config["level"]}.npz'
    np.savez(
        save_path,
        acc_list=acc_list,
        th_list=th_list,
        config=np.array([config], dtype=object)
    )
    print(f'[Saved] Result saved to: {save_path}')

    print(f'Final results: {acc_list}')
    print(f'Final results: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}')


if __name__ == '__main__':
    import argparse
    from utils.config import build_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--defense', type=str, required=True, help='Defense strategy name')
    parser.add_argument('--data', type=str, required=True, help='Dataset name')
    parser.add_argument('--device', type=str, default='0', help='GPU device id (e.g., 0, 1, 2)')

    args = parser.parse_args()

    base_config = {
        # exp setting
        'level': 'label',
        'variant_num': 15,
        # model setting
        'model_name': 'gcn',
        'hidden_dim': 128,
        # train setting
        'train_epochs': 200,
        'finetune_epochs': 50,
        'lr': 0.001,
        'weight_decay': 1e-5,
        # attack setting
        'query_ratio': 0.5,
        # misc
        'fixed_seed': 42,
        'device': f'cuda:{args.device}',
    }

    cfg = build_config(base_config, args.defense, args.data)

    # load threshold
    th, th_mean = load_threshold(cfg['model_name'], cfg['ds_name'], cfg['level'])

    # set slack
    th_mean = 0.5 * th_mean

    results = run_trials(cfg, th, th_mean, trial_num=3)
