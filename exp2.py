import numpy as np

from models.factory import get_model_by_name
from pipline.cited import CITED, CITEDOVPipeline
from pipline.defense import RandomWMPipeline, BackdoorWMPipeline, SurviveWMPipeline
from pipline.factory import IndependentFactory, AttackFactory
from pipline.verification import WMOVPipeline
from utils.dataset import CustomDataset


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

    aruc, R, U, asr, threshold = ov.verify(level=config['level'], plot_path=None)

    print(f'[Result - Trial {trial_id + 1}] ARUC = {aruc:.4f}, ASR = {asr:.4f}')
    return {
        'aruc': aruc,
        'asr': asr,
        'R': R,
        'U': U,
        'threshold': threshold
    }


def run_once_backdoorwm(config, trial_id=0):
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

    aruc, R, U, asr, threshold = ov.verify(level=config['level'], plot_path=None)

    print(f'[Result - Trial {trial_id + 1}] ARUC = {aruc:.4f}, ASR = {asr:.4f}')
    return {
        'aruc': aruc,
        'asr': asr,
        'R': R,
        'U': U,
        'threshold': threshold
    }


def run_once_survivewm(config, trial_id=0):
    print(f'\n========== Trial {trial_id + 1} ==========')

    dataset = CustomDataset(config['ds_name'])
    data = dataset.get()
    dataset.stats()

    print('[Defense] Start defense')
    target_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    defense_pipe = SurviveWMPipeline(target_model, data, device=config['device'])
    wm_data = defense_pipe.embed_wm(survive_node_num=config['survive_node_num'],
                                    survive_edge_prob=config['survive_edge_prob'])
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

    aruc, R, U, asr, threshold = ov.verify(level=config['level'], plot_path=None)

    print(f'[Result - Trial {trial_id + 1}] ARUC = {aruc:.4f}, ASR = {asr:.4f}')
    return {
        'aruc': aruc,
        'asr': asr,
        'R': R,
        'U': U,
        'threshold': threshold
    }


def run_once_cited(config, trial_id=0):
    print(f'\n========== Trial {trial_id + 1} ==========')

    dataset = CustomDataset(config['ds_name'])
    data = dataset.get()
    dataset.stats()

    print('[Defense] Start CITED defense')
    target_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    cited = CITED(target_model, data, device=config['device'])
    cited_data = cited.signature(cited_boundary_ratio=config['cited_boundary_ratio'],
                                 cited_signature_ratio=config['cited_signature_ratio'])
    cited.finetune_signature(cited_data, epochs=config['finetune_epochs'], lr=config['lr'],
                             weight_decay=config['weight_decay'])

    print('[Independent] Train independent models')
    defense_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    independent_factory = IndependentFactory(defense_model, config['ds_name'],
                                             variant_num=config['variant_num'], device=config['device'])
    independent_factory.train_independent(fixed_seed=config['fixed_seed'] + trial_id, lr=config['lr'],
                                          weight_decay=config['weight_decay'], epochs=config['train_epochs'])

    print('[Surrogate] Train surrogate models')
    surrogate_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    attack_factory = AttackFactory(surrogate_model, data, config['defense_name'], level=config['level'],
                                   variant_num=config['variant_num'], device=config['device'])
    attack_factory.train_surrogate(query_ratio=config['query_ratio'], conf_threshold=config['threshold'],
                                   lr=config['lr'],
                                   weight_decay=config['weight_decay'], fixed_seed=config['fixed_seed'] + trial_id)

    print('[Verification] Ownership verification')
    cited_data.wm_mask = cited_data.signature_mask

    ov = CITEDOVPipeline(target_model, cited_data, config['defense_name'],
                         independent_factory.independent_models,
                         attack_factory.surrogate_models,
                         device=config['device'])

    aruc, R, U, asr, threshold = ov.verify(level=config['level'], plot_path=None)

    print(f'[Result - Trial {trial_id + 1}] ARUC = {aruc:.4f}, ASR = {asr:.4f}')
    return {
        'aruc': aruc,
        'asr': asr,
        'R': R,
        'U': U,
        'threshold': threshold
    }


def run_trials(config, trial_num):
    aruc_list = []
    asr_list = []
    R_list = []
    U_list = []
    thre_list = []

    dispatcher = {
        'SurviveWM': run_once_survivewm,
        'RandomWM': run_once_randomwm,
        'BackdoorWM': run_once_backdoorwm,
        'CITED': run_once_cited,
    }

    for trial in range(trial_num):
        result = dispatcher[config['defense_name']](config, trial_id=trial)
        aruc_list.append(result['aruc'])
        asr_list.append(result['asr'])
        R_list.append(result['R'])
        U_list.append(result['U'])
        thre_list.append(result['threshold'])

    aruc_arr = np.array(aruc_list)
    asr_arr = np.array(asr_list)
    R_arr = np.array(R_list)  # shape [n_trial, 100]
    U_arr = np.array(U_list)  # shape [n_trial, 100]
    thre_list = np.array(thre_list)

    aruc_mean, aruc_std = aruc_arr.mean(), aruc_arr.std()
    asr_mean, asr_std = asr_arr.mean(), asr_arr.std()

    print_config_inline(config)

    print('\n========== Summary ==========')
    print(f'ARUC: {aruc_mean:.4f} ± {aruc_std:.4f}')
    print(f'ASR : {asr_mean:.4f} ± {asr_std:.4f}')

    save_path = f'./results/Res_{config["defense_name"]}_{config["model_name"]}_{config["ds_name"]}_{config["level"]}.npz'
    np.savez(
        save_path,
        aruc=aruc_arr,
        asr=asr_arr,
        R=R_arr,
        U=U_arr,
        threshold=thre_list,
        aruc_mean=aruc_mean,
        aruc_std=aruc_std,
        asr_mean=asr_mean,
        asr_std=asr_std,
        config=np.array([config], dtype=object)
    )
    print(f'[Saved] Result saved to: {save_path}')


def print_config_inline(config):
    print("Experiment Configuration: ", end="")
    print(" | ".join(f"{key}={value}" for key, value in config.items()))


if __name__ == '__main__':
    import argparse
    from utils.config import build_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--defense', type=str, required=True, help='Defense strategy name')
    parser.add_argument('--data', type=str, required=True, help='Dataset name')
    parser.add_argument('--trial_num', type=int, default=3)
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
        'lr': 0.001,
        'weight_decay': 1e-5,
        # attack setting
        'query_ratio': 0.5,
        # misc
        'fixed_seed': 42,
        'device': f'cuda:{args.device}',
    }

    cfg = build_config(base_config, args.defense, args.data)
    results = run_trials(cfg, trial_num=3)
