from models.factory import get_model_by_name
from pipline.cited import CITEDVar
from utils.dataset import CustomDataset


def run_once_cited(config, trial_id=0):
    print(f'\n========== Trial {trial_id + 1} ==========')

    dataset = CustomDataset(config['ds_name'])
    data = dataset.get()
    dataset.stats()

    print('[Defense] Start CITED defense')
    target_model = get_model_by_name(config['model_name'], data, config['hidden_dim'])
    cited = CITEDVar(target_model, data, device=config['device'])
    cited_data = cited.signature_by_num(cited_boundary_ratio=config['cited_boundary_ratio'],
                                        cited_signature_ratio=config['cited_signature_ratio'],
                                        choice=config['cited_choice'],
                                        signature_node_num=config['cited_signature_node_num'],)
    cited.finetune_signature(cited_data, epochs=config['finetune_epochs'], lr=config['lr'],
                             weight_decay=config['weight_decay'])


if __name__ == '__main__':
    config = {
        'ds_name': 'pubmed',
        # exp setting
        'level': 'label',
        'variant_num': 5,
        # model setting
        'model_name': 'gcn',
        'hidden_dim': 128,
        # train setting
        'train_epochs': 200,
        'finetune_epochs': 20,
        'lr': 0.001,
        'weight_decay': 1e-5,
        # attack setting
        'query_ratio': 0.5,
        # CITED choice
        'cited_boundary_ratio': 0.1,
        'cited_signature_ratio': 0.2,
        'cited_choice': 'all',
        'cited_signature_node_num': 30,
        # misc
        'fixed_seed': 42,
        'device': 'cpu',
    }

    run_once_cited(config)
