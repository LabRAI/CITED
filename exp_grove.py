import numpy as np

from models.factory import get_model_by_name
from pipline.grove import GrovePipeline
from utils.dataset import CustomDataset


def run_grove():
    # init data
    dataset = CustomDataset(args.data)
    data = dataset.get()
    dataset.stats()
    # init model
    target_model = get_model_by_name(model_name, data, hidden_dim=128)
    pipe = GrovePipeline(target_model, data, ds_name, variant_num, query_ratio, conf_ratio, epochs, lr, weight_decay,
                         device=device)
    pipe.train(3)


def run_cited_infer(trial_num=3):
    import os
    import torch
    from time import time
    # init data
    dataset = CustomDataset(args.data)
    data = dataset.get()
    dataset.stats()
    # init model
    target_model = get_model_by_name(model_name, data, hidden_dim=128)
    target_path = './output/defense'
    model_path = os.path.join(target_path, f'{data.name}_{target_model.name}_CITED.pth')
    state_dict = torch.load(model_path, map_location=device)
    target_model.load_state_dict(state_dict)
    target_model.eval()
    data.to(device)
    target_model.to(device)

    def inference_once(data, target_model):
        x, edge_index = data.x, data.edge_index
        test_mask = data.test_mask
        with torch.no_grad():
            target_model.eval()
            logits, O = target_model(x, edge_index)
            target_pred = O['embedding'][test_mask].argmax(dim=1).cpu()

    cited_inference_time_list = []
    for i in range(trial_num):
        t0 = time()
        inference_once(data, target_model)
        t1 = time()
        inference_time = t1 - t0
        cited_inference_time_list.append(inference_time)

    print(
        f"CITED inference time: {np.mean(cited_inference_time_list):.4f} ± {np.std(cited_inference_time_list):.4f}, using device {device}")
    return cited_inference_time_list


def run_grove_infer(trial_num=3):
    import os
    import torch
    # init data
    dataset = CustomDataset(args.data)
    data = dataset.get()
    dataset.stats()
    # init model
    target_model = get_model_by_name(model_name, data, hidden_dim=128)
    target_path = './output/defense'
    model_path = os.path.join(target_path, f'{data.name}_{target_model.name}_CITED.pth')
    state_dict = torch.load(model_path, map_location=device)
    target_model.load_state_dict(state_dict)
    target_model.eval()
    data.to(device)
    target_model.to(device)

    pipe = GrovePipeline(target_model, data, ds_name, variant_num, query_ratio, conf_ratio, epochs, lr, weight_decay,
                         device=device)
    grove_inference_time_list = []
    for i in range(trial_num):
        inference_time = pipe.inference_time()
        grove_inference_time_list.append(inference_time)

    print(
        f"GrOVe inference time: {np.mean(grove_inference_time_list):.4f} ± {np.std(grove_inference_time_list):.4f}, using device {device}")
    return grove_inference_time_list


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Dataset name')
    parser.add_argument('--device', type=str, default='0', help='GPU device id (e.g., 0, 1, 2)')

    args = parser.parse_args()

    # ds_name = 'cora'
    ds_name = args.data
    model_name = 'gcn'
    variant_num = 15
    query_ratio = 0.5
    conf_ratio = 0.2
    epochs = 200
    lr = 0.001
    weight_decay = 1e-5
    device = f'cuda:{args.device}'
    # device = 'cpu'

    # run_grove()

    grove_list = run_grove_infer(3)
    # cited_list = run_cited_infer(3)
    print(f"GrOVe inference time: {np.mean(grove_list):.4f} ± {np.std(grove_list):.4f}, using device {device}")
    # print(f"CITED inference time: {np.mean(cited_list):.4f} ± {np.std(cited_list):.4f}, using device {device}")

