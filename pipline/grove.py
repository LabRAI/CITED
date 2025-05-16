import os
import random
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.factory import generate_model_variants, get_model_by_name
from utils.dataset import IndependentDataset, CustomDataset
from utils.metric import WARUC


class GrovePipeline:
    def __init__(self, model, data, ds_name, variant_num, query_ratio, conf_ratio, epochs, lr, weight_decay,
                 device='cpu'):
        self.target_model = model.to(device)
        self.data = data.to(device)
        self.ds_name = ds_name
        self.device = device
        self.variant_num = variant_num
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.query_ratio = query_ratio
        self.conf_ratio = conf_ratio

    def _load_target_model(self):
        target_path = './output/target'
        model_path = os.path.join(target_path, f'{self.data.name}_{self.target_model.name}.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.target_model.load_state_dict(state_dict)
        self.target_model.eval()
        print('[Grove]target model load: ', model_path)

    def _train_target(self):
        self._load_target_model()
        return self.target_model

    def _train_independent_once(self, independent_model, independent_data):
        independent_model.reset_parameters()
        optimizer = torch.optim.Adam(independent_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        pbar = tqdm(range(self.epochs), desc='Train Grove Independent')
        for epoch in pbar:
            independent_model.train()
            optimizer.zero_grad()
            logits, _ = independent_model(independent_data.x, independent_data.edge_index)
            loss = F.cross_entropy(logits[independent_data.train_mask], independent_data.y[independent_data.train_mask])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'],
                             weight_decay=optimizer.param_groups[0]['weight_decay'], epoch=epoch)
        # Evaluate after training
        independent_model.eval()
        with torch.no_grad():
            logits, _ = independent_model(independent_data.x, independent_data.edge_index)
            pred = logits.argmax(dim=1)

            train_acc = (pred[independent_data.train_mask] == independent_data.y[
                independent_data.train_mask]).float().mean().item()
            val_acc = (pred[independent_data.val_mask] == independent_data.y[
                independent_data.val_mask]).float().mean().item()
            test_acc = (pred[independent_data.test_mask] == independent_data.y[
                independent_data.test_mask]).float().mean().item()
            print(f"Independent train acc: {train_acc}, val acc: {val_acc}, test acc: {test_acc}")
        print('One independent model training complete.')
        return independent_model

    def _train_independent(self):
        # init model
        independent_models = generate_model_variants(self.target_model, self.variant_num)
        independent_models = [model.to(self.device) for model in independent_models]
        trained_models = []
        # random seed
        seed_list = [random.randint(0, 100000) for _ in range(len(independent_models))]
        for i, (model, seed) in enumerate(zip(independent_models, seed_list)):
            # data
            dataset = IndependentDataset(self.ds_name)
            data = dataset.generate(num_class_samples=50, seed=seed).to(self.device)
            independent_model_trained = self._train_independent_once(model, data)
            trained_models.append(independent_model_trained)
        return trained_models

    def _prepare_query_plus(self, seed=42):
        """
        Prepare a query mask by selecting ambiguous nodes whose top-2 logits are close.
        The number of queries is query_ratio * |train_set|.
        conf_ratio determines the proportion selected based on top1-top2 closeness.
        The rest are randomly chosen from unused nodes.
        """
        x, edge_index = self.data.x, self.data.edge_index
        with torch.no_grad():
            logits, O = self.target_model(x, edge_index)
            soft_label = O['soft_label']  # shape: [N, C]

        # Step 1: Identify unused nodes
        used_mask = self.data.train_mask | self.data.val_mask | self.data.test_mask
        unused_mask = ~used_mask
        unused_index = unused_mask.nonzero(as_tuple=True)[0]

        # Step 2: Compute top1 - top2 margin for all nodes
        top_values, _ = torch.topk(soft_label, k=2, dim=1)  # shape: [N, 2]
        margin = top_values[:, 0] - top_values[:, 1]  # smaller margin = more ambiguous

        # Step 3: Rank unused nodes by smallest margin
        margin_unused = margin[unused_mask]
        margin_unused_indices = unused_index[torch.argsort(margin_unused)]

        # Step 4: Select query_size nodes
        train_size = int(self.data.train_mask.sum().item())
        query_size = int(self.query_ratio * train_size)
        conf_size = int(self.conf_ratio * query_size)
        random_size = query_size - conf_size

        generator = torch.Generator().manual_seed(seed)

        # Top-k most ambiguous (lowest margin)
        ambiguous_index = margin_unused_indices[:conf_size]

        # Remaining candidates: remove the already selected ambiguous ones
        remaining_candidates = unused_index[~torch.isin(unused_index, ambiguous_index)]
        if remaining_candidates.size(0) >= random_size:
            random_index = remaining_candidates[
                torch.randperm(remaining_candidates.size(0), generator=generator)[:random_size]]
        else:
            # Not enough to fill, just use all remaining
            random_index = remaining_candidates

        # Final query index
        selected_query_index = torch.cat([ambiguous_index, random_index], dim=0)

        # Build query mask
        query_mask = torch.zeros_like(unused_mask)
        query_mask[selected_query_index] = True

        print(f"[Query PLUS]Generated query_mask with {query_mask.sum().item()} nodes (target={query_size})")
        return query_mask

    def _query_victim_all(self, query_mask):
        """
        Query victim model to obtain output for all nodes
        """
        with torch.no_grad():
            x, edge_index = self.data.x, self.data.edge_index
            logits, O = self.target_model(x, edge_index)
            query_response = O['embedding'][query_mask]  # [N, D]
            print(f"Queried victim at embedding level.")
        return query_response

    def _train_surrogate_once(self, surrogate_model, query_mask, query_response):
        """
        Train surrogate model using query input and victim response
        """
        surrogate_model.reset_parameters()
        optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        pbar = tqdm(range(self.epochs), desc='Train Grove Surrogate NLL loss')
        for epoch in pbar:
            surrogate_model.train()
            optimizer.zero_grad()
            logits, O = surrogate_model(self.data.x, self.data.edge_index)
            lambda_ = 0.01
            loss_mse = F.mse_loss(O['embedding'][query_mask], query_response)
            loss_ce = F.cross_entropy(logits[query_mask], self.data.y[query_mask])
            loss = lambda_ * loss_mse + (1 - lambda_) * loss_ce
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'],
                             weight_decay=optimizer.param_groups[0]['weight_decay'], epoch=epoch)

        surrogate_model.eval()
        with torch.no_grad():
            logits, _ = surrogate_model(self.data.x, self.data.edge_index)
            pred = logits.argmax(dim=1)

            train_acc = (pred[self.data.train_mask] == self.data.y[self.data.train_mask]).float().mean().item()
            val_acc = (pred[self.data.val_mask] == self.data.y[self.data.val_mask]).float().mean().item()
            test_acc = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).float().mean().item()
            print(f"Surrogate train acc: {train_acc}, val acc: {val_acc}, test acc: {test_acc}")

        print("One surrogate model training complete.")
        return surrogate_model

    def _train_surrogate(self):
        # init model
        surrogate_models = generate_model_variants(self.target_model, self.variant_num)
        surrogate_models = [model.to(self.device) for model in surrogate_models]
        trained_models = []
        seed_list = [random.randint(0, 100000) for _ in range(len(surrogate_models))]
        for i, (model, seed) in enumerate(zip(surrogate_models, seed_list)):
            query_mask = self._prepare_query_plus(seed)
            query_response = self._query_victim_all(query_mask)
            surrogate_model_trained = self._train_surrogate_once(model, query_mask, query_response)
            trained_models.append(surrogate_model_trained)
        return trained_models

    def train_once(self):
        target_model = self._train_target()
        independent_models = self._train_independent()
        surrogate_models = self._train_surrogate()
        suspicious_models, labels = self._prepare_models(independent_models, surrogate_models)
        t0 = time()
        target_pred, models_preds = self._inference_fingerprint(target_model, suspicious_models)
        t1 = time()
        inference_time = t1 - t0
        res_aruc, R, U, res_asr, threshold = self._compute_metric(target_pred, models_preds, labels)
        return {
            'aruc': res_aruc,
            'asr': res_asr,
            'R': R,
            'U': U,
            'threshold': threshold,
            'inference_time': inference_time,
        }

    def train(self, trial_num=3):
        aruc_list = []
        asr_list = []
        R_list = []
        U_list = []
        thre_list = []
        inference_time_list = []
        for trial in range(trial_num):
            result = self.train_once()
            aruc_list.append(result['aruc'])
            asr_list.append(result['asr'])
            R_list.append(result['R'])
            U_list.append(result['U'])
            thre_list.append(result['threshold'])
            inference_time_list.append(result['inference_time'])
        aruc_arr = np.array(aruc_list)
        asr_arr = np.array(asr_list)
        R_arr = np.array(R_list)  # shape [n_trial, 100]
        U_arr = np.array(U_list)  # shape [n_trial, 100]
        thre_list = np.array(thre_list)
        inference_time_list = np.array(inference_time_list)

        print('\n========== Summary ==========')
        print(f'ARUC: {np.mean(aruc_list):.4f} ± {np.std(aruc_list):.4f}')
        print(f'ASR : {np.mean(asr_list):.4f} ± {np.std(asr_list):.4f}')
        print(f'Inference : {np.mean(inference_time_list):.4f} ± {np.std(inference_time_list):.4f}')

        save_path = f'./results/Res_grove_{self.target_model.name}_{self.ds_name}_embedding.npz'
        np.savez(
            save_path,
            aruc=aruc_arr,
            asr=asr_arr,
            R=R_arr,
            U=U_arr,
            threshold=thre_list,
            inference_time=inference_time_list
        )
        print(f'[Saved] Result saved to: {save_path}')

    def _prepare_models(self, independent_models, surrogate_models):
        all_models = independent_models + surrogate_models
        all_labels = [0] * len(independent_models) + [1] * len(surrogate_models)  # 0 = neg, 1 = pos

        combined = list(zip(all_models, all_labels))
        random.shuffle(combined)
        suspicious_models, labels = zip(*combined)
        return suspicious_models, labels

    def _inference_fingerprint(self, target_model, suspicious_models):
        x, edge_index = self.data.x, self.data.edge_index
        # TODO need a mask design
        fingerprint_mask = self.data.train_mask

        with torch.no_grad():
            target_model.eval()
            _, O = target_model(x, edge_index)
            target_pred = O['embedding'][fingerprint_mask].cpu()

        model_outputs = []
        for model in suspicious_models:
            model.eval()
            with torch.no_grad():
                _, O = model(x, edge_index)
                pred = O['embedding'][fingerprint_mask].cpu()
                model_outputs.append(pred)

        return target_pred, model_outputs

    def _compute_metric(self, target_pred, models_preds, labels):
        """
        mode = 'label' or 'embedding'
        """
        metric = WARUC(tau=0.5, r=100)
        metric.init_target_pred(target_pred)

        for pred, label in zip(models_preds, labels):
            metric.update(pred, sample_label=label)
        print('[DEBUG]metric:', metric)
        print('[DEBUG]metric target pred:', metric.target_pred.shape)
        print('[DEBUG]metric pos pred:', len(metric.pos_samples))
        print('[DEBUG]metric neg pred:', len(metric.neg_samples))
        res_aruc, R, U, threshold = metric.compute()
        res_asr = metric.compute_dsr()
        print(f"[GrOVe]Verification ARUC emb result: {res_aruc}, ASR: {res_asr}")
        return res_aruc, R, U, res_asr, threshold

    def inference_time(self):
        # init target model
        target_model = self._train_target()
        # init independent models
        independent_models = generate_model_variants(self.target_model, self.variant_num)
        independent_models = [model.to(self.device) for model in independent_models]
        # init surrogate models
        surrogate_models = generate_model_variants(self.target_model, self.variant_num)
        surrogate_models = [model.to(self.device) for model in surrogate_models]
        # combine
        all_models = independent_models + surrogate_models
        for model in all_models:
            model.reset_parameters()
        # inference
        t0 = time()
        self._inference_fingerprint(target_model, all_models)
        t1 = time()
        inference_time = t1 - t0
        return inference_time


if __name__ == '__main__':
    ds_name = 'cora'
    model_name = 'gcn'
    variant_num = 5
    query_ratio = 0.5
    conf_ratio = 0.2
    epochs = 200
    lr = 0.001
    weight_decay = 1e-5
    device = 'cpu'
    # init data
    dataset = CustomDataset(ds_name)
    data = dataset.get()
    dataset.stats()
    # init model
    target_model = get_model_by_name(model_name, data, hidden_dim=128)
    pipe = GrovePipeline(target_model, data, variant_num, query_ratio, conf_ratio, epochs, lr, weight_decay)
    pipe.train_once()
