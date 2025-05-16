import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from tqdm import tqdm

from utils.metric import ARUC, WARUC


class CITED:
    def __init__(self, model, data, level='N/A', device='cpu'):
        """
        1. load target model
        2. extract signature
        3. deploy detection module
        4. using cited inference
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.level = level
        self.device = device
        self._load_model()

    def _load_model(self):
        target_path = './output/target'
        model_path = os.path.join(target_path, f'{self.data.name}_{self.model.name}.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print('model load: ', model_path)

    def _save_model(self):
        os.makedirs('./output/defense', exist_ok=True)
        save_path = f'./output/defense/{self.data.name}_{self.model.name}_CITED.pth'
        torch.save(self.model.state_dict(), save_path)
        print(f'Defense Model saved to {save_path}')

    def signature(self, cited_boundary_ratio=0.1, cited_signature_ratio=0.2):
        """
        1. boundary nodes
        2. signature nodes
        """
        x = self.data.x
        edge_index = self.data.edge_index
        logits, O = self.model(x, edge_index)  # inference once

        # boundary nodes
        boundary_nodes_index = self._extract_boundary_nodes_plus(logits, cited_boundary_ratio)
        # signature nodes
        signature_nodes_index, signature_threshold = self._extract_signature_nodes(boundary_nodes_index, O,
                                                                                   cited_signature_ratio)
        # Step 3: create deep copy of original data
        signature_data = copy.deepcopy(self.data)

        # Step 4: store signature metadata
        signature_data.signature_nodes_index = signature_nodes_index
        signature_data.signature_threshold = signature_threshold

        # Step 5: generate and store signature mask
        signature_mask = torch.zeros(signature_data.num_nodes, dtype=torch.bool)
        signature_mask[signature_nodes_index] = True
        signature_data.signature_mask = signature_mask

        return signature_data

    def finetune_signature(self, signature_data, epochs=50, lr=0.01, weight_decay=5e-4):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        x, edge_index, y = signature_data.x, signature_data.edge_index, signature_data.y
        train_mask = signature_data.train_mask
        test_mask = signature_data.test_mask
        signature_mask = signature_data.signature_mask

        pbar = tqdm(range(epochs), desc="Fine-tuning on signature data")
        for epoch in pbar:
            optimizer.zero_grad()
            out, _ = self.model(x, edge_index)
            loss = F.cross_entropy(out[signature_mask], y[signature_mask])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(epoch=epoch, loss=loss.item())

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(x, edge_index)

            # Test accuracy
            pred_test = logits[test_mask].argmax(dim=1)
            acc_test = (pred_test == y[test_mask]).float().mean().item()
            print(f'[Defense]>>>>>>>>>>>>>>>>>>>>>Test Acc: {acc_test:.4f}')

            # Signature accuracy
            pred_sig = logits[signature_mask].argmax(dim=1)
            acc_sig = (pred_sig == y[signature_mask]).float().mean().item()
            print(f'[Defense]>>>>>>>>>>>>>>>>>>>>>Signature Acc: {acc_sig:.4f}')

        self._save_model()

    def _extract_boundary_nodes(self, logits, boundary_ratio):
        train_mask = self.data.train_mask
        candidate_indices = train_mask.nonzero(as_tuple=True)[0]
        logits_train = logits[candidate_indices]

        topk = torch.topk(logits_train, 3, dim=1)
        top1, top2, top3 = topk.values[:, 0], topk.values[:, 1], topk.values[:, 2]
        s_boundary = F.relu(top2 - top1) - F.relu(top2 - top1)

        m = int(boundary_ratio * candidate_indices.size(0))
        topk_indices = torch.topk(s_boundary, m, largest=False).indices
        boundary_nodes = candidate_indices[topk_indices]
        return boundary_nodes

    def _extract_boundary_nodes_plus(self, logits, boundary_ratio, lambda_coef=1.0):
        """
        Select boundary nodes according to the scoring rule
            s(v) = ReLU(z_q − z_p) − λ · H(softmax(z))
        """
        train_mask = self.data.train_mask
        candidate_indices = train_mask.nonzero(as_tuple=True)[0]
        logits_train = logits[candidate_indices]

        # Top‑1 and top‑2 logits
        topk_vals, _ = torch.topk(logits_train, 2, dim=1)
        top1_vals = topk_vals[:, 0]
        top2_vals = topk_vals[:, 1]

        # Margin term (ReLU on the gap between top‑2 and top‑1)
        margin_term = F.relu(top2_vals - top1_vals)

        # Entropy term
        probs = torch.softmax(logits_train, dim=1)
        entropy_term = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

        # Boundary score
        s_boundary = margin_term - lambda_coef * entropy_term

        # Select the lowest‑scoring nodes
        m = max(1, int(boundary_ratio * candidate_indices.size(0)))
        selected = torch.topk(s_boundary, m, largest=False).indices
        boundary_nodes = candidate_indices[selected]
        return boundary_nodes

    def _extract_signature_nodes(self, signature_nodes, O: dict, area_ratio: float):
        train_mask = self.data.train_mask
        candidate_indices = train_mask.nonzero(as_tuple=True)[0]
        s_sig_all = self._signature_area_score(signature_nodes, O)
        s_sig = s_sig_all[candidate_indices]

        m = int(area_ratio * candidate_indices.size(0))
        topk_scores, topk_local_idx = torch.topk(s_sig, m, largest=False)
        topk_indices = candidate_indices[topk_local_idx]
        threshold = topk_scores.max().item()

        area_nodes = torch.cat([topk_indices, signature_nodes]).unique()
        return area_nodes, threshold

    def _signature_area_score(self, signature_nodes, O: dict):
        embedding = O['embedding']  # [N, D]
        soft_label = O['soft_label']  # [N, C]
        hard_label = O['hard_label']  # [N]
        edge_index = self.data.edge_index
        num_nodes = embedding.size(0)
        device = embedding.device

        y_i = hard_label
        y_sig = hard_label[signature_nodes]

        # ---------- Margin ----------
        emb_dist = torch.cdist(embedding, embedding[signature_nodes])
        same_class_mask = (y_i.unsqueeze(1) == y_sig.unsqueeze(0))
        masked_emb_dist = emb_dist.clone()
        masked_emb_dist[~same_class_mask] = float('inf')
        s_margin = masked_emb_dist.min(dim=1).values

        # ---------- Thickness ----------
        soft_dist = torch.cdist(soft_label, soft_label[signature_nodes])
        masked_soft_dist = soft_dist.clone()
        masked_soft_dist[~same_class_mask] = float('inf')
        nearest_sig_idx = masked_soft_dist.argmin(dim=1)
        s_thickness_raw = masked_soft_dist[torch.arange(num_nodes), nearest_sig_idx]

        t_i = soft_label[torch.arange(num_nodes), y_i]
        j_star = signature_nodes[nearest_sig_idx]
        t_j = soft_label[j_star, y_sig[nearest_sig_idx]]
        gamma, k = 0.1, 10.0
        sigmoid_weight = torch.sigmoid(k * (gamma - (t_i - t_j)))
        s_thickness = s_thickness_raw * sigmoid_weight

        # ---------- Complexity ----------
        row, col = edge_index
        disagreement = (hard_label[row] != hard_label[col]).float()
        deg = torch.bincount(row, minlength=num_nodes).clamp(min=1)
        s_complexity = torch.zeros(num_nodes, device=device).scatter_add_(0, row, disagreement) / deg

        # ---------- Aggregate ----------
        s_margin = self.normalize(s_margin)
        s_thickness = self.normalize(s_thickness)
        s_complexity = self.normalize(s_complexity)
        alpha1, alpha2, alpha3 = 0.1, 0.8, 0.1
        s_sig = alpha1 * s_margin + alpha2 * s_thickness + alpha3 * s_complexity
        return s_sig

    def normalize(self, t):
        return (t - t.min()) / (t.max() - t.min() + 1e-8)


class CITEDOVPipeline:
    def __init__(self, cited_model, cited_data, defense_name, independent_models, surrogate_models, device='cpu'):
        self.cited_model = cited_model.to(device)
        self.cited_data = cited_data.to(device)
        self.defense_name = defense_name
        self.device = device
        self.independent_models = independent_models
        self.surrogate_models = surrogate_models

    def _load_model(self):
        target_path = './output/defense'
        model_path = os.path.join(target_path,
                                  f'{self.cited_data.name}_{self.cited_model.name}_{self.defense_name}.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.cited_model.load_state_dict(state_dict)
        self.cited_model.eval()
        print('[Verification CITED]model load: ', model_path)

    # def _cited(self, boundary_ratio, area_ratio):
    #     cited = CITED(self.target_model, self.target_data, device=self.device)
    #     _, sig_area_index, _ = cited.signature(boundary_ratio, area_ratio)
    #     self.signature = sig_area_index  # shape [N], bool tensor
    #     self.target_model = cited.model

    def _prepare_models(self):
        all_models = self.independent_models + self.surrogate_models
        all_labels = [0] * len(self.independent_models) + [1] * len(self.surrogate_models)  # 0 = neg, 1 = pos

        combined = list(zip(all_models, all_labels))
        random.shuffle(combined)
        self.suspicious_models, self.labels = zip(*combined)

    def _infer_signature(self, level: str):
        assert level in {"label", "embedding"}, f"Unsupported level: {level}"

        # signature_index = self.signature
        x, edge_index = self.cited_data.x, self.cited_data.edge_index
        signature_mask = self.cited_data.signature_mask

        # Extract subgraph
        sig_edge_index, _ = subgraph(signature_mask, edge_index, relabel_nodes=True)
        sig_x = x[signature_mask]

        with torch.no_grad():
            self.target_model.eval()
            logits, O = self.target_model(sig_x, sig_edge_index)
            if level == "label":
                self.target_pred = logits.argmax(dim=1).cpu()
            elif level == "embedding":
                self.target_pred = O[level].cpu()

        self.model_outputs = []
        for model in self.suspicious_models:
            model.eval()
            with torch.no_grad():
                logits, O = model(sig_x, sig_edge_index)
                if level == "label":
                    pred = logits.argmax(dim=1).cpu()
                elif level == "embedding":
                    pred = O[level].cpu()
                self.model_outputs.append(pred)

    def _infer_signature_all(self, level: str):
        assert level in {"label", "embedding"}, f"Unsupported level: {level}"

        x, edge_index = self.cited_data.x, self.cited_data.edge_index
        signature_mask = self.cited_data.signature_mask

        with torch.no_grad():
            self.cited_model.eval()
            logits, O = self.cited_model(x, edge_index)
            if level == "label":
                self.target_pred = logits[signature_mask].argmax(dim=1).cpu()
            elif level == "embedding":
                self.target_pred = O[level][signature_mask].cpu()

        self.model_outputs = []
        for model in self.suspicious_models:
            model.eval()
            with torch.no_grad():
                logits, O = model(x, edge_index)
                if level == "label":
                    pred = logits[signature_mask].argmax(dim=1).cpu()
                elif level == "embedding":
                    pred = O[level][signature_mask].cpu()
                self.model_outputs.append(pred)

    def _compute_metric(self, mode='label', plot_path=None):
        """
        mode = 'label' or 'embedding'
        """
        if mode == 'label':
            metric = ARUC(tau=0.5, r=100)
        elif mode == 'embedding':
            metric = WARUC(tau=0.5, r=100)
        else:
            raise ValueError(f"Unknown verification level: {mode}")

        metric.init_target_pred(self.target_pred)

        for pred, label in zip(self.model_outputs, self.labels):
            metric.update(pred, sample_label=label)
        print('[DEBUG]metric:', metric)
        print('[DEBUG]metric target pred:', metric.target_pred.shape)
        print('[DEBUG]metric pos pred:', len(metric.pos_samples))
        print('[DEBUG]metric neg pred:', len(metric.neg_samples))
        aruc, R, U, threshold = metric.compute(plot_path)
        if mode == 'label':
            asr = metric.compute_asr()
        elif mode == 'embedding':
            asr = metric.compute_dsr()
        else:
            raise ValueError(f"Unknown verification level: {mode}")
        print(f"[{mode.upper()}]Verification ARUC result: {aruc}, ASR: {asr}")
        return aruc, R, U, asr, threshold

    def verify(self, level='label', plot_path=None):
        """
        1. concat suspicious models, with related labels, then shuffle
        2. using signature inference target_model to get ground truth
        3. using signature inference all suspicious models to get output
        4. apply metric class
        5. emb level: calc wasserstein distance; label level: calc matching score
        """
        # self._cited(cited_boundary_ratio, cited_signature_ratio)
        self._prepare_models()
        # TODO use 'subgraph' or 'all graph' to inference
        # self._infer_signature(level)
        self._infer_signature_all(level)
        aruc, R, U, asr, threshold = self._compute_metric(level, plot_path)
        return aruc, R, U, asr, threshold

    def _match_label(self, pred, target_pred):
        """
        Compute label agreement ratio between a predicted label array and the target prediction.
        pred: shape [N,] int array from a suspicious model
        target_pred: shape [N,] int array from target model
        Returns: float in [0,1], proportion of matching labels
        """
        assert pred.shape == target_pred.shape
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target_pred, torch.Tensor):
            target_pred = target_pred.detach().cpu().numpy()
        return np.mean(pred == target_pred)

    def _match_dist(self, pred, target_pred):
        """
        Compute label agreement ratio between a predicted label array and the target prediction.
        pred: shape [N,] int array from a suspicious model
        target_pred: shape [N,] int array from target model
        Returns: float in [0,1], proportion of matching labels
        """
        assert pred.shape == target_pred.shape
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target_pred, torch.Tensor):
            target_pred = target_pred.detach().cpu().numpy()
        return np.mean(np.linalg.norm(pred - target_pred, axis=1))

    def _compute_acc(self, threshold, level):
        """
        Predict whether a sample is positive (same as target) based on threshold,
        and compute classification accuracy compared to ground-truth label.
        """
        correct = 0
        total = 0
        print('[DEBUG]model outputs:', len(self.model_outputs), self.target_pred.shape)
        for pred, label in zip(self.model_outputs, self.labels):
            print('[DEBUG] pred: ', pred, 'label: ', label, 'target: ', self.target_pred)
            if level == "label":
                score = self._match_label(pred, self.target_pred)
                if score >= threshold:
                    total += 1
                    if label == 1:
                        correct += 1
            elif level == "embedding":
                score = self._match_dist(pred, self.target_pred)
                if score <= threshold:
                    total += 1
                    if label == 1:
                        correct += 1
        acc = correct / total if total > 0 else 0
        print(f'[DEBUG]Threshold: {threshold}, Accuracy: {acc}, Correct: {correct}, Total: {total}')
        return acc

    def accuracy(self, threshold, level='label'):
        assert level in {'label', 'embedding'}, f"Unsupported level: {level}"
        self._prepare_models()
        self._infer_signature_all(level)
        acc = self._compute_acc(threshold, level)
        return acc


class CITEDVar:
    def __init__(self, model, data, level='N/A', device='cpu'):
        """
        1. load target model
        2. extract signature
        3. deploy detection module
        4. using cited inference
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.level = level
        self.device = device
        self._load_model()

    def _load_model(self):
        target_path = './output/target'
        model_path = os.path.join(target_path, f'{self.data.name}_{self.model.name}.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print('model load: ', model_path)

    def _save_model(self):
        os.makedirs('./output/defense', exist_ok=True)
        save_path = f'./output/defense/{self.data.name}_{self.model.name}_CITED.pth'
        torch.save(self.model.state_dict(), save_path)
        print(f'Defense Model saved to {save_path}')

    def signature(self, cited_boundary_ratio=0.1, cited_signature_ratio=0.5, choice='margin'):
        """
        1. boundary nodes
        2. signature nodes
        """
        assert choice in {'margin', 'thickness', 'heterogeneity', 'all'}
        print(f'>>>>>>>>>>>>>>>>signature-{choice}>>>>>>>>>>>>>>>')
        x = self.data.x
        edge_index = self.data.edge_index
        logits, O = self.model(x, edge_index)  # inference once

        # boundary nodes
        boundary_nodes_index = self._extract_boundary_nodes_plus(logits, cited_boundary_ratio)
        # signature nodes
        signature_nodes_index, signature_threshold = self._extract_signature_nodes(boundary_nodes_index, O,
                                                                                   cited_signature_ratio, choice)
        # Step 3: create deep copy of original data
        signature_data = copy.deepcopy(self.data)

        # Step 4: store signature metadata
        signature_data.signature_nodes_index = signature_nodes_index
        signature_data.signature_threshold = signature_threshold

        # Step 5: generate and store signature mask
        signature_mask = torch.zeros(signature_data.num_nodes, dtype=torch.bool)
        signature_mask[signature_nodes_index] = True
        signature_data.signature_mask = signature_mask

        return signature_data

    def signature_by_num(self, cited_boundary_ratio=0.1, cited_signature_ratio=0.5, choice='margin', signature_node_num=None):
        """
        1. boundary nodes
        2. signature nodes
        """
        assert choice in {'margin', 'thickness', 'heterogeneity', 'all'}
        print(f'>>>>>>>>>>>>>>>>signature-{choice}>>>>>>>>>>>>>>>')
        x = self.data.x
        edge_index = self.data.edge_index
        logits, O = self.model(x, edge_index)  # inference once

        # boundary nodes
        boundary_nodes_index = self._extract_boundary_nodes_plus(logits, cited_boundary_ratio)
        # signature nodes
        signature_nodes_index, signature_threshold = self._extract_signature_nodes(boundary_nodes_index, O,
                                                                                   cited_signature_ratio, choice)

        if signature_node_num is not None and len(signature_nodes_index) > signature_node_num:
            signature_nodes_index = signature_nodes_index[:signature_node_num]

        print('[CITED]>>>>>>>>>>>>>>>>>>>>>signature_nodes_index: ', signature_nodes_index.shape)
        # Step 3: create deep copy of original data
        signature_data = copy.deepcopy(self.data)

        # Step 4: store signature metadata
        signature_data.signature_nodes_index = signature_nodes_index
        signature_data.signature_threshold = signature_threshold

        # Step 5: generate and store signature mask
        signature_mask = torch.zeros(signature_data.num_nodes, dtype=torch.bool)
        signature_mask[signature_nodes_index] = True
        signature_data.signature_mask = signature_mask

        return signature_data

    def finetune_signature(self, signature_data, epochs=50, lr=0.01, weight_decay=5e-4):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        x, edge_index, y = signature_data.x, signature_data.edge_index, signature_data.y
        train_mask = signature_data.train_mask
        test_mask = signature_data.test_mask
        signature_mask = signature_data.signature_mask

        pbar = tqdm(range(epochs), desc="Fine-tuning on signature data")
        for epoch in pbar:
            optimizer.zero_grad()
            out, _ = self.model(x, edge_index)
            loss = F.cross_entropy(out[signature_mask], y[signature_mask])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(epoch=epoch, loss=loss.item())

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(x, edge_index)

            # Test accuracy
            pred_test = logits[test_mask].argmax(dim=1)
            acc_test = (pred_test == y[test_mask]).float().mean().item()
            print(f'[Defense]>>>>>>>>>>>>>>>>>>>>>Test Acc: {acc_test:.4f}')

            # Signature accuracy
            pred_sig = logits[signature_mask].argmax(dim=1)
            acc_sig = (pred_sig == y[signature_mask]).float().mean().item()
            print(f'[Defense]>>>>>>>>>>>>>>>>>>>>>Signature Acc: {acc_sig:.4f}')

        self._save_model()

    def _extract_boundary_nodes_plus(self, logits, boundary_ratio, lambda_coef=1.0):
        """
        Select boundary nodes according to the scoring rule
            s(v) = ReLU(z_q − z_p) − λ · H(softmax(z))
        """
        train_mask = self.data.train_mask
        candidate_indices = train_mask.nonzero(as_tuple=True)[0]
        logits_train = logits[candidate_indices]

        # Top‑1 and top‑2 logits
        topk_vals, _ = torch.topk(logits_train, 2, dim=1)
        top1_vals = topk_vals[:, 0]
        top2_vals = topk_vals[:, 1]

        # Margin term (ReLU on the gap between top‑2 and top‑1)
        margin_term = F.relu(top2_vals - top1_vals)

        # Entropy term
        probs = torch.softmax(logits_train, dim=1)
        entropy_term = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

        # Boundary score
        s_boundary = margin_term - lambda_coef * entropy_term

        # Select the lowest‑scoring nodes
        m = max(1, int(boundary_ratio * candidate_indices.size(0)))
        selected = torch.topk(s_boundary, m, largest=False).indices
        boundary_nodes = candidate_indices[selected]
        return boundary_nodes

    def _extract_signature_nodes(self, signature_nodes, O: dict, area_ratio: float, choice):
        train_mask = self.data.train_mask
        candidate_indices = train_mask.nonzero(as_tuple=True)[0]
        if choice == 'margin':
            s_sig_all = self._signature_area_score_margin(signature_nodes, O)
        elif choice == 'thickness':
            s_sig_all = self._signature_area_score_thickness(signature_nodes, O)
        elif choice == 'heterogeneity':
            s_sig_all = self._signature_area_score_heterogeneity(signature_nodes, O)
        elif choice == 'all':
            s_sig_all = self._signature_area_score(signature_nodes, O)
        else:
            raise NotImplementedError

        s_sig = s_sig_all[candidate_indices]

        m = int(area_ratio * candidate_indices.size(0))
        topk_scores, topk_local_idx = torch.topk(s_sig, m, largest=False)
        topk_indices = candidate_indices[topk_local_idx]
        threshold = topk_scores.max().item()

        area_nodes = torch.cat([topk_indices, signature_nodes]).unique()
        return area_nodes, threshold

    def _signature_area_score(self, signature_nodes, O: dict):
        embedding = O['embedding']  # [N, D]
        soft_label = O['soft_label']  # [N, C]
        hard_label = O['hard_label']  # [N]
        edge_index = self.data.edge_index
        num_nodes = embedding.size(0)
        device = embedding.device

        y_i = hard_label
        y_sig = hard_label[signature_nodes]

        # ---------- Margin ----------
        emb_dist = torch.cdist(embedding, embedding[signature_nodes])
        same_class_mask = (y_i.unsqueeze(1) == y_sig.unsqueeze(0))
        masked_emb_dist = emb_dist.clone()
        masked_emb_dist[~same_class_mask] = float('inf')
        s_margin = masked_emb_dist.min(dim=1).values

        # ---------- Thickness ----------
        soft_dist = torch.cdist(soft_label, soft_label[signature_nodes])
        masked_soft_dist = soft_dist.clone()
        masked_soft_dist[~same_class_mask] = float('inf')
        nearest_sig_idx = masked_soft_dist.argmin(dim=1)
        s_thickness_raw = masked_soft_dist[torch.arange(num_nodes), nearest_sig_idx]

        t_i = soft_label[torch.arange(num_nodes), y_i]
        j_star = signature_nodes[nearest_sig_idx]
        t_j = soft_label[j_star, y_sig[nearest_sig_idx]]
        gamma, k = 0.1, 10.0
        sigmoid_weight = torch.sigmoid(k * (gamma - (t_i - t_j)))
        s_thickness = s_thickness_raw * sigmoid_weight

        # ---------- Complexity ----------
        row, col = edge_index
        disagreement = (hard_label[row] != hard_label[col]).float()
        deg = torch.bincount(row, minlength=num_nodes).clamp(min=1)
        s_complexity = torch.zeros(num_nodes, device=device).scatter_add_(0, row, disagreement) / deg

        # ---------- Aggregate ---------- TODO finetune in ablation study
        s_margin = self.normalize(s_margin)
        s_thickness = self.normalize(s_thickness)
        s_complexity = self.normalize(s_complexity)
        alpha1, alpha2, alpha3 = 0.1, 0.8, 0.1
        s_sig = alpha1 * s_margin + alpha2 * s_thickness + alpha3 * s_complexity
        return s_sig

    def _signature_area_score_margin(self, signature_nodes, O: dict):
        embedding = O['embedding']  # [N, D]
        soft_label = O['soft_label']  # [N, C]
        hard_label = O['hard_label']  # [N]
        edge_index = self.data.edge_index
        num_nodes = embedding.size(0)
        device = embedding.device

        y_i = hard_label
        y_sig = hard_label[signature_nodes]

        # ---------- Margin ----------
        emb_dist = torch.cdist(embedding, embedding[signature_nodes])
        same_class_mask = (y_i.unsqueeze(1) == y_sig.unsqueeze(0))
        masked_emb_dist = emb_dist.clone()
        masked_emb_dist[~same_class_mask] = float('inf')
        s_margin = masked_emb_dist.min(dim=1).values

        s_margin = self.normalize(s_margin)
        s_sig = s_margin
        return s_sig

    def _signature_area_score_thickness(self, signature_nodes, O: dict):
        embedding = O['embedding']  # [N, D]
        soft_label = O['soft_label']  # [N, C]
        hard_label = O['hard_label']  # [N]
        edge_index = self.data.edge_index
        num_nodes = embedding.size(0)
        device = embedding.device

        y_i = hard_label
        y_sig = hard_label[signature_nodes]

        same_class_mask = (y_i.unsqueeze(1) == y_sig.unsqueeze(0))

        # ---------- Thickness ----------
        soft_dist = torch.cdist(soft_label, soft_label[signature_nodes])
        masked_soft_dist = soft_dist.clone()
        masked_soft_dist[~same_class_mask] = float('inf')
        nearest_sig_idx = masked_soft_dist.argmin(dim=1)
        s_thickness_raw = masked_soft_dist[torch.arange(num_nodes), nearest_sig_idx]

        t_i = soft_label[torch.arange(num_nodes), y_i]
        j_star = signature_nodes[nearest_sig_idx]
        t_j = soft_label[j_star, y_sig[nearest_sig_idx]]
        gamma, k = 0.2, 10.0
        sigmoid_weight = torch.sigmoid(k * (gamma - (t_i - t_j)))
        s_thickness = s_thickness_raw * sigmoid_weight

        s_thickness = self.normalize(s_thickness)
        s_sig = s_thickness
        return s_sig

    def _signature_area_score_heterogeneity(self, signature_nodes, O: dict):
        embedding = O['embedding']  # [N, D]
        soft_label = O['soft_label']  # [N, C]
        hard_label = O['hard_label']  # [N]
        edge_index = self.data.edge_index
        num_nodes = embedding.size(0)
        device = embedding.device

        # ---------- Complexity ----------
        row, col = edge_index
        disagreement = (hard_label[row] != hard_label[col]).float()
        deg = torch.bincount(row, minlength=num_nodes).clamp(min=1)
        s_complexity = torch.zeros(num_nodes, device=device).scatter_add_(0, row, disagreement) / deg

        s_complexity = self.normalize(s_complexity)
        s_sig = s_complexity
        return s_sig

    def normalize(self, t):
        return (t - t.min()) / (t.max() - t.min() + 1e-8)
