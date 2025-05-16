import os
import random

import numpy as np
import torch
from torch_geometric.utils import subgraph

from utils.metric import ARUC


class WMOVPipeline:
    def __init__(self, defense_model, wm_data, defense_name, independent_models, surrogate_models, device='cpu'):
        self.defense_model = defense_model.to(device)
        self.wm_data = wm_data.to(device)
        self.defense_name = defense_name
        self.device = device
        self.independent_models = independent_models
        self.surrogate_models = surrogate_models
        self._load_model()

    def _load_model(self):
        target_path = './output/defense'
        model_path = os.path.join(target_path, f'{self.wm_data.name}_{self.defense_model.name}_{self.defense_name}.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.defense_model.load_state_dict(state_dict)
        self.defense_model.eval()
        print('[Verification]model load: ', model_path)

    def _prepare_models(self):
        all_models = self.independent_models + self.surrogate_models
        all_labels = [0] * len(self.independent_models) + [1] * len(self.surrogate_models)  # 0 = neg, 1 = pos

        combined = list(zip(all_models, all_labels))
        random.shuffle(combined)
        self.suspicious_models, self.labels = zip(*combined)

    def _infer_signature(self, level: str):
        assert level in {"label"}, f"Unsupported level: {level}"

        signature_index = self.signature
        x, edge_index = self.wm_data.x, self.wm_data.edge_index

        # Extract subgraph
        sig_edge_index, _ = subgraph(signature_index, edge_index, relabel_nodes=True)
        sig_x = x[signature_index]

        with torch.no_grad():
            self.defense_model.eval()
            logits, O = self.defense_model(sig_x, sig_edge_index)
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
        assert level in {"label"}, f"Unsupported level: {level}"

        x, edge_index = self.wm_data.x, self.wm_data.edge_index
        wm_mask = self.wm_data.wm_mask

        # with torch.no_grad():
        #     self.defense_model.eval()
        #     logits, O = self.defense_model(x, edge_index)
        #     self.target_pred = logits[wm_mask].argmax(dim=1).cpu()
        self.target_pred = self.wm_data.y[wm_mask].cpu()

        self.model_outputs = []
        for model in self.suspicious_models:
            model.eval()
            with torch.no_grad():
                logits, O = model(x, edge_index)
                pred = logits[wm_mask].argmax(dim=1).cpu()
                self.model_outputs.append(pred)

    def _compute_metric(self, mode='label', plot_path=None):
        """
        mode = 'label' or 'embedding'
        """
        assert mode in {"label"}
        metric = ARUC(tau=0.5, r=100)
        metric.init_target_pred(self.target_pred)

        for pred, label in zip(self.model_outputs, self.labels):
            metric.update(pred, sample_label=label)
        print('[DEBUG]metric:', metric)
        print('[DEBUG]metric target pred:', metric.target_pred.shape)
        print('[DEBUG]metric pos pred:', len(metric.pos_samples))
        print('[DEBUG]metric neg pred:', len(metric.neg_samples))
        res_aruc, R, U, threshold = metric.compute(plot_path)
        res_asr = metric.compute_asr()
        print(f"[{mode.upper()}]Verification ARUC result: {res_aruc}, ASR: {res_asr}")
        return res_aruc, R, U, res_asr, threshold

    def verify(self, level='label', plot_path=None):
        """
        1. concat suspicious models, with related labels, then shuffle
        2. using signature inference target_model to get ground truth
        3. using signature inference all suspicious models to get output
        4. apply metric class
        5. emb level: calc wasserstein distance; label level: calc matching score
        """
        self._prepare_models()
        # TODO wm we use all
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

    def _compute_acc(self, threshold):
        """
        Predict whether a sample is positive (same as target) based on threshold,
        and compute classification accuracy compared to ground-truth label.
        """
        correct = 0
        total = 0
        print('[DEBUG]model outputs:', len(self.model_outputs), self.target_pred.shape)
        for pred, label in zip(self.model_outputs, self.labels):
            print('[DEBUG] pred: ', pred, 'label: ', label, 'target: ', self.target_pred)
            score = self._match_label(pred, self.target_pred)
            if label == 1:
                total += 1
                if score > threshold:
                    correct += 1
        acc = correct / total if total > 0 else 0
        print(f'[DEBUG]Threshold: {threshold}, Accuracy: {acc}, Correct: {correct}, Total: {total}')
        return acc

    def accuracy(self, threshold, level='label'):
        self._prepare_models()
        self._infer_signature_all(level)
        acc = self._compute_acc(threshold)
        return acc
