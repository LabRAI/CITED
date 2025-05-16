import os
import random

import torch
from tqdm import tqdm

from models.factory import generate_model_variants
from pipline.attack import GNNStealingPipeline
from pipline.target import TargetPipeline
from utils.dataset import IndependentDataset


class IndependentFactory:
    def __init__(self, target_model, dataset_name, variant_num, device='cpu'):
        self.target_model = target_model.to(device)
        # self.target_data = target_data.to(device)
        self.device = device
        self.dataset_name = dataset_name
        self.variant_num = variant_num
        self.independent_models = self._generate_independent_variants()

    def _generate_independent_variants(self):
        independent_models = generate_model_variants(base_model=self.target_model, n_variants=self.variant_num)
        independent_models = [model.to(self.device) for model in independent_models]
        return independent_models

    def train_independent(self, fixed_seed=42, lr=0.01, weight_decay=5e-4, epochs=200):
        """
        Automatically generate num_class_samples_list and seed_list with fixed randomness,
        then train all independent models accordingly.
        """
        torch.manual_seed(fixed_seed)
        random.seed(fixed_seed)

        n_models = len(self.independent_models)

        # Generate reproducible num_class_samples_list in range [10, 150]
        # random
        # num_class_samples_list = [random.randint(10, 150) for _ in range(n_models)]
        # fix 50 (same as surrogate)
        num_class_samples_list = [50] * n_models

        # Generate reproducible seed list from a large range
        seed_list = [random.randint(0, 100000) for _ in range(n_models)]

        self._train_all(
            num_class_samples_list=num_class_samples_list,
            seed_list=seed_list,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs
        )

    def _train_all(self, num_class_samples_list, seed_list, lr=0.01, weight_decay=5e-4, epochs=200):
        assert len(self.independent_models) == len(num_class_samples_list) == len(seed_list)

        trained_models = []
        pbar = tqdm(zip(self.independent_models, num_class_samples_list, seed_list),
                    total=len(self.independent_models),
                    desc='Training Independent Models')

        for idx, (model, n_sample, seed) in enumerate(pbar):
            dataset = IndependentDataset(self.dataset_name)
            data = dataset.generate(num_class_samples=n_sample, seed=seed)

            pipeline = TargetPipeline(self.target_model, data, device=self.device, lr=lr, weight_decay=weight_decay,
                                      epochs=epochs)
            pipeline.independent_once(model)

            trained_models.append(pipeline.independent_model)
            pbar.set_postfix(index=idx, num_class_samples=n_sample, seed=seed)

        self.independent_models = trained_models

    def _save_models(self, save_dir='saved_models'):
        # need to modify
        os.makedirs(save_dir, exist_ok=True)
        meta = []

        for idx, pipeline in enumerate(self.independent_models):
            model_path = os.path.join(save_dir, f"model_{idx}.pt")
            torch.save(pipeline.state_dict(), model_path)

            meta.append({
                'model_idx': idx,
                'num_class_samples': pipeline.data.train_mask.sum().item(),  # Approximate
                'seed': getattr(pipeline, 'seed', None)  # If stored
            })

        torch.save(meta, os.path.join(save_dir, "meta.pt"))

    def _load_models(self, save_dir='saved_models'):
        # need to modify
        meta = torch.load(os.path.join(save_dir, "meta.pt"))
        loaded_models = []

        for item in meta:
            model_path = os.path.join(save_dir, f"model_{item['model_idx']}.pt")
            model = self._recreate_model(item['model_idx'])
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            loaded_models.append(model)

        self.independent_models = loaded_models


class AttackFactory:
    def __init__(self, target_model, target_data, defense_name, level, variant_num, device='cpu'):
        self.target_model = target_model.to(device)
        self.target_data = target_data.to(device)
        self.defense_name = defense_name
        self.device = device
        self.level = level
        self.variant_num = variant_num
        self.surrogate_models = self._generate_attack_variants()
        self.attack_pipelines = []

    def _generate_attack_variants(self):
        surrogate_models = generate_model_variants(self.target_model, n_variants=self.variant_num)
        surrogate_models = [model.to(self.device) for model in surrogate_models]
        print('Generate variants', len(surrogate_models))
        return surrogate_models

    def train_surrogate(self, query_ratio, conf_threshold, lr=0.01, weight_decay=5e-4, fixed_seed=42):
        random.seed(fixed_seed)
        seed_list = [random.randint(0, 100000) for _ in range(self.variant_num)]

        self._train_all(seed_list, query_ratio, conf_threshold, lr, weight_decay)

    def _train_all(self, seed_list, query_ratio, conf_threshold, lr=0.01, weight_decay=5e-4):
        assert hasattr(self, 'surrogate_models') and self.surrogate_models, \
            "Call generate_attack_variants() before training."
        assert len(self.surrogate_models) == len(seed_list)

        surrogate_models = []
        pbar = tqdm(enumerate(zip(self.surrogate_models, seed_list)), total=len(self.surrogate_models),
                    desc="Training Surrogates")

        for idx, (model, seed) in pbar:
            pipeline = GNNStealingPipeline(self.target_model, self.target_data, defense_name=self.defense_name, lr=lr, weight_decay=weight_decay,
                                           level=self.level, device=self.device)
            pipeline.attack_factory(model, query_ratio=query_ratio, conf_threshold=conf_threshold, seed=seed)
            surrogate_models.append(pipeline.surrogate_model)
            pbar.set_postfix(index=idx, seed=seed)

        self.surrogate_models = surrogate_models
