import os

import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from tqdm import tqdm

from models.gcn import GCN


class AttackPipeline:
    def __init__(self):
        ...

    def _load_victim_model(self):
        ...

    def _save_model(self, state_dict):
        ...

    def _query_victim(self, query_mask):
        ...

    def _attack(self):
        ...


class GNNStealingPipeline(AttackPipeline):
    def __init__(self, model, data, defense_name, lr, weight_decay, level, device='cpu'):
        """
        1. load victim model
        2. query victim model and get response
        3. using query and response to train surrogate model
        """
        super().__init__()
        self.model = model.to(device)
        self.data = data.to(device)
        self.defense_name = defense_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.level = level
        self.device = device
        self._load_victim_model()

    def _load_victim_model(self):
        target_path = './output/defense'
        model_path = os.path.join(target_path, f'{self.data.name}_{self.model.name}_{self.defense_name}.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print('Load targe model: ', model_path)

    def _load_surrogate_model(self):
        target_path = './output/surrogate'
        model_path = os.path.join(target_path,
                                  f'{self.data.name}_{self.model.name}_{self.defense_name}_{self.level}.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.surrogate_model.load_state_dict(state_dict)
        self.surrogate_model.eval()
        print('Load surrogate model: ', model_path)

    def inference_surrogate(self):
        """
        inference on downstream tasks
        """
        self._load_surrogate_model()
        with torch.no_grad():
            x, edge_index = self.data.x, self.data.edge_index
            logits, _ = self.surrogate_model(x, edge_index)
            pred = logits.argmax(dim=1)
            test_acc = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).float().mean().item()
            print('[Surrogate]Test Accuracy: {:.4f}'.format(test_acc))

    def attack(self, query_ratio=0.5, conf_threshold=0.9, seed=42):
        # simulate prepare query
        query_mask = self.prepare_query_plus(query_ratio=query_ratio, conf_ratio=conf_threshold, seed=seed)
        # query_mask = self.prepare_query_random(query_ratio=1.0)
        # perform query
        # q_x, q_e, q_r = self._query_victim(query_mask)
        q_r = self._query_victim_all(query_mask)
        # train surrogate
        self._train_surrogate_all(query_mask, q_r)

    def attack_factory(self, surrogate_model, query_ratio=0.5, conf_threshold=0.94, seed=42):
        # TODO change to eval signature quality
        # TODO enable 'subgraph' or 'all graph'
        # subgraph
        # query_mask = self.prepare_query(query_ratio=query_ratio, conf_threshold=conf_threshold, seed=seed)
        # q_x, q_e, q_r = self._query_victim(query_mask)
        # self._train_surrogate_with_model(surrogate_model, q_x, q_e, q_r)
        # all graph
        query_mask = self.prepare_query_plus(query_ratio=query_ratio, conf_ratio=conf_threshold, seed=seed)
        q_r = self._query_victim_all(query_mask)
        self._train_surrogate_with_model_all(surrogate_model, query_mask, q_r)

    def prepare_query(self, query_ratio=0.5, conf_threshold=0.5, seed=42):
        """
        Prepare a query mask by selecting low-confidence nodes from unlabeled data.
        """
        x, edge_index = self.data.x, self.data.edge_index
        with torch.no_grad():
            logits, O = self.model(x, edge_index)
            soft_label = O['soft_label']  # shape: [N, C]

        # Step 1: Identify unused nodes
        used_mask = self.data.val_mask | self.data.test_mask
        unused_mask = ~used_mask
        unused_index = unused_mask.nonzero(as_tuple=True)[0]

        # Step 2: Compute max confidence per node
        confidence = soft_label.max(dim=1).values  # [N]
        # print('[DEBUG]soft-label: ', soft_label[:10])
        # print('[DEBUG]conf: ', confidence)
        low_conf_mask = (confidence < conf_threshold) & unused_mask
        low_conf_index = low_conf_mask.nonzero(as_tuple=True)[0]

        # Step 3: Determine target query size
        train_size = int(self.data.train_mask.sum().item())
        query_size = int(query_ratio * train_size)

        # Step 4: Sample query set
        generator = torch.Generator().manual_seed(seed)

        print(
            f'[DEBUG]Generate Query conf_threshold: {conf_threshold}, low conf num: {low_conf_index.size(0)}, low conf rate: {low_conf_index.size(0) / query_size}')
        if low_conf_index.size(0) >= query_size:
            selected_query_index = low_conf_index[
                torch.randperm(low_conf_index.size(0), generator=generator)[:query_size]]
        else:
            # Add remaining candidates from unused high-confidence nodes
            remaining_needed = query_size - low_conf_index.size(0)
            high_conf_unused = unused_mask.clone()
            high_conf_unused[low_conf_index] = False
            high_conf_index = high_conf_unused.nonzero(as_tuple=True)[0]

            combined_index = torch.cat([low_conf_index, high_conf_index], dim=0)
            if combined_index.size(0) >= query_size:
                selected_query_index = combined_index[
                    torch.randperm(combined_index.size(0), generator=generator)[:query_size]]
            else:
                # Not enough, use all unused nodes
                selected_query_index = unused_index

        # Build query mask
        query_mask = torch.zeros_like(unused_mask)
        query_mask[selected_query_index] = True

        print(f"[Query]Generated query_mask with {query_mask.sum().item()} nodes (target={query_size})")
        return query_mask

    def prepare_query_plus(self, query_ratio=0.5, conf_ratio=0.5, seed=42):
        """
        Prepare a query mask by selecting ambiguous nodes whose top-2 logits are close.
        The number of queries is query_ratio * |train_set|.
        conf_ratio determines the proportion selected based on top1-top2 closeness.
        The rest are randomly chosen from unused nodes.
        """
        x, edge_index = self.data.x, self.data.edge_index
        with torch.no_grad():
            logits, O = self.model(x, edge_index)
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
        query_size = int(query_ratio * train_size)
        conf_size = int(conf_ratio * query_size)
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

    def prepare_query_random(self, query_ratio=0.5, seed=42):
        """
        Randomly select query nodes from unlabeled set (excluding train/val/test).
        """
        # Get train/val/test masks
        used_mask = self.data.train_mask | self.data.val_mask | self.data.test_mask
        unused_mask = ~used_mask
        unused_index = unused_mask.nonzero(as_tuple=True)[0]

        # Determine query size
        train_size = int(self.data.train_mask.sum().item())
        query_size = int(query_ratio * train_size)

        generator = torch.Generator().manual_seed(seed)

        if unused_index.size(0) >= query_size:
            selected_query_index = unused_index[torch.randperm(unused_index.size(0), generator=generator)[:query_size]]
        else:
            selected_query_index = unused_index  # all available

        # Build query mask
        query_mask = torch.zeros_like(unused_mask)
        query_mask[selected_query_index] = True

        print(f"[Query-Random] Generated query_mask with {query_mask.sum().item()} nodes (target={query_size})")
        return query_mask

    def _query_victim(self, query_mask):
        """
        Query victim model to obtain output for all nodes
        """
        with torch.no_grad():
            x, edge_index = self.data.x, self.data.edge_index
            query_index = query_mask.nonzero(as_tuple=True)[0]

            # Extract subgraph
            query_edge_index, _ = subgraph(query_index, edge_index, relabel_nodes=True)
            query_x = x[query_index]

            # Map model input to subgraph
            logits, O = self.model(query_x, query_edge_index)

            if self.level == 'embedding':
                query_response = O['embedding']  # [N, D]
            elif self.level == 'soft_label':
                # query_response = O['soft_label']  # [N, C]
                raise ValueError('soft label not supported any more')
            elif self.level == 'hard_label':
                # query_response = O['hard_label']  # [N]
                # using distillation
                # query_response = logits  # [N, C]
                raise ValueError('hard label not supported any more')
            elif self.level == 'label':
                query_response = logits  # [N, C]
            else:
                raise ValueError(f"Unknown level: {self.level}")

            print(f"Queried victim at {self.level} level.")
            # print('q x: ', query_x.shape, 'q resp: ', query_response.shape)
        return query_x, query_edge_index, query_response

    def _query_victim_all(self, query_mask):
        """
        Query victim model to obtain output for all nodes
        """
        with torch.no_grad():
            x, edge_index = self.data.x, self.data.edge_index

            # Map model input to subgraph
            logits, O = self.model(x, edge_index)

            if self.level == 'embedding':
                query_response = O['embedding'][query_mask]  # [N, D]
            elif self.level == 'soft_label':
                # query_response = O['soft_label']  # [N, C]
                raise ValueError('soft label not supported any more')
            elif self.level == 'hard_label':
                # query_response = O['hard_label']  # [N]
                # using distillation
                # query_response = logits[query_mask]  # [N, C]
                raise ValueError('hard label not supported any more')
            elif self.level == 'label':
                query_response = logits[query_mask]  # [N, C]
            else:
                raise ValueError(f"Unknown level: {self.level}")

            print(f"Queried victim at {self.level} level.")
            # print('q x: ', query_x.shape, 'q resp: ', query_response.shape)
        return query_response

    def _train_surrogate(self, query_x, query_edge_index, query_response):
        """
        Train surrogate model using query input and victim response
        """
        surrogate = GCN(in_feats=self.data.num_features, out_feats=self.data.num_classes, hidden_dim=128).to(
            self.device)
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        surrogate.train()
        pbar = tqdm(range(200), desc='Train Surrogate')
        for epoch in pbar:
            optimizer.zero_grad()
            logits, O = surrogate(query_x, query_edge_index)

            if self.level == 'embedding':
                loss = F.mse_loss(O['embedding'], query_response)
            elif self.level == 'soft_label':
                # loss = F.cross_entropy(logits, query_response.argmax(dim=1))
                raise ValueError('soft label not supported any more')
            elif self.level == 'label':
                # >>>vanilla
                # loss = F.cross_entropy(logits, query_response)
                # >>>using distillation option 1
                # student_log_probs = F.log_softmax(logits, dim=1)  # [B, C]
                # teacher_probs = F.softmax(query_response, dim=1)  # [B, C]
                # loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                # >>>using distillation option 2
                T = 4.0
                lambda_ = 0.5
                # 1. distillation loss
                student_log_probs = F.log_softmax(logits / T, dim=1)
                teacher_probs = F.softmax(query_response / T, dim=1)
                loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
                # 2. standard classification loss
                loss_ce = F.cross_entropy(logits, query_response.argmax(dim=1))
                # 3. total loss
                loss = lambda_ * loss_kd + (1 - lambda_) * loss_ce
            else:
                raise NotImplementedError

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'],
                             weight_decay=optimizer.param_groups[0]['weight_decay'], epoch=epoch)

        self.surrogate_model = surrogate
        print("Surrogate model training complete.")
        self._save_model(self.surrogate_model.state_dict())

    def _train_surrogate_all(self, query_mask, query_response):
        """
        Train surrogate model using query input and victim response
        """
        surrogate = GCN(in_feats=self.data.num_features, out_feats=self.data.num_classes, hidden_dim=128).to(
            self.device)
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        surrogate.train()
        pbar = tqdm(range(200), desc='Train Surrogate')
        for epoch in pbar:
            optimizer.zero_grad()
            logits, O = surrogate(self.data.x, self.data.edge_index)

            if self.level == 'embedding':
                loss = F.mse_loss(O['embedding'][query_mask], query_response)
            elif self.level == 'soft_label':
                # loss = F.cross_entropy(logits, query_response.argmax(dim=1))
                raise ValueError('soft label not supported any more')
            elif self.level == 'label':
                T = 4.0
                lambda_ = 0.5
                # 1. distillation loss
                student_log_probs = F.log_softmax(logits[query_mask] / T, dim=1)
                teacher_probs = F.softmax(query_response / T, dim=1)
                loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
                # 2. standard classification loss
                loss_ce = F.cross_entropy(logits[query_mask], query_response.argmax(dim=1))
                # 3. total loss
                loss = lambda_ * loss_kd + (1 - lambda_) * loss_ce
            else:
                raise NotImplementedError

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'],
                             weight_decay=optimizer.param_groups[0]['weight_decay'], epoch=epoch)

        self.surrogate_model = surrogate
        print("Surrogate model training complete.")
        self._save_model(self.surrogate_model.state_dict())

    def _train_surrogate_with_model(self, surrogate_model, query_x, query_edge_index, query_response):
        """
        Train surrogate model using query input and victim response
        """
        surrogate_model.reset_parameters()
        optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        print(f'[DEBUG]Attack training size: {query_x.shape}, {query_edge_index.shape}')

        pbar = tqdm(range(200), desc='Train Surrogate')
        for epoch in pbar:
            surrogate_model.train()
            optimizer.zero_grad()
            logits, O = surrogate_model(query_x, query_edge_index)

            if self.level == 'embedding':
                loss = F.mse_loss(O[self.level], query_response)
            elif self.level == 'soft_label':
                # loss = F.cross_entropy(logits, query_response.argmax(dim=1))
                raise ValueError('soft label not supported any more')
            elif self.level == 'label':
                # >>>vanilla
                # loss = F.cross_entropy(logits, query_response.argmax(dim=1))
                # >>>using distillation option 1
                # student_log_probs = F.log_softmax(logits, dim=1)  # [B, C]
                # teacher_probs = F.softmax(query_response, dim=1)  # [B, C]
                # loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                # >>>using distillation option 2
                T = 4.0
                lambda_ = 0.5
                # 1. distillation loss
                student_log_probs = F.log_softmax(logits / T, dim=1)
                teacher_probs = F.softmax(query_response / T, dim=1)
                loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
                # 2. standard classification loss
                loss_ce = F.cross_entropy(logits, query_response.argmax(dim=1))
                # 3. total loss
                loss = lambda_ * loss_kd + (1 - lambda_) * loss_ce
            else:
                raise NotImplementedError

            loss.backward()
            optimizer.step()
            # print(f'Epoch {epoch}, loss: {loss.item()}')
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

        self.surrogate_model = surrogate_model
        print("One surrogate model training complete.")

    def _train_surrogate_with_model_all(self, surrogate_model, query_mask, query_response):
        """
        Train surrogate model using query input and victim response
        """
        surrogate_model.reset_parameters()
        optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        pbar = tqdm(range(200), desc='Train Surrogate')
        for epoch in pbar:
            surrogate_model.train()
            optimizer.zero_grad()
            logits, O = surrogate_model(self.data.x, self.data.edge_index)

            if self.level == 'embedding':
                loss = F.mse_loss(O[self.level][query_mask], query_response)
            elif self.level == 'soft_label':
                # loss = F.cross_entropy(logits, query_response.argmax(dim=1))
                raise ValueError('soft label not supported any more')
            elif self.level == 'label':
                T = 4.0
                lambda_ = 0.5
                # 1. distillation loss
                student_log_probs = F.log_softmax(logits[query_mask] / T, dim=1)
                teacher_probs = F.softmax(query_response / T, dim=1)
                loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
                # 2. standard classification loss
                loss_ce = F.cross_entropy(logits[query_mask], query_response.argmax(dim=1))
                # 3. total loss
                loss = lambda_ * loss_kd + (1 - lambda_) * loss_ce
            else:
                raise NotImplementedError

            loss.backward()
            optimizer.step()
            # print(f'Epoch {epoch}, loss: {loss.item()}')
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

        self.surrogate_model = surrogate_model
        print("One surrogate model training complete.")

    def _save_model(self, state_dict):
        output_dir = './output/surrogate/'
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir,
                                  f'{self.data.name}_{self.model.name}_{self.defense_name}_{self.level}.pth')
        torch.save(state_dict, model_path)
        print(f"Surrogate model saved to {model_path}")
