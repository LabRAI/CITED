import copy
import os
import random

import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm


class DefensePipeline:
    def __init__(self):
        ...

    def _load_target(self):
        ...

    def defense(self):
        ...


class RandomWMPipeline(DefensePipeline):
    def __init__(self, target_model, data, level='N/A', device='cpu'):
        super().__init__()
        self.target_model = target_model.to(device)
        self.data = data.to(device)
        self.level = level
        self.device = device
        self._load_model()

    def _load_model(self):
        target_path = './output/target'
        model_path = os.path.join(target_path, f'{self.data.name}_{self.target_model.name}.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.target_model.load_state_dict(state_dict)
        self.target_model.eval()
        print('model load: ', model_path)

    def _generate_trigger_graph(self, num_nodes=10, edge_prob=0.1, p_feat=0.1):
        """
        generate random watermark data
        """
        feat_dim = self.data.x.size(1)
        num_classes = int(self.data.y.max().item()) + 1

        G = nx.erdos_renyi_graph(num_nodes, edge_prob)
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        x = torch.zeros((num_nodes, feat_dim))
        for i in range(num_nodes):
            ones_idx = torch.randperm(feat_dim)[:int(p_feat * feat_dim)]
            x[i, ones_idx] = 1
        y = torch.tensor([random.randint(0, num_classes - 1) for _ in range(num_nodes)], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)

    def _combine_graphs(self, trigger_data):
        """
        combine original data and wm data
        """
        wm_data = copy.deepcopy(self.data)
        trigger_data.to(self.device)

        offset = wm_data.x.size(0)
        wm_data.x = torch.cat([wm_data.x, trigger_data.x], dim=0)
        wm_data.edge_index = torch.cat([wm_data.edge_index, trigger_data.edge_index + offset], dim=1)
        wm_data.y = torch.cat([wm_data.y, trigger_data.y], dim=0)

        wm_data.train_mask = torch.cat([
            wm_data.train_mask,
            torch.ones(trigger_data.num_nodes, dtype=torch.bool, device=self.device)
        ])
        wm_data.val_mask = torch.cat([
            wm_data.val_mask,
            torch.zeros(trigger_data.num_nodes, dtype=torch.bool, device=self.device)
        ])
        wm_data.test_mask = torch.cat([
            wm_data.test_mask,
            torch.zeros(trigger_data.num_nodes, dtype=torch.bool, device=self.device)
        ])

        wm_mask = torch.zeros(wm_data.x.size(0), dtype=torch.bool, device=self.device)
        wm_mask[-trigger_data.num_nodes:] = True
        wm_data.wm_mask = wm_mask

        return wm_data

    def embed_watermark_trigger(self, random_node_num=10, random_edge_prob=0.1, random_feat_ratio=0.1):
        trigger_data = self._generate_trigger_graph(random_node_num, random_edge_prob, random_feat_ratio)
        watermarked_data = self._combine_graphs(trigger_data)
        return watermarked_data

    def finetune_on_watermarked_data(self, watermarked_data, epochs=100, lr=0.01, weight_decay=5e-4):
        self.target_model.train()
        optimizer = torch.optim.Adam(self.target_model.parameters(), lr=lr, weight_decay=weight_decay)

        x, edge_index, y = watermarked_data.x, watermarked_data.edge_index, watermarked_data.y
        train_mask = watermarked_data.train_mask
        test_mask = watermarked_data.test_mask
        wm_mask = watermarked_data.wm_mask

        pbar = tqdm(range(epochs), desc="Fine-tuning defense model")
        for epoch in pbar:
            optimizer.zero_grad()
            out, _ = self.target_model(x, edge_index)
            loss = F.cross_entropy(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(epoch=epoch, loss=loss.item())

        self.target_model.eval()
        with torch.no_grad():
            logits, _ = self.target_model(x, edge_index)
            pred = logits[test_mask].argmax(dim=1)
            acc_test = (pred == y[test_mask]).float().mean().item()
            print(f'[Defense]Test Acc: {acc_test:.4f}')

            pred = logits[wm_mask].argmax(dim=1)
            acc_wm = (pred == y[wm_mask]).float().mean().item()
            print(f'[Defense]WM Acc: {acc_wm:.4f}')

        self._save_model()

    def _save_model(self):
        os.makedirs('./output/defense', exist_ok=True)
        save_path = f'./output/defense/{self.data.name}_{self.target_model.name}_RandomWM.pth'
        torch.save(self.target_model.state_dict(), save_path)
        print(f'Defense Model saved to {save_path}')


class BackdoorWMPipeline(DefensePipeline):
    def __init__(self, target_model, data, level='N/A', device='cpu'):
        super().__init__()
        self.target_model = target_model.to(device)
        self.data = data.to(device)
        self.level = level
        self.device = device
        self._load_model()

    def _load_model(self):
        target_path = './output/target'
        model_path = os.path.join(target_path, f'{self.data.name}_{self.target_model.name}.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.target_model.load_state_dict(state_dict)
        self.target_model.eval()
        print('model load: ', model_path)

    def _inject_backdoor_trigger(self, backdoor_ratio=0.01, backdoor_len=20):
        backdoor_data = copy.deepcopy(self.data)

        num_nodes = backdoor_data.x.size(0)
        num_feats = backdoor_data.num_features
        num_classes = backdoor_data.num_classes

        train_indices = backdoor_data.train_mask.nonzero(as_tuple=True)[0]
        num_train_nodes = train_indices.size(0)
        num_trigger_nodes = int(backdoor_ratio * num_train_nodes)

        trigger_nodes = random.sample(train_indices.tolist(), min(num_trigger_nodes, num_train_nodes))

        for node_index in trigger_nodes:
            feat_val = random.uniform(0, 1)
            feature_indices = random.sample(range(num_feats), backdoor_len)
            backdoor_data.x[node_index][feature_indices] = feat_val
            random_label = random.randint(0, num_classes - 1)
            backdoor_data.y[node_index] = random_label

        backdoor_mask = torch.zeros(num_nodes, dtype=torch.bool)
        backdoor_mask[trigger_nodes] = True
        backdoor_data.wm_mask = backdoor_mask

        return backdoor_data

    def _generate_backdoor_trigger(self, backdoor_ratio=0.01, backdoor_len=20):
        """
        Generate an ER random graph as backdoor trigger, inject feature pattern and random labels.
        """
        feat_dim = self.data.x.size(1)
        num_classes = int(self.data.y.max().item()) + 1

        train_indices = self.data.train_mask.nonzero(as_tuple=True)[0]
        num_train_nodes = train_indices.size(0)
        num_trigger_nodes = max(1, int(backdoor_ratio * num_train_nodes))

        G = nx.erdos_renyi_graph(n=num_trigger_nodes, p=0.1)
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        x = torch.zeros((num_trigger_nodes, feat_dim))
        for i in range(num_trigger_nodes):
            feat_val = random.uniform(0, 1)
            feature_indices = random.sample(range(feat_dim), min(backdoor_len, feat_dim))
            x[i, feature_indices] = feat_val

        y = torch.tensor([random.randint(0, num_classes - 1) for _ in range(num_trigger_nodes)], dtype=torch.long)

        backdoor_mask = torch.ones(num_trigger_nodes, dtype=torch.bool)
        backdoor_data = Data(x=x, edge_index=edge_index, y=y, wm_mask=backdoor_mask)

        return backdoor_data

    def _combine_graphs(self, backdoor_data):
        """
        combine original data and wm data
        """
        wm_data = copy.deepcopy(self.data)
        backdoor_data.to(self.device)

        offset = wm_data.x.size(0)
        wm_data.x = torch.cat([wm_data.x, backdoor_data.x], dim=0)
        wm_data.edge_index = torch.cat([wm_data.edge_index, backdoor_data.edge_index + offset], dim=1)
        wm_data.y = torch.cat([wm_data.y, backdoor_data.y], dim=0)

        wm_data.train_mask = torch.cat([
            wm_data.train_mask,
            torch.ones(backdoor_data.num_nodes, dtype=torch.bool, device=self.device)
        ])
        wm_data.val_mask = torch.cat([
            wm_data.val_mask,
            torch.zeros(backdoor_data.num_nodes, dtype=torch.bool, device=self.device)
        ])
        wm_data.test_mask = torch.cat([
            wm_data.test_mask,
            torch.zeros(backdoor_data.num_nodes, dtype=torch.bool, device=self.device)
        ])

        wm_mask = torch.zeros(wm_data.x.size(0), dtype=torch.bool, device=self.device)
        wm_mask[-backdoor_data.num_nodes:] = True
        wm_data.wm_mask = wm_mask

        return wm_data

    def embed_backdoor(self, backdoor_ratio, backdoor_len):
        watermarked_data = self._inject_backdoor_trigger(backdoor_ratio=backdoor_ratio, backdoor_len=backdoor_len)
        # backdoor_data = self._generate_backdoor_trigger(backdoor_ratio=backdoor_ratio, backdoor_len=backdoor_len)
        # watermarked_data = self._combine_graphs(backdoor_data)
        print('[Defense]>>>>>>>>>>>>>>>Backdoor-Inject>>>>>>>>>>>>>>>')
        return watermarked_data

    def finetune_on_backdoor_data(self, backdoor_data, epochs=50, lr=0.01, weight_decay=5e-4):
        self.target_model.train()
        optimizer = torch.optim.Adam(self.target_model.parameters(), lr=lr, weight_decay=weight_decay)

        x, edge_index, y = backdoor_data.x, backdoor_data.edge_index, backdoor_data.y
        train_mask = backdoor_data.train_mask
        test_mask = backdoor_data.test_mask
        backdoor_mask = backdoor_data.wm_mask

        pbar = tqdm(range(epochs), desc="Fine-tuning defense model")
        for epoch in pbar:
            optimizer.zero_grad()
            out, _ = self.target_model(x, edge_index)
            loss = F.cross_entropy(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(epoch=epoch, loss=loss.item())

        self.target_model.eval()
        with torch.no_grad():
            logits, _ = self.target_model(x, edge_index)
            pred = logits[test_mask].argmax(dim=1)
            acc_test = (pred == y[test_mask]).float().mean().item()
            print(f'[Defense]>>>>>>>>>>>>>>>>>>>>>Test Acc: {acc_test:.4f}')

            pred = logits[backdoor_mask].argmax(dim=1)
            acc_wm = (pred == y[backdoor_mask]).float().mean().item()
            print(f'[Defense]>>>>>>>>>>>>>>>>>>>>>WM Acc: {acc_wm:.4f}')

        self._save_model()

    def _save_model(self):
        os.makedirs('./output/defense', exist_ok=True)
        save_path = f'./output/defense/{self.data.name}_{self.target_model.name}_BackdoorWM.pth'
        torch.save(self.target_model.state_dict(), save_path)
        print(f'Defense Model saved to {save_path}')


class SurviveWMPipeline(DefensePipeline):
    def __init__(self, target_model, data, level='N/A', device='cpu'):
        super().__init__()
        self.target_model = target_model.to(device)
        self.data = data.to(device)
        self.level = level
        self.device = device
        self._load_model()

    def _load_model(self):
        target_path = './output/target'
        model_path = os.path.join(target_path, f'{self.data.name}_{self.target_model.name}.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.target_model.load_state_dict(state_dict)
        self.target_model.eval()
        print('model load: ', model_path)

    def _snn_loss(self, x, y, T=0.5):
        x = F.normalize(x, p=2, dim=1)
        dist_matrix = torch.cdist(x, x, p=2) ** 2
        eye = torch.eye(len(x), device=x.device).bool()
        sim = torch.exp(-dist_matrix / T)
        mask_same = y.unsqueeze(1) == y.unsqueeze(0)
        sim = sim.masked_fill(eye, 0)
        denom = sim.sum(1)
        nom = (sim * mask_same.float()).sum(1)
        loss = -torch.log(nom / (denom + 1e-10) + 1e-10).mean()
        return loss

    def _generate_key_graph(self, num_nodes=10, edge_prob=0.3):
        trigger = nx.erdos_renyi_graph(num_nodes, edge_prob)
        edge_index = torch.tensor(list(trigger.edges), dtype=torch.long).t().contiguous()
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        x = torch.randn((num_nodes, self.data.num_features)) * 0.1
        label = torch.randint(0, self.data.num_classes, (num_nodes,))
        return Data(x=x, edge_index=edge_index, y=label)

    def _combine_with_trigger(self, key_graph):
        wm_data = copy.deepcopy(self.data)
        key_graph.to(self.device)

        offset = wm_data.x.size(0)

        wm_data.x = torch.cat([wm_data.x, key_graph.x], dim=0)
        wm_data.edge_index = torch.cat([wm_data.edge_index, key_graph.edge_index + offset], dim=1)
        wm_data.y = torch.cat([wm_data.y, key_graph.y], dim=0)

        wm_data.train_mask = torch.cat([
            wm_data.train_mask,
            torch.ones(key_graph.num_nodes, dtype=torch.bool, device=self.device)
        ])
        wm_data.val_mask = torch.cat([
            wm_data.val_mask,
            torch.zeros(key_graph.num_nodes, dtype=torch.bool, device=self.device)
        ])
        wm_data.test_mask = torch.cat([
            wm_data.test_mask,
            torch.zeros(key_graph.num_nodes, dtype=torch.bool, device=self.device)
        ])

        wm_mask = torch.cat([
            torch.zeros(offset, dtype=torch.bool, device=self.device),
            torch.ones(key_graph.num_nodes, dtype=torch.bool, device=self.device)
        ])
        wm_data.wm_mask = wm_mask

        return wm_data

    def embed_wm(self, survive_node_num=10, survive_edge_prob=0.3):
        key_graph = self._generate_key_graph(survive_node_num, survive_edge_prob)
        wm_data = self._combine_with_trigger(key_graph)
        return wm_data

    def finetune_on_wm_data(self, wm_data, epochs=50, lr=0.01, weight_decay=5e-4):
        self.target_model.train()
        optimizer = torch.optim.Adam(self.target_model.parameters(), lr=lr, weight_decay=weight_decay)

        x, edge_index, y = wm_data.x, wm_data.edge_index, wm_data.y
        train_mask = wm_data.train_mask
        test_mask = wm_data.test_mask
        backdoor_mask = wm_data.wm_mask

        pbar = tqdm(range(epochs), desc="Fine-tuning defense model")
        for epoch in pbar:
            optimizer.zero_grad()
            out, _ = self.target_model(x, edge_index)
            loss_ce = F.cross_entropy(out[train_mask], y[train_mask])
            snnl = self._snn_loss(out[train_mask], y[train_mask], T=0.5)
            loss = loss_ce - 0.1 * snnl
            loss.backward()
            optimizer.step()
            pbar.set_postfix(epoch=epoch, loss=loss.item())

        self.target_model.eval()
        with torch.no_grad():
            logits, _ = self.target_model(x, edge_index)
            pred = logits[test_mask].argmax(dim=1)
            acc_test = (pred == y[test_mask]).float().mean().item()
            print(f'[Defense]Test Acc: {acc_test:.4f}')

            pred = logits[backdoor_mask].argmax(dim=1)
            acc_wm = (pred == y[backdoor_mask]).float().mean().item()
            print(f'[Defense]WM Acc: {acc_wm:.4f}')

        self._save_model()

    def _save_model(self):
        os.makedirs('./output/defense', exist_ok=True)
        save_path = f'./output/defense/{self.data.name}_{self.target_model.name}_SurviveWM.pth'
        torch.save(self.target_model.state_dict(), save_path)
        print(f'Defense Model saved to {save_path}')
