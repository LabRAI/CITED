import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import subgraph

class TargetPipeline:
    def __init__(self, model, data, device='cpu', lr=0.01, weight_decay=5e-4, epochs=200):
        """
        Args:
            model_class: GNN model class (e.g., GCN)
            model_kwargs: dict, arguments for initializing the model
            data: PyG dataset object (must have data.x, data.edge_index, data.train_mask, etc.)
            device: 'cpu' or 'cuda'
            lr: learning rate
            weight_decay: weight decay for optimizer
            epochs: number of training epochs
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

    def train_once(self):
        self.model.reset_parameters()
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        pbar = tqdm(range(self.epochs), desc='Once')
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()
            logits, _ = model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(logits[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'], epoch=epoch)

        # Evaluate after training
        model.eval()
        with torch.no_grad():
            logits, _ = model(self.data.x, self.data.edge_index)
            pred = logits.argmax(dim=1)

            train_acc = (pred[self.data.train_mask] == self.data.y[self.data.train_mask]).float().mean().item()
            val_acc = (pred[self.data.val_mask] == self.data.y[self.data.val_mask]).float().mean().item()
            test_acc = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).float().mean().item()

        return train_acc, val_acc, test_acc

    def independent_once(self, independent_model):
        # TODO enable 'subgraph' or 'all graph'
        # self._train_independent_with_model(independent_model)
        self._train_independent_with_model_all(independent_model)

    def _train_independent_with_model(self, independent_model):
        independent_model.reset_parameters()
        optimizer = torch.optim.Adam(independent_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        x, edge_index = self.data.x, self.data.edge_index
        independent_index = self.data.train_mask.nonzero(as_tuple=True)[0]
        # Extract subgraph
        independent_edge_index, _ = subgraph(independent_index, edge_index, relabel_nodes=True)
        independent_x = x[independent_index]
        print(f'[DEBUG]Target training size: {independent_x.shape}, {independent_edge_index.shape}')
        pbar = tqdm(range(self.epochs), desc='Train Independent')
        for epoch in pbar:
            independent_model.train()
            optimizer.zero_grad()
            logits, _ = independent_model(independent_x, independent_edge_index)
            loss = F.cross_entropy(logits, self.data.y[independent_index])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'],
                             weight_decay=optimizer.param_groups[0]['weight_decay'], epoch=epoch)
        # Evaluate after training
        independent_model.eval()
        with torch.no_grad():
            logits, _ = independent_model(self.data.x, self.data.edge_index)
            pred = logits.argmax(dim=1)

            train_acc = (pred[self.data.train_mask] == self.data.y[self.data.train_mask]).float().mean().item()
            val_acc = (pred[self.data.val_mask] == self.data.y[self.data.val_mask]).float().mean().item()
            test_acc = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).float().mean().item()
            print(f"Independent train acc: {train_acc}, val acc: {val_acc}, test acc: {test_acc}")
        self.independent_model = independent_model
        print('One independent model training complete.')

    def _train_independent_with_model_all(self, independent_model):
        independent_model.reset_parameters()
        optimizer = torch.optim.Adam(independent_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        pbar = tqdm(range(self.epochs), desc='Train Independent')
        for epoch in pbar:
            independent_model.train()
            optimizer.zero_grad()
            logits, _ = independent_model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(logits[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'],
                             weight_decay=optimizer.param_groups[0]['weight_decay'], epoch=epoch)
        # Evaluate after training
        independent_model.eval()
        with torch.no_grad():
            logits, _ = independent_model(self.data.x, self.data.edge_index)
            pred = logits.argmax(dim=1)

            train_acc = (pred[self.data.train_mask] == self.data.y[self.data.train_mask]).float().mean().item()
            val_acc = (pred[self.data.val_mask] == self.data.y[self.data.val_mask]).float().mean().item()
            test_acc = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).float().mean().item()
            print(f"Independent train acc: {train_acc}, val acc: {val_acc}, test acc: {test_acc}")
        self.independent_model = independent_model
        print('One independent model training complete.')

    def run(self, runs=3):
        train_accs = []
        val_accs = []
        test_accs = []

        best_test_acc = -1
        best_model_state = None

        for _ in tqdm(range(runs), desc='Runs'):
            train_acc, val_acc, test_acc = self.train_once()
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        train_accs = np.array(train_accs)
        val_accs = np.array(val_accs)
        test_accs = np.array(test_accs)

        self._save_model(best_model_state)

        print(f"Train Acc: {train_accs.mean():.4f} ± {train_accs.std():.4f}")
        print(f"Val Acc:   {val_accs.mean():.4f} ± {val_accs.std():.4f}")
        print(f"Test Acc:  {test_accs.mean():.4f} ± {test_accs.std():.4f}")

    def _save_model(self, state_dict):
        output_dir = './output/target/'
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, f'{self.data.name}_{self.model.name}.pth')
        torch.save(state_dict, model_path)
        print(f"Best model saved to {model_path}")
