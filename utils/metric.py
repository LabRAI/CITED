import torch


class MetricBase:
    def __init__(self):
        pass

    def init_target_pred(self, target_pred):
        raise NotImplementedError

    def update(self, pred, sample_label):
        raise NotImplementedError


class WARUC(MetricBase):
    def __init__(self, tau=0.5, r=100):
        super().__init__()
        self.name = 'WARUC'
        self.tau = tau
        self.r = r
        self.target_pred = None
        self.pos_samples = []
        self.neg_samples = []

    def init_target_pred(self, target_pred):
        self.target_pred = target_pred  # numpy array [N, D]

    def update(self, pred, sample_label):
        assert sample_label in {0, 1}
        if sample_label == 1:
            self.pos_samples.append(pred)
        else:
            self.neg_samples.append(pred)

    def compute(self, plot_path=None):
        pos_dists = np.array([self._avg_l2_to_target(p, self.target_pred) for p in self.pos_samples])
        neg_dists = np.array([self._avg_l2_to_target(n, self.target_pred) for n in self.neg_samples])

        pos_norm, neg_norm = self._normalize(pos_dists, neg_dists)

        thresholds = np.linspace(1 / self.r, 1.0, self.r)  # fix here
        R, U = [], []
        for tau in thresholds:
            R.append(np.mean(pos_norm <= tau))
            U.append(np.mean(neg_norm > tau))
        print('[DEBUG]pos dist: ', pos_dists, pos_norm, ', neg dist: ', neg_dists, neg_norm)
        waruc = np.mean([min(r, u) for r, u in zip(R, U)])
        thre_dist = self._get_thre_dist(pos_dists, neg_dists)
        return waruc, R, U, thre_dist

    def _get_thre_dist(self, pos_dists, neg_dists, r=100):
        all_dists = np.concatenate([pos_dists, neg_dists])
        thresholds = np.linspace(all_dists.min(), all_dists.max(), r)

        best_score = -1
        best_thre = None
        for tau in thresholds:
            R = np.mean(pos_dists <= tau)
            U = np.mean(neg_dists > tau)
            score = min(R, U)
            if score > best_score:
                best_score = score
                best_thre = tau
        return best_thre

    def compute_dsr(self):
        pos_dists = np.array([self._avg_l2_to_target(p, self.target_pred) for p in self.pos_samples])
        neg_dists = np.array([self._avg_l2_to_target(n, self.target_pred) for n in self.neg_samples])

        total = len(pos_dists) * len(neg_dists)
        count = 0
        for d in pos_dists:
            count += np.sum(d < neg_dists)

        dsr = count / total
        print(f"[DEBUG] DSR: {dsr:.4f}, pos<neg count: {count}/{total}")
        return dsr

    def _normalize(self, pos_dists, neg_dists):
        all_dists = np.concatenate([pos_dists, neg_dists])
        min_d, max_d = all_dists.min(), all_dists.max()
        scale = max_d - min_d + 1e-12
        pos_norm = (pos_dists - min_d) / scale
        neg_norm = (neg_dists - min_d) / scale
        return pos_norm, neg_norm

    def _avg_l2_to_target(self, sample, target):
        """
        Approximate Wasserstein-2 by average L2 distance between aligned points
        """
        return np.mean(np.linalg.norm(sample - target, axis=1))

    def _normalize_exp(self, pos_dists, neg_dists):
        all_dists = np.concatenate([pos_dists, neg_dists])
        min_d, max_d = all_dists.min(), all_dists.max()
        scale = max_d - min_d + 1e-12

        pos_scaled = (pos_dists - min_d) / scale
        neg_scaled = (neg_dists - min_d) / scale

        # Apply exponential stretching
        pos_norm = np.exp(pos_scaled) - 1
        neg_norm = np.exp(neg_scaled) - 1

        # Normalize again to [0,1]
        all_norm = np.concatenate([pos_norm, neg_norm])
        min_n, max_n = all_norm.min(), all_norm.max()
        scale_n = max_n - min_n + 1e-12

        pos_final = (pos_norm - min_n) / scale_n
        neg_final = (neg_norm - min_n) / scale_n
        return pos_final, neg_final


class ARUC(MetricBase):
    def __init__(self, tau=0.5, r=100):
        super().__init__()
        self.name = 'ARUC'
        self.tau = tau
        self.r = r
        self.target_pred = None
        self.pos_samples = []
        self.neg_samples = []

    def init_target_pred(self, target_pred):
        # [N,] int array
        if isinstance(target_pred, torch.Tensor):
            target_pred = target_pred.detach().cpu().numpy()
        self.target_pred = target_pred

    def update(self, pred, sample_label):
        assert sample_label in {0, 1}
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if sample_label == 1:
            self.pos_samples.append(pred)
        else:
            self.neg_samples.append(pred)

    def compute(self, plot_path=None):
        pos_scores = np.array([self._match_score(p, self.target_pred) for p in self.pos_samples])
        neg_scores = np.array([self._match_score(n, self.target_pred) for n in self.neg_samples])

        pos_norm, neg_norm = self._normalize(pos_scores, neg_scores)

        thresholds = np.linspace(1 / self.r, 1.0, self.r)
        R, U = [], []
        for tau in thresholds:
            R.append(np.mean(pos_norm >= tau))
            U.append(np.mean(neg_norm < tau))
        print('[DEBUG]pred target: ', self.target_pred)
        print('[DEBUG]pos score: ', pos_scores, pos_norm, ', neg score: ', neg_scores, neg_norm)
        aruc = np.mean([min(r, u) for r, u in zip(R, U)])
        thre_acc = self._get_thre_acc(pos_scores, neg_scores)
        return aruc, R, U, thre_acc

    def _get_thre_acc(self, pos_scores, neg_scores, r=100):
        all_dists = np.concatenate([pos_scores, neg_scores])
        thresholds = np.linspace(all_dists.min(), all_dists.max(), r)

        best_score = -1
        best_thre = None
        for tau in thresholds:
            R = np.mean(pos_scores <= tau)
            U = np.mean(neg_scores > tau)
            score = min(R, U)
            if score > best_score:
                best_score = score
                best_thre = tau
        return best_thre

    def compute_asr(self):
        pos_scores = np.array([self._match_score(p, self.target_pred) for p in self.pos_samples])
        neg_scores = np.array([self._match_score(n, self.target_pred) for n in self.neg_samples])

        total = len(pos_scores) * len(neg_scores)
        count = 0
        for p in pos_scores:
            count += np.sum(p > neg_scores)

        asr = count / total
        print(f"[DEBUG] ASR: {asr:.4f}, pos>neg count: {count}/{total}")
        return asr

    def _match_score(self, pred, target):
        """
        Compute fraction of exact matches
        """
        assert pred.shape == target.shape
        return np.mean(pred == target)

    def _normalize(self, pos_scores, neg_scores):
        all_dists = np.concatenate([pos_scores, neg_scores])
        min_d, max_d = all_dists.min(), all_dists.max()
        scale = max_d - min_d + 1e-12
        pos_norm = (pos_scores - min_d) / scale
        neg_norm = (neg_scores - min_d) / scale
        return pos_norm, neg_norm

    def _normalize_exp(self, pos_scores, neg_scores):
        all_dists = np.concatenate([pos_scores, neg_scores])
        min_d, max_d = all_dists.min(), all_dists.max()
        scale = max_d - min_d + 1e-12

        pos_scaled = (pos_scores - min_d) / scale
        neg_scaled = (neg_scores - min_d) / scale

        # Apply exponential stretching
        pos_norm = np.exp(pos_scaled) - 1
        neg_norm = np.exp(neg_scaled) - 1

        # Normalize again to [0,1]
        all_norm = np.concatenate([pos_norm, neg_norm])
        min_n, max_n = all_norm.min(), all_norm.max()
        scale_n = max_n - min_n + 1e-12

        pos_final = (pos_norm - min_n) / scale_n
        neg_final = (neg_norm - min_n) / scale_n
        return pos_final, neg_final


def test_waruc():
    np.random.seed(0)
    N, D = 10, 5
    target_pred = np.random.rand(N, D)
    pos_samples = [target_pred + 0.01 * np.random.randn(N, D) for _ in range(50)]
    neg_samples = [np.random.rand(N, D) for _ in range(50)]
    metric = WARUC()
    metric.init_target_pred(target_pred)
    for p in pos_samples:
        metric.update(p, sample_label=1)
    for n in neg_samples:
        metric.update(n, sample_label=0)
    result = metric.compute()
    print("W-ARUC:", result)


def test_aruc():
    global num_classes
    np.random.seed(42)
    N = 10  # number of signature nodes
    num_classes = 5
    # target label prediction
    target_pred = np.random.randint(0, num_classes, size=N)
    # Positive samples: mostly same
    pos_samples = []
    for _ in range(50):
        noise = np.random.rand(N) < 0.9  # 90% match
        pred = np.where(noise, target_pred, np.random.randint(0, num_classes, size=N))
        pos_samples.append(pred)
    # Negative samples: mostly different
    neg_samples = []
    for _ in range(50):
        noise = np.random.rand(N) < 0.2  # 20% match
        pred = np.where(noise, target_pred, np.random.randint(0, num_classes, size=N))
        neg_samples.append(pred)
    metric = ARUC(tau=0.5)
    metric.init_target_pred(target_pred)
    for p in pos_samples:
        metric.update(p, sample_label=1)
    for n in neg_samples:
        metric.update(n, sample_label=0)
    result = metric.compute()
    print("Label-Level ARUC:", result)


import matplotlib.pyplot as plt
import numpy as np


def plot_aruc(R, U, aruc, save_path=None):
    """
    R: list or array of robustness scores (length r)
    U: list or array of uniqueness scores (length r)
    aruc: float, the final ARUC score
    title: str, model name
    save_path: str or None, if specified, save the figure to that path
    """
    thresholds = np.linspace(0, 1, len(R))

    fig, ax1 = plt.subplots(figsize=(3, 2.2))

    # Primary y-axis: Robustness
    ax1.plot(thresholds, R, color='tab:red', label='Robustness')
    ax1.set_xlabel("threshold")
    ax1.set_ylabel("Robustness", color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Fill area under min(R, U)
    ax1.fill_between(thresholds, np.minimum(R, U), alpha=0.2, color='gray')

    # Secondary y-axis: Uniqueness
    ax2 = ax1.twinx()
    ax2.plot(thresholds, U, color='tab:blue', linestyle='--', label='Uniqueness')
    ax2.set_ylabel("Uniqueness", color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Title with ARUC value
    plt.title(f"(ARUC={aruc:.3f})", fontsize=9)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left", fontsize=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def example1():
    global metric, result, R, U
    target = [
        0, 2, 6, 2, 1, 4, 4, 0, 4, 4, 5, 3, 1, 4, 0, 0, 4, 4, 5, 5, 5, 6, 4, 0, 0, 2, 4, 4, 4, 2, 4, 6, 6, 0, 0, 2, 2,
        4, 5, 5, 6, 1, 5, 5, 1, 0, 5, 4, 1, 3, 6, 1, 0, 5, 3, 3, 0, 0, 2, 4, 2, 3, 3, 3, 5, 0, 2, 4, 5, 3, 4, 3, 2, 4,
        5, 3, 0, 2, 0, 5, 4, 1, 2, 5, 2, 2, 2, 2, 2, 4, 2, 4, 1, 3, 3, 4, 4, 4, 3, 4, 2, 3, 3, 2, 5, 2, 4, 1, 6, 0, 0,
        6, 5, 6, 2, 3, 4, 4, 0, 3, 3, 5, 2, 1, 5, 0, 2, 3, 3, 0, 5, 0, 5, 2, 6, 2, 3, 3, 4, 5, 2, 2, 4, 2, 3, 5, 4, 1,
        6, 4, 2, 1, 0, 0, 3, 3, 2, 4, 0, 0, 6, 4, 2, 4, 2, 1, 1, 3, 3, 3, 6, 5, 5, 3, 2, 2, 2, 2, 4, 0, 5, 4, 0, 4, 2,
        6, 5, 0, 4, 3, 6, 1, 3, 3, 0, 5, 3, 0, 3, 4, 4, 0, 0, 2, 2, 0, 4, 0, 1, 2, 2, 4, 6, 5, 5, 0, 3, 4, 5, 0, 2, 1,
        0, 1, 3, 4, 2, 0, 0, 0, 5, 0, 3, 3, 5, 2, 2, 1, 1, 2, 2, 1, 2, 4, 4, 4, 5, 3, 3, 0, 5, 2, 2, 2, 6, 2, 6, 2, 0,
        3, 3, 2, 2, 5, 2, 4, 3, 4, 3, 3, 3, 4, 6, 2, 1, 5, 3, 3, 2, 3, 4, 4, 2, 5, 0, 1, 4, 2, 4, 6, 4, 0, 0, 2, 4, 5,
        5, 5, 4, 6, 4, 2, 0, 4, 3, 3, 3, 4, 4, 1, 4, 3, 0, 2, 3, 2, 6, 2, 4, 2, 4, 3, 3, 2, 6, 2, 4, 2, 0, 4, 4, 0, 2,
        4, 0, 4, 4, 5, 6, 0, 3, 5, 5, 2, 5, 2, 1, 5, 0, 2, 1, 1, 5, 2, 4, 3, 1, 1, 4, 0, 2, 3, 3, 2, 3, 2, 3, 3, 1, 5,
        0, 2, 4, 4, 3, 0, 1, 0, 4, 1, 4, 1, 0, 2, 1, 4, 6, 5, 3, 2, 5, 3, 0, 0, 5, 5
    ]
    pos_samples = [[0, 2, 3, 2, 3, 4, 4, 0, 4, 3, 4, 1, 3, 3, 3, 5, 4, 4, 5, 5, 0, 0,
                    4, 0, 0, 2, 6, 3, 4, 2, 4, 0, 5, 0, 3, 2, 2, 4, 5, 2, 3, 0, 3, 5,
                    5, 3, 3, 4, 1, 3, 0, 5, 4, 4, 3, 4, 3, 3, 2, 3, 2, 3, 3, 3, 5, 3,
                    3, 4, 4, 1, 4, 3, 2, 4, 3, 3, 0, 2, 3, 5, 4, 2, 2, 5, 2, 2, 2, 3,
                    2, 4, 2, 3, 2, 3, 3, 4, 3, 4, 3, 4, 2, 3, 3, 2, 3, 3, 4, 3, 6, 0,
                    0, 3, 2, 0, 2, 3, 5, 4, 3, 3, 3, 5, 2, 3, 5, 0, 2, 3, 3, 3, 5, 0,
                    4, 2, 6, 3, 3, 3, 4, 3, 2, 2, 4, 2, 3, 5, 4, 0, 1, 4, 2, 3, 0, 6,
                    3, 3, 2, 4, 3, 0, 3, 4, 2, 4, 2, 3, 0, 3, 3, 3, 6, 5, 3, 3, 2, 2,
                    2, 2, 4, 0, 5, 4, 0, 4, 2, 3, 5, 3, 4, 3, 0, 3, 3, 3, 4, 5, 3, 6,
                    3, 0, 3, 0, 4, 2, 2, 4, 4, 0, 2, 2, 2, 4, 2, 5, 5, 3, 3, 4, 5, 3,
                    2, 1, 0, 2, 3, 4, 2, 3, 0, 6, 5, 3, 3, 3, 5, 2, 2, 3, 2, 2, 2, 3,
                    2, 4, 4, 4, 4, 3, 3, 1, 5, 2, 2, 2, 2, 2, 4, 2, 0, 3, 3, 2, 2, 5,
                    2, 4, 0, 4, 3, 3, 3, 4, 3, 1, 3, 4, 5, 3, 2, 3, 4, 4, 2, 3, 3, 3,
                    6, 2, 4, 1, 3, 2, 4, 2, 4, 5, 5, 5, 4, 0, 4, 2, 0, 2, 3, 3, 5, 3,
                    4, 1, 4, 3, 0, 2, 3, 2, 2, 2, 4, 2, 3, 3, 3, 2, 3, 2, 4, 3, 0, 4,
                    3, 0, 2, 4, 4, 4, 4, 5, 3, 0, 3, 5, 2, 1, 5, 2, 3, 0, 0, 2, 3, 3,
                    3, 2, 4, 3, 2, 5, 4, 3, 2, 3, 3, 2, 3, 2, 3, 4, 2, 5, 0, 2, 4, 4,
                    3, 3, 4, 3, 4, 2, 4, 2, 2, 2, 3, 3, 5, 4, 3, 2, 3, 3, 0, 0, 0, 5]]
    neg_samples = [[5, 4, 6, 2, 1, 4, 4, 2, 4, 4, 5, 3, 1, 0, 4, 2, 4, 4, 5, 1, 0, 0,
                    0, 0, 0, 2, 4, 0, 4, 2, 4, 6, 6, 0, 0, 2, 2, 1, 5, 1, 4, 1, 5, 5,
                    1, 0, 1, 4, 1, 1, 0, 1, 2, 5, 3, 4, 3, 3, 2, 4, 2, 3, 3, 3, 5, 6,
                    5, 5, 1, 3, 4, 5, 2, 4, 1, 1, 6, 2, 0, 5, 4, 1, 2, 5, 5, 2, 2, 3,
                    2, 0, 2, 0, 0, 4, 6, 4, 3, 4, 3, 4, 2, 3, 3, 2, 6, 2, 4, 1, 6, 6,
                    0, 4, 5, 0, 2, 3, 5, 4, 0, 0, 3, 5, 2, 0, 5, 0, 2, 3, 5, 2, 0, 0,
                    6, 1, 6, 0, 3, 3, 4, 5, 2, 2, 4, 2, 3, 2, 5, 6, 6, 4, 5, 1, 0, 0,
                    3, 1, 2, 4, 4, 0, 6, 0, 2, 4, 2, 1, 1, 3, 0, 3, 0, 5, 5, 4, 2, 2,
                    2, 2, 4, 6, 2, 4, 0, 4, 5, 6, 5, 0, 4, 3, 6, 1, 3, 3, 5, 5, 3, 0,
                    3, 0, 4, 0, 0, 2, 2, 5, 4, 0, 1, 2, 2, 4, 6, 5, 5, 0, 3, 4, 5, 2,
                    2, 1, 6, 1, 3, 6, 2, 1, 0, 0, 5, 5, 6, 4, 5, 2, 2, 4, 1, 2, 6, 1,
                    2, 4, 4, 4, 5, 3, 0, 0, 5, 2, 2, 2, 6, 2, 6, 2, 4, 3, 3, 2, 2, 5,
                    2, 4, 3, 4, 3, 3, 3, 4, 4, 2, 1, 5, 5, 2, 2, 3, 4, 4, 2, 5, 4, 0,
                    4, 2, 4, 6, 4, 6, 0, 2, 4, 5, 5, 5, 4, 6, 4, 2, 0, 4, 3, 3, 6, 4,
                    4, 1, 4, 4, 4, 2, 1, 2, 6, 2, 6, 2, 0, 3, 3, 2, 4, 2, 4, 3, 0, 4,
                    2, 0, 2, 4, 3, 4, 4, 5, 6, 6, 3, 5, 5, 2, 0, 2, 1, 6, 0, 5, 1, 2,
                    1, 2, 4, 3, 1, 2, 4, 0, 2, 0, 3, 2, 1, 2, 3, 4, 1, 5, 0, 2, 4, 0,
                    3, 6, 1, 0, 4, 1, 4, 2, 6, 4, 1, 3, 6, 3, 1, 5, 4, 3, 0, 0, 4, 5]]
    target = np.array(target)
    pos_samples = np.array(pos_samples)
    neg_samples = np.array(neg_samples)
    # print(len(target), len(pos_samples), len(neg_samples))
    print(target.shape, pos_samples.shape, neg_samples.shape)
    metric = ARUC(tau=0.5)
    metric.init_target_pred(target)
    for p in pos_samples:
        metric.update(p, sample_label=1)
    for n in neg_samples:
        metric.update(n, sample_label=0)
    result, R, U = metric.compute()
    print("Label-Level ARUC:", result)
    plot_aruc(R, U, result, title='ARUC', save_path='test_practice.png')


def example2():
    global metric, R, U
    metric = ARUC(tau=0.5, r=100)
    metric.init_target_pred(np.array([1, 0, 1, 2]))
    # surrogate (positive) model with high agreement
    metric.update(np.array([1, 0, 1, 2]), sample_label=1)
    metric.update(np.array([1, 0, 0, 2]), sample_label=1)
    # independent (negative) model with low agreement
    metric.update(np.array([0, 1, 2, 1]), sample_label=0)
    metric.update(np.array([0, 1, 1, 1]), sample_label=0)
    aruc, R, U = metric.compute()
    print(f"ARUC = {aruc:.4f}")
    plot_aruc(R, U, aruc, title='ARUC', save_path='test_made.png')


if __name__ == '__main__':
    # test_aruc()
    ...
