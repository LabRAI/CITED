import numpy as np
import matplotlib.pyplot as plt


def load_results_exp2(config):
    results_path = f'./results/Res_{config["defense_name"]}_{config["model_name"]}_{config["ds_name"]}_{config["level"]}.npz'

    data = np.load(results_path, allow_pickle=True)
    print(f'[Loaded] Result loaded from: {results_path}')

    aruc_arr = data['aruc']  # [n_trial]
    asr_arr = data['asr']  # [n_trial]
    R_arr = data['R']  # [n_trial, r]
    U_arr = data['U']  # [n_trial, r]
    threshold = data['threshold']  # [n_trial]

    aruc_mean = np.mean(aruc_arr)
    aruc_std = np.std(aruc_arr)
    asr_mean = np.mean(asr_arr)
    asr_std = np.std(asr_arr)

    print(f'[ARUC] {aruc_mean:.4f} ± {aruc_std:.4f}')
    print(f'[ASR ] {asr_mean:.4f} ± {asr_std:.4f}')
    print(f'[Threshold] {threshold}, mean: {np.mean(threshold):.4f}')

    best_idx = np.argmax(aruc_arr)
    best_R = R_arr[best_idx]
    best_U = U_arr[best_idx]
    best_aruc = aruc_arr[best_idx]
    return best_aruc, best_R, best_U


def plot_aruc_plus(aruc, R, U):
    r = 100
    thresholds = np.linspace(0, 1.0, r)

    my_blue = 'tab:blue'  # "#5B7493"
    my_red = 'tab:red'  # "#8D7477"

    plt.rc('font', family='serif', serif='Times New Roman')
    fig, ax1 = plt.subplots(figsize=(4.8, 4))

    # Primary y-axis: Robustness
    ax1.plot(thresholds, R, color=my_red, label='Robustness')
    ax1.set_xlabel("threshold", fontsize=23)
    ax1.tick_params(axis='x', labelsize=18)
    ax1.set_ylabel("Robustness", color=my_red, fontsize=23)
    ax1.tick_params(axis='y', labelcolor=my_red, labelsize=18)

    # Secondary y-axis: Uniqueness
    ax2 = ax1.twinx()
    ax2.plot(thresholds, U, color=my_blue, linestyle='--', label='Uniqueness')
    ax2.set_ylabel("Uniqueness", color=my_blue, fontsize=23)
    ax2.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelcolor=my_blue, labelsize=18)

    # Fill area under min(R, U)
    base = np.minimum(R, U)
    fill_color = 'gray'
    fill_alpha = 0.2
    ax2.fill_between(thresholds, base, alpha=fill_alpha, color=fill_color)

    # Place legend on top, with border, and move slightly higher
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2,
               labels_1 + labels_2,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.3),
               ncol=2,
               fontsize=18,
               frameon=True,
               fancybox=False,
               edgecolor='black')

    # Add ARUC text with matching fill color
    ax1.text(0.5, 0.15, f"ARUC={aruc:.3f}",
             fontsize=18,
             ha='center',
             va='center',
             transform=ax1.transAxes,
             bbox=dict(facecolor=fill_color, edgecolor='none', alpha=0.0))

    plt.tight_layout()

    save_path = './imgs/plot_aruc_plus_coauthor-physics.png'
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')



if __name__ == '__main__':
    ds_names = ['cora', 'citeseer', 'pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-cs', 'coauthor-physics']
    config = {
        'defense_name': 'CITED',
        'model_name': 'gcn',
        'ds_name': 'coauthor-physics',
        'level': 'label'
    }

    aruc, R, U = load_results_exp2(config)
    plot_aruc_plus(aruc, R, U)
