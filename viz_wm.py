def plot_wm_line():
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import FormatStrFormatter
    import numpy as np

    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 20

    unlearn_ratio_array = np.linspace(0.01, 0.1, 10)
    x_range = ["5", "10", "15", "20", "25", "30", "35", "40", "45", "50"]

    # Pubmed
    cited = np.array([0.797, 0.79, 0.798, 0.8, 0.8, 0.806, 0.797, 0.806, 0.801, 0.816])
    random_wm = np.array([0.802, 0.789, 0.782, 0.78, 0.776, 0.77, 0.765, 0.761, 0.75, 0.745])
    backdoor_wm = np.array([0.8, 0.788, 0.785, 0.781, 0.778, 0.77, 0.765, 0.756, 0.744, 0.74])
    survive_wm = np.array([0.815, 0.784, 0.78, 0.774, 0.7725, 0.771, 0.77, 0.765, 0.76, 0.752])

    fig = plt.figure(figsize=(8, 3.5))

    plt.plot(x_range, cited, label='CITED', marker='s', linewidth=3, markersize=11,
             color=(123 / 255.0, 141 / 255.0, 191 / 255.0))
    plt.plot(x_range, random_wm, label='RandomWM', marker='v', linewidth=3,
             markersize=11, color=(248 / 255.0, 120 / 255.0, 80 / 255.0))
    plt.plot(x_range, backdoor_wm, label='BackdoorWM', marker='o', linewidth=3, markersize=11,
             color=(87 / 255.0, 184 / 255.0, 147 / 255.0))
    plt.plot(x_range, survive_wm, label='SurviveWM', marker='d', linewidth=3, markersize=11,
             color=(223 / 255.0, 113 / 255.0, 182 / 255.0))

    plt.gca().set_facecolor('#EEF0F2')
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.xlabel('Number of Ownership Indicators', fontsize=23)
    plt.ylabel('Accuracy', fontsize=23)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=21)
    ax.tick_params(axis='x', which='major', labelsize=18)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # plt.yscale('log')

    # plt.legend(fontsize=16)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
               ncol=2, fontsize=18, frameon=True, columnspacing=5)

    save_path = './imgs/plot_wm_line_pubmed.png'
    plt.savefig(save_path, dpi=600, format='png', bbox_inches='tight')
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    plot_wm_line()
    print("Plotting completed.")