def plot_efficiency():
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np

    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 20

    unlearn_ratio_array = np.linspace(0.01, 0.1, 10)
    x_labels = ["Cora", "Cite", "PubM", "Phot", "Comp", "CS", "Phys"]

    cited_inference_avg = np.array([0.1136, 0.1124, 0.1213, 0.1091, 0.1122, 0.1119, 0.1317])
    grove_inference_avg = np.array([0.1853, 0.1789, 0.1954, 0.2388, 0.3355, 0.4080, 1.0642])

    cited_inference_std = np.array([0.1585, 0.1569, 0.1692, 0.1509, 0.1532, 0.1540, 0.1621])
    grove_inference_std = np.array([0.1935, 0.1754, 0.1735, 0.1834, 0.1834, 0.1476, 0.1588])

    fig = plt.figure(figsize=(4.8, 4))

    plt.plot(x_labels, cited_inference_avg, label='CITED', marker='s', linewidth=3, markersize=11,
             color=(123 / 255.0, 141 / 255.0, 191 / 255.0))
    plt.plot(x_labels, grove_inference_avg, label='GrOVe', marker='v', linewidth=3,
             markersize=11, color=(248 / 255.0, 120 / 255.0, 80 / 255.0))

    plt.gca().set_facecolor('#EEF0F2')
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.xlabel('Datasets', fontsize=23)
    plt.ylabel('Inference Time (s)', fontsize=23)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=21)
    ax.tick_params(axis='x', which='major', labelsize=18)
    # plt.yscale('log')

    plt.legend(fontsize=16)

    save_path = './imgs/plot_efficiency.png'
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    plot_efficiency()
    print("Plotting completed.")
