import numpy as np
import matplotlib.pyplot as plt

# Shannon容量の計算（理論上の限界）
def shannon_limit_calculation(snr_list):
    return [np.log2(1 + 10**(snr/10)) for snr in snr_list]

# MI曲線を可視化する
def plot_mi_curves(snr_list, mi_results, labels, title="16QAM Mutual Information"):
    plt.figure(figsize=(8, 5))
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    for mi, label, style in mi_results:
        plt.plot(snr_list, mi, style, label=label, linewidth=2, markersize=8)
    plt.xlabel("SNR [dB]", fontsize=16)
    plt.ylabel("Mutual Information [bits/symbol]", fontsize=16)
    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.ylim([0, 4.1])
    plt.legend(fontsize=16)
    plt.show()
