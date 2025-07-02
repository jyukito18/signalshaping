from constellation import *
from symbol_probability import *
from mi_calculator import compute_mi_unshaped, compute_mi_shaped
from plot_utils import shannon_limit_calculation, plot_mi_curves
import numpy as np

# メイン関数
def main():
    snr_list = np.arange(3, 12, 1)  # SNR値（dB）

    # 非シェーピングの計算
    const_unshaped = generate_16QAM_constellation()
    mi_unshaped = [compute_mi_unshaped(snr, const_unshaped) for snr in snr_list]

    # シェーピングβ=0.1
    const1 = generate_16QAM_constellation_scaled1()
    prob1 = generate_symbol_prob_scaled1(const1)
    mi_shaped1 = [compute_mi_shaped(10, snr, const1, prob1) for snr in snr_list]

    # シェーピングβ=0.2
    const2 = generate_16QAM_constellation_scaled2()
    prob2 = generate_symbol_prob_scaled2(const2)
    mi_shaped2 = [compute_mi_shaped(4.69, snr, const2, prob2) for snr in snr_list]

    # シェーピングβ=0.3
    const3 = generate_16QAM_constellation_scaled3()
    prob3 = generate_symbol_prob_scaled3(const3)
    mi_shaped3 = [compute_mi_shaped(3.33, snr, const3, prob3) for snr in snr_list]

    # 結果のまとめと可視化
    mi_results = [
        (mi_unshaped, 'Unshaped', 'o-.'),
        (mi_shaped1, 'Shaped(β=0.1)', '^-'),
        (mi_shaped2, 'Shaped(β=0.2)', '>-'),
        (mi_shaped3, 'Shaped(β=0.3)', 'v-')
    ]
    plot_mi_curves(snr_list, mi_results, [r[1] for r in mi_results])

if __name__ == '__main__':
    main()
