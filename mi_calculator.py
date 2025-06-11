import numpy as np

# 非シェーピングなMI計算（均一分布）
def compute_mi_unshaped(snr_db, constellation, grid_step=0.05, axis_limit=2.5):
    M = len(constellation)
    px = 1.0 / M  # 一様分布
    snr_lin = 10**(snr_db/10)
    sigma_sq = 1.0 / snr_lin
    real_axis = np.arange(-axis_limit, axis_limit, grid_step)
    imag_axis = np.arange(-axis_limit, axis_limit, grid_step)
    area = grid_step * grid_step  # 積分領域
    c = 1 / (np.pi * sigma_sq)  # 正規化定数
    Nx, Ny = len(real_axis), len(imag_axis)
    pdf_yx = np.zeros((M, Ny, Nx), dtype=float)

    # 条件付き確率 p(y|x) を計算
    for m, x_val in enumerate(constellation):
        for iy, b in enumerate(imag_axis):
            for ix, a in enumerate(real_axis):
                diff = complex(a, b) - x_val
                pdf_yx[m, iy, ix] = c * np.exp(-(diff.real**2 + diff.imag**2) / sigma_sq)

    pdf_y = np.sum(px * pdf_yx, axis=0)  # 周辺確率 p(y)
    mi = 0.0
    eps = 1e-30
    for m in range(M):
        ratio = pdf_yx[m] / np.maximum(pdf_y, eps)
        log_term = np.log2(np.maximum(ratio, eps))
        mi += px * np.sum(pdf_yx[m] * log_term) * area  # 積分
    return mi

# シェーピングされたMI計算（任意のシンボル分布）
def compute_mi_shaped(energy, snr_db, constellation, symbol_prob, grid_step=0.05, axis_limit=5.0):
    snr_lin = 10**(snr_db/10)
    sigma_sq = energy / snr_lin  # 平均エネルギーで正規化
    real_axis = np.arange(-axis_limit, axis_limit, grid_step)
    imag_axis = np.arange(-axis_limit, axis_limit, grid_step)
    area = grid_step * grid_step
    c = 1 / (np.pi * sigma_sq)
    M = len(constellation)
    Nx, Ny = len(real_axis), len(imag_axis)
    pdf_yx = np.zeros((M, Ny, Nx), dtype=float)

    # 条件付き確率 p(y|x) を計算
    for m, xp in enumerate(constellation):
        for iy, b in enumerate(imag_axis):
            for ix, a in enumerate(real_axis):
                diff = complex(a, b) - xp
                pdf_yx[m, iy, ix] = c * np.exp(-(diff.real**2 + diff.imag**2) / sigma_sq)

    pdf_y = np.zeros((Ny, Nx), dtype=float)
    for m in range(M):
        pdf_y += symbol_prob[m] * pdf_yx[m]  # p(y) = Σp(x)p(y|x)

    mi = 0.0
    eps = 1e-30
    for m in range(M):
        ratio = pdf_yx[m] / np.maximum(pdf_y, eps)
        log_term = np.log2(np.maximum(ratio, eps))
        mi += symbol_prob[m] * np.sum(pdf_yx[m] * log_term) * area
    return mi