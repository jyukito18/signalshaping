import numpy as np

# 通常の16QAMコンステレーションを生成（±3, ±1）で構成され、平均電力が1に正規化される
def generate_16QAM_constellation():
    points_1d = np.array([-3, -1, 1, 3], dtype=float)
    constellation = [complex(i, q) for i in points_1d for q in points_1d]
    constellation = np.array(constellation)
    constellation /= np.sqrt(10)  # normalize to average power 1
    return constellation

# シェーピング用スケーリングバージョン1（±3.586, ±1.195）
def generate_16QAM_constellation_scaled1():
    p = np.array([-3.586, -1.195, 1.195, 3.586])
    return np.array([complex(i, q) for i in p for q in p])

# シェーピング用スケーリングバージョン2（±3, ±1）
def generate_16QAM_constellation_scaled2():
    p = np.array([-3, -1, 1, 3])
    return np.array([complex(i, q) for i in p for q in p])

# シェーピング用スケーリングバージョン3（±3, ±1）
def generate_16QAM_constellation_scaled3():
    p = np.array([-3, -1, 1, 3])
    return np.array([complex(i, q) for i in p for q in p])
