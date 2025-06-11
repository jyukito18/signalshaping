import numpy as np

# β=0.1 に基づくシンボル確率を生成
def generate_symbol_prob_scaled1(constellation):
    z = np.exp(-0.1 * 2) * 4 + np.exp(-0.1 * 10) * 8 + np.exp(-0.1 * 18) * 4
    probs = _assign_probs(constellation, 3.586, 1.195, 0.1, z)
    return np.array(probs)

# β=0.2 に基づくシンボル確率を生成
def generate_symbol_prob_scaled2(constellation):
    z = np.exp(-0.2 * 2) * 4 + np.exp(-0.2 * 10) * 8 + np.exp(-0.2 * 18) * 4
    probs = _assign_probs(constellation, 3, 1, 0.2, z)
    return np.array(probs)

# β=0.3 に基づくシンボル確率を生成
def generate_symbol_prob_scaled3(constellation):
    z = np.exp(-0.3 * 2) * 4 + np.exp(-0.3 * 10) * 8 + np.exp(-0.3 * 18) * 4
    probs = _assign_probs(constellation, 3, 1, 0.3, z)
    return np.array(probs)

# シンボルの位置に応じて corner / edge / center の確率を割り当てる
def _assign_probs(constellation, a, b, beta, z):
    corner = np.exp(-beta * 2) / z
    edge = np.exp(-beta * 10) / z
    center = np.exp(-beta * 18) / z
    def close(x, y, tol=1e-3): return np.isclose(x, y, atol=tol)

    probs = []
    for z in constellation:
        x, y = abs(z.real), abs(z.imag)
        if close(x, a) and close(y, a):  # corner
            probs.append(corner)
        elif (close(x, a) and close(y, b)) or (close(x, b) and close(y, a)):  # edge
            probs.append(edge)
        else:  # center
            probs.append(center)
    return probs
