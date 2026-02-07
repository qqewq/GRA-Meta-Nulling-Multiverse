def fractal_dimension(trajectory: np.ndarray, epsilons: np.logspace(-3, 0, 20)) -> float:
    """dim_H(A) = lim eps->0 log N(eps) / log(1/eps)"""
    N = [len(box_cover(trajectory, eps)) for eps in epsilons]
    return np.polyfit(np.log(epsilons), np.log(N), 1)[0]
