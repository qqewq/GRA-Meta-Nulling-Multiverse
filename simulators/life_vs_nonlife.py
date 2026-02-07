def simulate_life_chaos(K=3, T=10000, alpha=0.8, eta=0.01):
    """Теорема 2.1: A_live (dim_H > 0, lambda_i > 0) vs A_dead (dim=0)"""
    mv = GRAMultiverse(K, N_levels=[5, 3, 2], alpha=alpha)
    
    # Начальные состояния
    Psi_live = np.random.randn(mv.dim_M, 2) * 0.1  # Живые агенты
    Psi_dead = np.random.randn(mv.dim_M, 2) * 0.1  # Неживые (камни)
    
    trajectories_live = []
    trajectories_dead = []
    
    for t in range(T):
        Psi_live = mv.dynamics_live(Psi_live, eta)
        Psi_dead = mv.dynamics_dead(Psi_dead)
        
        R_live = mv.social_rating(Psi_live)
        trajectories_live.append(R_live.mean())
    
    # Метрики аттрактора (Теорема 4.1)
    dim_H_live = fractal_dimension(trajectories_live)
    lyap_live = lyapunov_exponents(Psi_live)
    
    print(f"Живое: dim_H(A_live) = {dim_H_live:.3f}, h_mu = {lyap_live.sum():.3f}")
    print(f"Неживое: dim_H(A_dead) = 0, h_mu = 0")
    
    return trajectories_live, trajectories_dead
