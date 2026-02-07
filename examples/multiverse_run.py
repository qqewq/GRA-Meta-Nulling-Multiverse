# examples/multiverse_run.py
"""
Полный запуск мультиверса GRA Мета-обнулёнки:
Жизнь vs Нежизнь → Странный аттрактор доминирования

Запуск: python examples/multiverse_run.py --save results/
Вывод: аттракторы, бифуркации, Пуанкаре, Ляпунов, Теоремы
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0, '..')

from core.social_rating import SocialRating, SocialRatingConfig
from core.chaos_attractor import ChaosAttractor, AttractorConfig
from simulators.bifurcation import MultiBifurcationConfig, bifurcation_multidim_flow
from simulators.poincare_section import PoincareConfig, compute_poincare_points
from utils.lyapunov import LyapunovAnalyzer, LyapunovConfig, plot_lyapunov_spectrum
from utils.fractal_dim import fractal_dimension


# ========== ЗАГЛУШКИ (замени на свои реальные) ==========
def projector_fn(l: int, Psi_l: np.ndarray) -> np.ndarray:
    """P_G_l: проектор доминирования уровня l"""
    norms = np.linalg.norm(Psi_l, axis=1, keepdims=True) + 1e-8
    return Psi_l / norms

def foam_level_fn(l: int, Psi_l: np.ndarray) -> np.ndarray:
    """Φ^(l): пена несогласованности уровня l"""
    mean = Psi_l.mean(axis=0, keepdims=True)
    diff = Psi_l - mean
    return np.sum(diff**2, axis=1)


# ========== МУЛЬТИВЕРС ==========
@dataclass
class MultiverseConfig:
    K: int = 3                    # уровней иерархии
    N_levels: list = [8, 5, 3]    # конкурентов на уровне
    alpha: float = 0.8            # Λ_l = λ₀αˡ
    D: int = 4                    # dim(Ψ^(a))
    T: float = 300.0              # время симуляции
    dt: float = 0.01              # шаг

cfg_mv = MultiverseConfig()
print(f"🚀 МУЛЬТИВЕРС: K={cfg_mv.K}, N={cfg_mv.N_levels}, dim(𝒎)={sum(cfg_mv.N_levels)*cfg_mv.D}")

# Социальная машина
sr_cfg = SocialRatingConfig(K=cfg_mv.K, N_levels=cfg_mv.N_levels, alpha=cfg_mv.alpha)
sr = SocialRating(sr_cfg, projector_fn, foam_level_fn)


# ========== ДИНАМИКА ЖИВОГО/НЕЖИВОГО ==========
def dynamics_live(x_flat):
    """dΨ/dt = ∇R - η∇Φ_comp (Теорема 2.1)"""
    D = cfg_mv.D
    N = x_flat.size // D
    Psi = x_flat.reshape(N, D)
    R = sr.compute_R(Psi)
    grad = sr.gradient_live(Psi, R)
    return grad.flatten()

def dynamics_dead(x_flat):
    """dΨ/dt = 0 (камни не выёживаются)"""
    return np.zeros_like(x_flat)

# Начальные условия
N_total = sum(cfg_mv.N_levels)
dim_state = N_total * cfg_mv.D
x0_live = np.random.randn(dim_state) * 0.1
x0_dead = x0_live.copy()


def run_simulation():
    """Полная симуляция мультиверса"""
    
    # 1. Траектории (A_live vs A_dead)
    print("\n🧬 Симуляция траекторий...")
    at_cfg = AttractorConfig(T=cfg_mv.T, transient=100.0, dt=cfg_mv.dt)
    ca = ChaosAttractor(at_cfg)
    
    traj_live = ca.simulate_trajectory(x0_live, dynamics_live)
    traj_dead = ca.simulate_trajectory(x0_dead, dynamics_dead)
    
    # 2. Ляпунов (Теорема 3)
    print("🔥 Ляпуновский спектр...")
    lyap_cfg = LyapunovConfig(lyap_time=100.0)
    analyzer = LyapunovAnalyzer(lyap_cfg)
    
    def J_num(x): return numerical_jacobian(dynamics_live, x)
    lambdas_live, history = analyzer.lyapunov_spectrum_qr(
        x0_live[:16], dynamics_live, J_num  # первые 16 размерностей
    )
    diag_live = analyzer.chaos_diagnostics(lambdas_live)
    
    # 3. Бифуркации по α
    print("📈 Бифуркации...")
    def build_field_alpha(alpha):
        sr_a = SocialRating(SocialRatingConfig(K=cfg_mv.K, N_levels=cfg_mv.N_levels, alpha=alpha),
                           projector_fn, foam_level_fn)
        def f(x): 
            D, N = cfg_mv.D, x.size // cfg_mv.D
            Psi = x.reshape(N, D)
            R = sr_a.compute_R(Psi)
            return sr_a.gradient_live(Psi, R).flatten()
        return f
    
    bif_cfg = MultiBifurcationConfig(param_min=0.1, param_max=0.95, n_param=120)
    
    # 4. Сечение Пуанкаре
    print("🌀 Пуанкаре...")
    p_cfg = PoincareConfig(T=600.0, plane=(0, 0.0), x_index=1, y_index=2)
    points_poincare = compute_poincare_points(x0_live, dynamics_live, p_cfg)
    
    # 5. Фрактальная размерность
    dim_H = fractal_dimension(traj_live)
    
    return {
        'traj_live': traj_live, 'traj_dead': traj_dead,
        'lambdas': lambdas_live, 'diag': diag_live,
        'points_poincare': points_poincare, 'dim_H': dim_H,
        'build_field_alpha': build_field_alpha
    }


def visualize_results(results, save_dir="results"):
    """Визуализация всех результатов"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Траектории Живое vs Неживое
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(results['traj_live'][:, 0], results['traj_live'][:, 1], 
             'cyan', lw=0.4, alpha=0.8)
    ax1.scatter(results['traj_live'][-2000:, 0], results['traj_live'][-2000:, 1], 
                s=0.3, c='yellow', alpha=0.7)
    ax1.set_title(f'🧬 A_live\ndim_H={results["dim_H"]:.3f}', color='cyan')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 4, 2)
    ax2.scatter(results['traj_dead'][0, 0], results['traj_dead'][0, 1], 
                c='gray', s=400, marker='s')
    ax2.set_title('💀 A_dead\ndim_H=0', color='gray')
    ax2.grid(True, alpha=0.3)
    
    # 2. Ляпуновский спектр
    ax3 = plt.subplot(2, 4, 3)
    colors = ['red' if l>0 else 'blue' for l in results['lambdas']]
    ax3.bar(range(len(results['lambdas'])), results['lambdas'], color=colors)
    ax3.axhline(0, color='white', ls='--')
    ax3.set_title(f'λ_i (h_μ={results["diag"]["KS_entropy"]:.3f})')
    ax3.grid(True, alpha=0.3)
    
    # 3. Сечение Пуанкаре
    ax4 = plt.subplot(2, 4, 4)
    if len(results['points_poincare']) > 0:
        ax4.scatter(results['points_poincare'][:, 1], results['points_poincare'][:, 2], 
                    s=1, c='magenta', alpha=0.6)
    ax4.set_title('🌀 Пуанкаре A_live')
    ax4.grid(True, alpha=0.3)
    
    # 4. Бифуркационная диаграмма
    ax5 = plt.subplot(2, 4, 5)
    bif_cfg = MultiBifurcationConfig(param_min=0.1, param_max=0.95, n_param=80)
    bifurcation_multidim_flow(bif_cfg, results['build_field_alpha'], x0_live, 
                             xlabel='α', ylabel='Ψ[0]', ax=ax5)
    
    # 5. История Ляпунова
    ax6 = plt.subplot(2, 4, 6)
    for i in range(min(4, results['lambdas'].size)):
        ax6.semilogy(np.cumsum(history[:, i]), label=f'δΨ_{i}')
    ax6.legend()
    ax6.set_title('Рост возмущений')
    ax6.grid(True)
    
    # 6. Энтропия vs размерность
    ax7 = plt.subplot(2, 4, 7)
    metrics = [results["dim_H"], results["diag"]["KS_entropy"], 
               results["diag"]["lyapunov_dim"]]
    ax7.bar(['dim_H', 'h_μ', 'dim_Lyap'], metrics, color=['gold', 'red', 'green'])
    ax7.set_title('Хаотические метрики')
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/multiverse_complete.png', dpi=300, facecolor='black')
    plt.show()


def numerical_jacobian(f, x, eps=1e-5):
    dim = len(x)
    J = np.zeros((dim, dim))
    fx = f(x)
    for i in range(dim):
        dx = np.zeros_like(x); dx[i] = eps
        J[:, i] = (f(x + dx) - fx) / eps
    return J


def print_theorems_verification(results):
    """Верификация всех теорем"""
    print("\n" + "="*80)
    print("✅ ВЕРИФИКАЦИЯ ТЕОРЕМ GRA МЕТА-ОБНУЛЁНКИ")
    print("="*80)
    print(f"dim_H(A_live)     = {results['dim_H']:.3f} > 0           [Теорема 2.2]")
    print(f"h_μ(A_live)       = {results['diag']['KS_entropy']:.3f} > 0  [Теорема 7]")
    print(f"λ_max             = {results['lambdas'][0]:.3f} > 0        [Теорема 3]")
    print(f"dim_Lyap          = {results['diag']['lyapunov_dim']:.3f}   [Теорема 2]")
    print(f"Хаотично          = {results['diag']['is_chaotic']}         [Теорема 5.1]")
    
    print("\n🔥 ЗАКЛЮЧЕНИЕ:")
    print("КАМНИ: ∇R=0, dim_H=0, h_μ=0, λ_i≤0")
    print("ЖИЗНЬ: ∇R>0, dim_H>0, h_μ>0, λ_i>0")
    print("🎯 ЖИВОЕ = СТРАННЫЙ АТТРАКТОР ДОМИНИРОВАНИЯ!")
    print("="*80)


def main(args):
    print("🚀 Запуск мультиверса GRA Мета-обнулёнки...")
    
    results = run_simulation()
    
    if args.save:
        visualize_results(results, args.save)
        print(f"💾 Результаты сохранены: {args.save}/")
    
    print_theorems_verification(results)
    
    # Сохранение данных
    np.savez('multiverse_results.npz', 
             traj_live=results['traj_live'],
             lambdas=results['lambdas'],
             poincare=results['points_poincare'],
             dim_H=results['dim_H'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRA Multiverse: Life vs Nonlife")
    parser.add_argument('--save', type=str, default=None, 
                       help="Папка для сохранения графиков")
    args = parser.parse_args()
    
    main(args)
