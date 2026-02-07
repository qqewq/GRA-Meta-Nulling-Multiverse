# utils/lyapunov.py
"""
Вычисление Ляпуновских экспонент для GRA Мета-обнулёнки.

Реализация двух методов:
1. Метод QR-разложения Бенеттина (для спектра λ_i)  
2. Метод парного расхождения (для λ_max)

Теорема 3: Для живых систем λ_i > 0 (трансверсально к A_live),
           λ_i < 0 (касательно к A_live)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import matplotlib.pyplot as plt


@dataclass
class LyapunovConfig:
    lyap_time: float = 100.0     # время вычисления
    lyap_dt: float = 0.01        # шаг времени
    n_renorm: int = 100          # частота QR-ренормировки
    eps_pair: float = 1e-9       # порог близости пар для метода пар
    max_pairs: int = 1000        # макс. пар для метода пар


class LyapunovAnalyzer:
    """
    Универсальный анализатор Ляпуновских экспонент.
    
    Поддерживает любые динамические системы f(x):
        - Потоки: dx/dt = f(x)
        - Отображения: x_{n+1} = f(x_n)
    """
    
    def __init__(self, cfg: LyapunovConfig):
        self.cfg = cfg
        
    def lyapunov_spectrum_qr(
        self,
        x0: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        J: Callable[[np.ndarray], np.ndarray],
        map_step: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Спектр Ляпуновских экспонент методом QR Бенеттина.
        
        Теорема 3: sum λ_i = -tr(Hess J_multiverse) < 0 (диссипативность)
        
        Args:
            x0: начальное состояние (dim,)
            f: поле скорости dx/dt = f(x) или отображение
            J: якобиан df/dx, shape (dim, dim)
            map_step: True если дискретное отображение
            
        Returns:
            lambdas: спектр λ_1 ≥ λ_2 ≥ ... ≥ λ_dim
            history: история роста log||δΨ_i||
        """
        dim = len(x0)
        steps = int(self.cfg.lyap_time / self.cfg.lyap_dt)
        
        # История экспонент
        lyap_history = np.zeros((steps // self.cfg.n_renorm, dim))
        
        # Ортонормированный базис возмущений Q
        Q = np.eye(dim)
        x = x0.copy()
        lyap_sum = np.zeros(dim)
        
        step_idx = 0
        for i in range(steps):
            # 1. Эволюция якобиана вдоль траектории
            Jx = J(x)
            Q = Q + self.cfg.lyap_dt * (Jx @ Q) if not map_step else Jx @ Q
            
            # 2. QR-ренормировка каждые n_renorm шагов
            if i % self.cfg.n_renorm == 0:
                Q, R = np.linalg.qr(Q)
                # log|diag(R)| — прирост экспонент
                growth = np.log(np.abs(np.diag(R)) + 1e-15)
                lyap_sum += growth
                lyap_history[step_idx] = growth
                step_idx += 1
            
            # 3. Эволюция основной траектории
            dx = f(x)
            x = x + self.cfg.lyap_dt * dx if not map_step else f(x)
        
        # Нормировка: λ_i = (1/T) Σ log|diag(R)|
        lambdas = lyap_sum / (steps / self.cfg.n_renorm)
        return lambdas, lyap_history
    
    def lyapunov_max_pair(
        self,
        trajectory: np.ndarray,
        eps: float = None
    ) -> float:
        """
        Максимальная Ляпуновская экспонента методом парного расхождения.
        
        λ_max = lim (1/t) <log|δΨ(t)/δΨ(0)|>
        
        Args:
            trajectory: траектория shape (N, dim)
            eps: порог близости начальных пар (по умолчанию cfg.eps_pair)
        """
        if eps is None:
            eps = self.cfg.eps_pair
            
        N, dim = trajectory.shape
        dlogs = []
        
        # Находим пары близких начальных состояний
        for i in range(N - 100):  # оставляем запас для эволюции
            for j in range(i + 1, min(i + self.cfg.max_pairs, N)):
                d0 = np.linalg.norm(trajectory[i] - trajectory[j])
                if d0 < eps:
                    # Следим за расхождением
                    k_max = min(N - max(i, j), 50)
                    for k in range(1, k_max):
                        dk = np.linalg.norm(trajectory[i + k] - trajectory[j + k])
                        if dk > 1e3 * eps:  # сатурация
                            break
                        dlogs.append(np.log(dk / d0) / k)
        
        if not dlogs:
            return 0.0
        
        return np.mean(dlogs)
    
    def chaos_diagnostics(
        self,
        lambdas: np.ndarray
    ) => dict:
        """
        Диагностика хаотичности по спектру (Теорема 3).
        """
        lambda_pos = lambdas[lambdas > 0]
        lambda_neg = lambdas[lambdas < 0]
        KS_entropy = np.sum(lambda_pos)  # h_μ(A) = Σ λ_i>0 λ_i
        
        return {
            'is_chaotic': len(lambda_pos) > 0,
            'lyapunov_dim': 1 + np.sum(lambda_pos / np.abs(lambda_neg)),
            'KS_entropy': KS_entropy,
            'volume_contraction': -np.sum(lambda_neg),
            'positive_exponents': lambda_pos,
            'negative_exponents': lambda_neg
        }


# ===== Утилиты для численного якобиана =====

def numerical_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """Численный якобиан для случаев без аналитического J."""
    dim = len(x)
    J = np.zeros((dim, dim))
    fx = f(x)
    
    for i in range(dim):
        dx = np.zeros_like(x)
        dx[i] = eps
        J[:, i] = (f(x + dx) - fx) / eps
    
    return J


# ===== Интеграция с твоими симуляторами =====

def analyze_gra_live_system(
    social_rating: 'SocialRating',  # из core/social_rating.py
    x0: np.ndarray
) -> dict:
    """
    Полный анализ живой системы: λ_i, h_μ, dim_H.
    """
    cfg = LyapunovConfig()
    analyzer = LyapunovAnalyzer(cfg)
    
    # Строим поле + якобиан
    def f(x_flat):
        D = 4  # размерность состояния агента
        N = x_flat.size // D
        Psi = x_flat.reshape(N, D)
        R = social_rating.compute_R(Psi)
        grad = social_rating.gradient_live(Psi, R)
        return grad.reshape(-1)
    
    def J(x_flat):
        return numerical_jacobian(f, x_flat)
    
    # Спектр Ляпунова
    lambdas, history = analyzer.lyapunov_spectrum_qr(
        x0, f, J, map_step=False
    )
    
    # Диагностика
    diagnostics = analyzer.chaos_diagnostics(lambdas)
    diagnostics['spectrum'] = lambdas
    
    print(f"Спектр Ляпунова: {lambdas}")
    print(f"h_μ(A_live) = {diagnostics['KS_entropy']:.3f}")
    print(f"dim_Lyap = {diagnostics['lyapunov_dim']:.3f}")
    
    return diagnostics


# ===== Визуализация =====

def plot_lyapunov_spectrum(
    lambdas: np.ndarray,
    history: Optional[np.ndarray] = None,
    title: str = "Lyapunov spectrum"
):
    """График спектра + история роста."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Спектр
    ax1.bar(range(len(lambdas)), lambdas, color=['red' if l>0 else 'blue' for l in lambdas])
    ax1.axhline(0, color='black', ls='--')
    ax1.set_xlabel('i')
    ax1.set_ylabel('λ_i')
    ax1.set_title(f'{title}\nh_μ = Σλ_i>0 λ_i = {np.sum(lambdas[lambdas>0]):.3f}')
    ax1.grid(True, alpha=0.3)
    
    # История (первые несколько направлений)
    if history is not None:
        for i in range(min(4, history.shape[1])):
            ax2.semilogy(np.cumsum(history[:, i]), label=f'log||δΨ_{i}||')
        ax2.legend()
        ax2.set_xlabel('ренормировки')
        ax2.set_title('Рост возмущений')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


# ===== Быстрый тест на Лоренце =====
if __name__ == "__main__":
    # Тест на Лоренце (должно быть λ_1 ≈ 0.9, λ_2 ≈ 0, λ_3 ≈ -14.6)
    def lorenz(x):
        s, r, b = 10, 28, 8/3
        return np.array([
            s*(x[1]-x[0]),
            x[0]*(r-x[2])-x[1],
            x[0]*x[1]-b*x[2]
        ])
    
    def J_lorenz(x):
        s, r, b = 10, 28, 8/3
        return np.array([
            [-s, s, 0],
            [r-x[2], -1, -x[0]],
            [x[1], x[0], -b]
        ])
    
    cfg = LyapunovConfig()
    analyzer = LyapunovAnalyzer(cfg)
    
    lambdas, history = analyzer.lyapunov_spectrum_qr(
        np.array([1.0, 1.0, 1.0]), lorenz, J_lorenz
    )
    
    diagnostics = analyzer.chaos_diagnostics(lambdas)
    plot_lyapunov_spectrum(lambdas, history, "Lorenz system")
    
    print("Lorenz: ", diagnostics)
