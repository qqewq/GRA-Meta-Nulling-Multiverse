# models/lorenz_analogy.py
"""
Аналогия аттрактора Лоренца с многоуровневой GRA Мета-обнулёнкой.
Таблица параллелей из твоей теории:

| Система Лоренца      | Мультиверсное обнуление     |
|----------------------|-----------------------------|
| Три переменные (x,y,z) | Мультииндекс 𝐚             |
| Параметры (σ,ρ,β)    | {Λ_l, G_l}                 |
| Странный аттрактор   | Множество A                 |
| Бабочка Лоренца      | Фрактальная структура уровней |

Реализация + симуляция для визуализации параллелей.
"""

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class LorenzConfig:
    sigma: float = 10.0
    rho: float = 28.0  
    beta: float = 8.0 / 3.0
    dt: float = 0.01
    T: float = 50.0
    transient: float = 10.0


class LorenzAnalogy:
    """
    Класс для симуляции Лоренца + параллелей с социальной динамикой.
    
    Социальная интерпретация (по аналогии из web:58):
    - x ~ интенсивность социальной передачи (T)
    - y ~ воспринимаемая инфекция/угроза (I)  
    - z ~ коллективная память риска (M)
    """
    
    def __init__(self, cfg: LorenzConfig):
        self.cfg = cfg
        
    def lorenz_field(self, state: np.ndarray) -> np.ndarray:
        """Классические уравнения Лоренца."""
        x, y, z = state
        dx = self.cfg.sigma * (y - x)
        dy = x * (self.cfg.rho - z) - y
        dz = x * y - self.cfg.beta * z
        return np.array([dx, dy, dz])
    
    def social_lorenz_field(self, state: np.ndarray, alpha: float = 0.8) -> np.ndarray:
        """
        Социальная аналогия Лоренца для GRA Мета-обнулёнки.
        
        Мультиуровневая интерпретация:
        - x ~ R^{(0)}(Ψ^{(0)}) — базовый социальный рейтинг
        - y ~ Φ^{(1)}(Ψ^{(1)}) — пена конкуренции уровня 1  
        - z ~ Σ Λ_l Φ^{(l)} — суммарная мультиверсная пена
        """
        x, y, z = state
        
        # Λ_0 * grad R^{(0)} ~ sigma * (y - x) 
        sigma_social = 10.0 * alpha  # зависит от затухания уровней
        dx = sigma_social * (y - x)
        
        # Λ_1 * grad Φ^{(1)} ~ x * (ρ - z) - y
        rho_social = 28.0  # число конкурентов N_l
        dy = x * (rho_social - z) - y * (1 + alpha)
        
        # Σ Λ_l Φ^{(l)} ~ x * y - β * z
        beta_social = 8.0 / 3.0 * (1 - alpha)  # затухание памяти уровней
        dz = x * y * alpha - beta_social * z
        
        return np.array([dx, dy, dz])
    
    def simulate_trajectory(self, x0: np.ndarray, social: bool = False):
        """Симуляция траектории (RK4)."""
        n_steps = int(self.cfg.T / self.cfg.dt)
        n_trans = int(self.cfg.transient / self.cfg.dt)
        
        x = np.array(x0)
        traj = []
        
        f = self.social_lorenz_field if social else self.lorenz_field
        
        for i in range(n_steps):
            # RK4 шаг
            k1 = f(x)
            k2 = f(x + 0.5 * self.cfg.dt * k1)
            k3 = f(x + 0.5 * self.cfg.dt * k2)
            k4 = f(x + self.cfg.dt * k3)
            x = x + self.cfg.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            
            if i >= n_trans:
                traj.append(x.copy())
        
        return np.array(traj)
    
    def plot_butterfly(self, traj_classic: np.ndarray, traj_social: np.ndarray):
        """Визуализация 'бабочки' + параллели."""
        fig = plt.figure(figsize=(15, 5))
        
        # 1) Классический Лоренц
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(traj_classic[:, 0], traj_classic[:, 1], traj_classic[:, 2], 
                lw=0.5, color='blue', alpha=0.7)
        ax1.set_title("Lorenz Attractor\n(σ,ρ,β)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        
        # 2) Социальный Лоренц (alpha=0.8)
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot(traj_social[:, 0], traj_social[:, 1], traj_social[:, 2], 
                lw=0.5, color='red', alpha=0.7)
        ax2.set_title("Social Lorenz\n(Λ_l=λ₀αˡ, N_l, G_l)")
        ax2.set_xlabel("R⁽⁰⁾")
        ax2.set_ylabel("Φ⁽¹⁾") 
        ax2.set_zlabel("ΣΛ_lΦ⁽ˡ⁾")
        
        # 3) Сравнение проекций (x-y плоскость)
        ax3 = fig.add_subplot(133)
        ax3.plot(traj_classic[:, 0], traj_classic[:, 1], 'b-', lw=0.8, alpha=0.7, label="Classic")
        ax3.plot(traj_social[:, 0], traj_social[:, 1], 'r-', lw=0.8, alpha=0.7, label="Social (α=0.8)")
        ax3.set_title("Projection comparison")
        ax3.set_xlabel("x / R⁽⁰⁾")
        ax3.set_ylabel("y / Φ⁽¹⁾")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def bifurcation_rho_social(self, alpha_values: np.ndarray):
        """
        Бифуркационная диаграмма по α (затухание уровней).
        Аналог изменения ρ в классическом Лоренце.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for ax, social in zip([ax1, ax2], [False, True]):
            xs = []
            zs = []
            
            for alpha in alpha_values:
                if social:
                    cfg = self.cfg.__dict__.copy()
                    cfg['rho'] = 28.0  # фиксируем N_l
                else:
                    cfg['rho'] = alpha * 30  # масштабируем ρ
                
                lorenz = LorenzAnalogy(LorenzConfig(**cfg))
                traj = lorenz.simulate_trajectory([0.1, 0.1, 0.1], social=social)
                
                # Выбираем локальные минимумы z
                z_diff = np.diff(traj[:, 2])
                minima_idx = np.where((z_diff[:-1] > 0) & (z_diff[1:] < 0))[0] + 1
                if len(minima_idx) > 0:
                    xs.append(alpha)
                    zs.append(traj[minima_idx[:10], 2])  # первые 10 минимумов
            
            if social:
                ax.scatter(np.repeat(alpha_values, 10), np.concatenate(zs), s=1, c='red', alpha=0.6)
                ax.set_xlabel("α (level decay)")
            else:
                ax.scatter(np.repeat(alpha_values * 30, 10), np.concatenate(zs), s=1, c='blue', alpha=0.6)
                ax.set_xlabel("ρ (classic)")
            
            ax.set_ylabel("z local minima")
            ax.grid(True, alpha=0.3)
            ax.set_title("Classic Lorenz" if not social else "Social Lorenz")
        
        plt.tight_layout()
        plt.show()


# ===== Быстрый тест =====
if __name__ == "__main__":
    cfg = LorenzConfig()
    lorenz = LorenzAnalogy(cfg)
    
    # 1) Классическая бабочка
    traj_classic = lorenz.simulate_trajectory([1.0, 1.0, 1.0])
    
    # 2) Социальная бабочка (GRA Мета-обнулёнка)
    traj_social = lorenz.simulate_trajectory([1.0, 1.0, 1.0], social=True)
    
    # 3) Визуализация параллелей
    lorenz.plot_butterfly(traj_classic, traj_social)
    
    # 4) Бифуркация по α (аналог ρ)
    alphas = np.linspace(0.1, 1.0, 50)
    lorenz.bifurcation_rho_social(alphas)
