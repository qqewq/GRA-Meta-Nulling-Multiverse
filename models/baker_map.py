# models/baker_map.py
"""
Отображение пекаря для мультиверса — дискретный аналог хаоса обнуления.

Вспомним твою формулу из раздела 8.2:

B(Ψ^(a)) = { 2Ψ^(a)     если Φ^(l) > ε    # растяжение (доминирование)
          { 1/2Ψ^(a)   если Φ^(l) ≤ ε    # складывание (обнуление)

Это **идеальная реализация свойств странного аттрактора**:
- Растяжение: ||DF|| > 1 (конкуренция среди равных)  
- Складывание: det(DF) < 1 (установление иерархии)
"""

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class BakerMapConfig:
    n_iter: int = 12          # число итераций (2^12 = 4096 точек)
    epsilon: float = 0.5      # порог Φ^(l) для обнуления
    grid_size: int = 512      # размер сетки для визуализации
    figsize: tuple = (12, 10)


class BakerMap:
    """
    Отображение пекаря как дискретная модель:
    1) Растяжение по вертикали (конкуренция)
    2) Сгибание пополам (обнуление пены)  
    3) Перестановка половин (установка доминирования)
    """
    
    def __init__(self, cfg: BakerMapConfig):
        self.cfg = cfg
        
    def apply_map(self, x: float, y: float) -> tuple[float, float]:
        """
        B(x,y) = дискретное отображение пекаря
        
        Если y > ε (высокая пена Φ^(l)): 
            растягиваем вертикально → складываем → меняем порядок
        """
        if y > self.cfg.epsilon:
            # Верхняя половина: растяжение 2y → складывание → слева
            new_x = 2 * x
            new_y = 0.5 * y
        else:
            # Нижняя половина: растяжение 2y → складывание → справа  
            new_x = 2 * x - 1
            new_y = 0.5 * y
            
        return new_x % 1, new_y  # берём дробную часть
    
    def iterate_trajectory(self, x0: float = 0.1, y0: float = 0.2, 
                          n_iter: Optional[int] = None) -> np.ndarray:
        """
        Итерации отображения: [(x0,y0), (x1,y1), (x2,y2), ...]
        """
        if n_iter is None:
            n_iter = self.cfg.n_iter
            
        trajectory = np.empty((n_iter, 2))
        trajectory[0] = [x0, y0]
        
        for i in range(1, n_iter):
            x, y = trajectory[i-1]
            trajectory[i] = self.apply_map(x, y)
            
        return trajectory
    
    def density_map(self, n_points: int = 10000) -> np.ndarray:
        """
        Строит плотность аттрактора (историю всех точек).
        """
        density, xedges, yedges = np.histogram2d(
            [], [], bins=self.cfg.grid_size, range=[[0,1], [0,1]]
        )
        
        for _ in range(n_points):
            x, y = np.random.uniform(0, 1, 2)
            traj = self.iterate_trajectory(x, y, n_iter=20)
            density += np.histogram2d(
                traj[:,0], traj[:,1], bins=density.shape, 
                range=[[0,1], [0,1]]
            )[0]
            
        return density / n_points
    
    def lyapunov_exponent(self, x0: float, y0: float, n_iter: int = 1000) -> float:
        """
        Ляпуновская экспонента для отображения пекаря:
        
        λ = lim (1/N) Σ log|DF(x_n)|
        где DF = якобиан отображения (всегда |det DF| = 1/2 * 2 = 1)
        """
        traj = self.iterate_trajectory(x0, y0, n_iter)
        
        # Для пекаря: растяжение по x (|DF_xx| = 2), сжатие по y (|DF_yy| = 1/2)
        lyap_sum = 0.0
        for i in range(n_iter-1):
            # Локально: log|2| для растяжения, log|1/2| для сжатия
            lyap_sum += np.log(2.0)  # доминирующее направление
            
        return lyap_sum / (n_iter - 1)


# ===== Визуализация: Растяжение → Складывание → Перестановка =====

def visualize_baker_step(ax: plt.Axes, step: int):
    """Показывает геометрию одного шага пекаря."""
    cfg = BakerMapConfig()
    baker = BakerMap(cfg)
    
    # Исходный квадрат [0,1]×[0,1]
    X, Y = np.meshgrid(np.linspace(0,1,20), np.linspace(0,1,20))
    
    # Применяем отображение
    X_new, Y_new = np.zeros_like(X), np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_new[i,j], Y_new[i,j] = baker.apply_map(X[i,j], Y[i,j])
    
    ax.scatter(X, Y, c='blue', s=10, alpha=0.6, label='До')
    ax.scatter(X_new, Y_new, c='red', s=10, alpha=0.6, label='После')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Baker map: шаг {step}')


def plot_complete_analysis():
    """Полная визуализация пекаря + аналогия с обнулением."""
    cfg = BakerMapConfig(n_iter=15, grid_size=256)
    baker = BakerMap(cfg)
    
    fig = plt.figure(figsize=(15, 12))
    
    # 1) Геометрия отображения
    ax1 = plt.subplot(2, 3, 1)
    visualize_baker_step(ax1, 1)
    
    # 2) Траектория
    ax2 = plt.subplot(2, 3, 2)
    traj = baker.iterate_trajectory(0.123, 0.456, 1000)
    ax2.plot(traj[:,0], traj[:,1], 'k-', lw=0.3, alpha=0.8)
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.set_aspect('equal')
    ax2.set_title('Траектория на аттракторе')
    ax2.grid(True, alpha=0.3)
    
    # 3) Плотность аттрактора
    ax3 = plt.subplot(2, 3, 3)
    density = baker.density_map()
    im = ax3.imshow(density.T, origin='lower', cmap='hot', extent=[0,1,0,1])
    ax3.set_title('Плотность аттрактора')
    plt.colorbar(im, ax=ax3)
    
    # 4) Ляпунов по начальным условиям
    ax4 = plt.subplot(2, 3, 4)
    lyaps = []
    x0s = np.linspace(0.1, 0.9, 20)
    for x0 in x0s:
        lyap = baker.lyapunov_exponent(x0, 0.5, 500)
        lyaps.append(lyap)
    ax4.scatter(x0s, lyaps, c='green', s=50)
    ax4.axhline(0, color='red', ls='--', label='λ=0')
    ax4.set_xlabel('x₀')
    ax4.set_ylabel('λ')
    ax4.legend()
    ax4.grid(True)
    ax4.set_title('Ляпуновская экспонента')
    
    # 5) Множество точек после N итераций
    ax5 = plt.subplot(2, 3, 5)
    points = []
    for _ in range(1000):
        x, y = np.random.uniform(0,1,2)
        traj = baker.iterate_trajectory(x, y, cfg.n_iter)
        points.append(traj[-1])
    points = np.array(points)
    ax5.scatter(points[:,0], points[:,1], s=1, c='purple', alpha=0.6)
    ax5.set_xlim(0,1)
    ax5.set_ylim(0,1)
    ax5.set_aspect('equal')
    ax5.set_title('Аттрактор после 2^12 итераций')
    
    # 6) Энтропия Колмогорова-Синая
    ax6 = plt.subplot(2, 3, 6)
    entropies = []
    for n_part in [4, 8, 16, 32, 64]:
        traj_long = baker.iterate_trajectory(0.123, 0.456, 5000)
        H = baker._kolmogorov_entropy(traj_long, n_part)
        entropies.append(H)
    ax6.plot(np.log2([4,8,16,32,64]), entropies, 'bo-')
    ax6.set_xlabel('log₂N')
    ax6.set_ylabel('H_μ')
    ax6.grid(True)
    ax6.set_title('Энтропия Колмогорова-Синая')
    
    plt.tight_layout()
    plt.show()


# Дополнение класса для энтропии (внутренний метод)
def _kolmogorov_entropy(self, trajectory: np.ndarray, n_partitions: int) -> float:
    """h_μ = lim (1/n) H(ξ_n) — энтропия Колмогорова-Синая."""
    x_bins = np.linspace(0, 1, n_partitions + 1)
    y_bins = np.linspace(0, 1, n_partitions + 1)
    
    hist, _, _ = np.histogram2d(trajectory[:,0], trajectory[:,1], 
                               bins=[x_bins, y_bins])
    probs = hist.flatten() / len(trajectory)
    probs = probs[probs > 0]
    
    return -np.sum(probs * np.log2(probs))


BakerMap._kolmogorov_entropy = _kolmogorov_entropy


if __name__ == "__main__":
    plot_complete_analysis()
    
    # Быстрый тест твоей формулы из 8.2
    cfg = BakerMapConfig()
    baker = BakerMap(cfg)
    
    print("Тест B(Ψ^(a)): растяжение/складывание")
    x, y = 0.3, 0.7  # Φ^(l) = 0.7 > ε = 0.5
    print(f"Φ^(l)={y}>ε → B=({baker.apply_map(x,y)})")  # (0.6, 0.35)
    
    x, y = 0.3, 0.3  # Φ^(l) = 0.3 < ε
    print(f"Φ^(l)={y}<ε → B=({baker.apply_map(x,y)})")  # (-0.4, 0.15) → (0.6, 0.15)
