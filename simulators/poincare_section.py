# simulators/poincare_section.py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

from core.chaos_attractor import ChaosAttractor, AttractorConfig


@dataclass
class PoincareConfig:
    dt: float = 0.01
    T: float = 200.0
    transient: float = 50.0
    plane: Tuple[int, float] = (0, 0.0)  # (index, value)
    method: str = "rk4"
    figsize: Tuple[int, int] = (8, 8)
    x_index: int = 1  # что рисуем по оси X
    y_index: int = 2  # что рисуем по оси Y


def compute_poincare_points(
    x0: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    cfg: PoincareConfig,
) -> np.ndarray:
    """
    Строит точки сечения Пуанкаре для потока dΨ/dt = f(Ψ):

        плоскость: x[plane_index] = plane_value

    Возвращает массив shape (N_points, dim_state).
    """
    plane_index, plane_value = cfg.plane

    ca = ChaosAttractor(
        AttractorConfig(
            dt=cfg.dt,
            T=cfg.T,
            transient=cfg.transient,
            poincare_plane=cfg.plane,
        )
    )

    points = ca.poincare_section(
        x0=x0,
        f=f,
        plane=cfg.plane,
        method=cfg.method,
    )
    return points


def plot_poincare_section(
    points: np.ndarray,
    cfg: PoincareConfig,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
):
    """
    Рисует сечение Пуанкаре в плоскости (x_index, y_index).
    """
    if points.size == 0:
        print("Нет точек сечения Пуанкаре.")
        return

    x = points[:, cfg.x_index]
    y = points[:, cfg.y_index]

    plt.figure(figsize=cfg.figsize)
    plt.scatter(x, y, s=2, c="black")
    plt.xlabel(f"Psi[{cfg.x_index}]")
    plt.ylabel(f"Psi[{cfg.y_index}]")
    plt.title(title if title is not None else "Poincaré section")
    plt.axis("equal")
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()


# ===== Пример использования с любым твоим полем f =====

def example_with_field(f: Callable[[np.ndarray], np.ndarray], dim_state: int = 3):
    """
    Пример: передаешь сюда своё поле f(Ψ) из мультиверса (живое/неживое).
    """
    x0 = np.random.randn(dim_state) * 0.1

    cfg = PoincareConfig(
        dt=0.01,
        T=300.0,
        transient=100.0,
        plane=(0, 0.0),   # сечение по Psi[0] = 0
        method="rk4",
        figsize=(8, 8),
        x_index=1,        # рисуем Psi[1]
        y_index=2,        # против Psi[2]
    )

    points = compute_poincare_points(x0, f, cfg)
    plot_poincare_section(points, cfg, title="Poincaré section for GRA multiverse")


if __name__ == "__main__":
    # Простейший тест: Лоренц-подобное поле
    def lorenz_like(x: np.ndarray, sigma=10.0, rho=28.0, beta=8.0/3.0) -> np.ndarray:
        dx = np.empty_like(x)
        dx[0] = sigma * (x[1] - x[0])
        dx[1] = x[0] * (rho - x[2]) - x[1]
        dx[2] = x[0] * x[1] - beta * x[2]
        return dx

    example_with_field(lambda x: lorenz_like(x), dim_state=3)
