# simulators/bifurcation.py
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import matplotlib.pyplot as plt


@dataclass
class BifurcationConfig:
    param_min: float          # минимум параметра (например, alpha_min или N_min)
    param_max: float          # максимум параметра
    n_param: int = 400        # число шагов по параметру
    n_transient: int = 500    # сколько итераций выбрасывать (разгон к аттрактору)
    n_sample: int = 200       # сколько точек на аттракторе рисовать для каждого параметра
    x0: float = 0.1           # начальное состояние для 1D карты
    figsize: Tuple[int, int] = (10, 6)


def bifurcation_1d_map(
    cfg: BifurcationConfig,
    f: Callable[[float, float], float],
    param_name: str = "r",
    xlabel: Optional[str] = None,
    ylabel: str = "x",
    savepath: Optional[str] = None,
):
    """
    Бифуркационная диаграмма для 1D отображения:

        x_{n+1} = f(x_n, param)

    Примеры:
        - Логистическое отображение: f(x, r) = r * x * (1 - x)
        - Твой упрощённый «социальный рейтинг» в одном измерении.
    """
    params = np.linspace(cfg.param_min, cfg.param_max, cfg.n_param)
    xs = []
    ps = []

    x = cfg.x0

    for p in params:
        # транзиент
        x = cfg.x0
        for _ in range(cfg.n_transient):
            x = f(x, p)

        # выборка на аттракторе
        for _ in range(cfg.n_sample):
            x = f(x, p)
            ps.append(p)
            xs.append(x)

    xs = np.array(xs)
    ps = np.array(ps)

    plt.figure(figsize=cfg.figsize)
    plt.scatter(ps, xs, s=0.2, color="black")
    plt.xlabel(xlabel if xlabel is not None else param_name)
    plt.ylabel(ylabel)
    plt.title(f"Bifurcation diagram for 1D map: x_(n+1) = f(x_n, {param_name})")
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()


# ===== Расширенный вариант для многомерной живой динамики =====

@dataclass
class MultiBifurcationConfig:
    param_min: float
    param_max: float
    n_param: int = 60        # поменьше, т.к. динамика тяжёлая
    n_steps: int = 1500
    n_transient: int = 800
    dt: float = 0.01
    dim: int = 1             # какую координату рисуем (индекс)
    figsize: Tuple[int, int] = (10, 6)


def bifurcation_multidim_flow(
    cfg: MultiBifurcationConfig,
    build_field: Callable[[float], Callable[[np.ndarray], np.ndarray]],
    x0: np.ndarray,
    param_name: str = "alpha",
    xlabel: Optional[str] = None,
    ylabel: str = "x_dim",
    savepath: Optional[str] = None,
):
    """
    Общая бифуркационная диаграмма для потока dΨ/dt = f(Ψ; param)
    с фиксацией одной координаты Ψ[cfg.dim].

    build_field(param) -> f(x): возвращает поле для данного параметра
    x0: начальное состояние (dim_state,)
    """
    params = np.linspace(cfg.param_min, cfg.param_max, cfg.n_param)
    xs = []
    ps = []

    for p in params:
        f = build_field(p)
        x = x0.copy()

        # транзиент (разгон к аттрактору)
        for _ in range(cfg.n_transient):
            x = x + cfg.dt * f(x)

        # выборка (берём локальные минимумы/максимумы данной координаты)
        prev = x[cfg.dim]
        trend = 0.0
        for _ in range(cfg.n_steps - cfg.n_transient):
            x = x + cfg.dt * f(x)
            curr = x[cfg.dim]
            new_trend = np.sign(curr - prev)

            # фиксируем смену тренда (примерно экстремум)
            if trend != 0 and new_trend != 0 and new_trend != trend:
                xs.append(curr)
                ps.append(p)

            prev = curr
            trend = new_trend

    xs = np.array(xs)
    ps = np.array(ps)

    plt.figure(figsize=cfg.figsize)
    plt.scatter(ps, xs, s=0.4, color="black")
    plt.xlabel(xlabel if xlabel is not None else param_name)
    plt.ylabel(ylabel)
    plt.title(f"Bifurcation diagram for flow dPsi/dt = f(Psi; {param_name})")
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()


# ===== Примеры использования =====

def example_logistic():
    """
    Пример: классическая логистическая карта
        x_{n+1} = r x_n (1 - x_n)
    """
    def logistic_map(x, r):
        return r * x * (1.0 - x)

    cfg = BifurcationConfig(
        param_min=2.5,
        param_max=4.0,
        n_param=600,
        n_transient=300,
        n_sample=200,
        x0=0.1,
    )
    bifurcation_1d_map(cfg, logistic_map, param_name="r", ylabel="x_n")


if __name__ == "__main__":
    # Быстрый запуск примера:
    example_logistic()
