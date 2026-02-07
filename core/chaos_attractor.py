# core/chaos_attractor.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable, Optional


@dataclass
class AttractorConfig:
    dt: float = 0.01          # шаг интегрирования
    T: float = 100.0          # общее "время" симуляции
    transient: float = 10.0   # отрезок разгона до записи аттрактора
    lyap_dt: float = 0.01     # шаг для оценки Ляпунова
    lyap_t: float = 50.0      # время для оценки Ляпунова
    poincare_plane: Optional[Tuple[int, float]] = None
    # (index, value) => x[index] = value для сечения Пуанкаре


class ChaosAttractor:
    """
    Обёртка для моделирования странного аттрактора
    над общей динамикой dΨ/dt = f(Ψ).

    Здесь f(Ψ) ты подаёшь снаружи — это может быть:
        f(Ψ) = dynamics_live(Ψ)   # Живое (∇R - η∇Φ_comp)
        f(Ψ) = ...                # Любая твоя нелинейная динамика

    Основные функции:
        - simulate_trajectory    — траектория на аттракторе
        - lyapunov_exponents     — спектр Ляпунова
        - poincare_section       — сечение Пуанкаре
    """

    def __init__(self, cfg: AttractorConfig):
        self.cfg = cfg

    # --------- Базовый интегратор (Эйлер / RK4) ---------

    def _step_euler(self, x: np.ndarray, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        return x + self.cfg.dt * f(x)

    def _step_rk4(self, x: np.ndarray, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        dt = self.cfg.dt
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # --------- Симуляция траектории ---------

    def simulate_trajectory(
        self,
        x0: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        method: str = "rk4",
    ) -> np.ndarray:
        """
        Возвращает траекторию после отбрасывания транзиента:
            X: shape (N_steps_eff, dim)

        x0 — начальное состояние (dim,)
        f  — поле векторного поля dΨ/dt
        """
        step = self._step_rk4 if method == "rk4" else self._step_euler

        n_total = int(self.cfg.T / self.cfg.dt)
        n_trans = int(self.cfg.transient / self.cfg.dt)

        x = x0.copy()
        traj = []

        for i in range(n_total):
            x = step(x, f)
            if i >= n_trans:
                traj.append(x.copy())

        return np.array(traj)

    # --------- Ляпуновские экспоненты (метод Бенеттина) ---------

    def lyapunov_exponents(
        self,
        x0: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        J: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Спектр Ляпуновских экспонент методом Бенеттина.

        x0 — начальное состояние
        f  — векторное поле
        J  — якобиан J(x) = df/dx, shape (dim, dim)
        """
        dt = self.cfg.lyap_dt
        T = self.cfg.lyap_t
        steps = int(T / dt)

        x = x0.copy()
        dim = x.size

        # Ортонормированный базис возмущений
        Q = np.eye(dim)
        lyap_sum = np.zeros(dim)

        for _ in range(steps):
            # Эволюция базиса по линеаризованной системе
            Jx = J(x)
            Q = Q + dt * (Jx @ Q)

            # QR-разложение для нормализации
            Q, R = np.linalg.qr(Q)
            diag_R = np.diag(R)
            lyap_sum += np.log(np.abs(diag_R) + 1e-15)

            # Эволюция основного состояния
            x = x + dt * f(x)

        lambdas = lyap_sum / (T)
        return lambdas

    # --------- Сечение Пуанкаре ---------

    def poincare_section(
        self,
        x0: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        plane: Optional[Tuple[int, float]] = None,
        method: str = "rk4",
    ) -> np.ndarray:
        """
        Строит точки пересечения траектории с плоскостью Пуанкаре:

            x[index] = value

        Возвращает массив точек пересечения.
        """
        if plane is None and self.cfg.poincare_plane is None:
            raise ValueError("Не задана плоскость Пуанкаре (index, value).")

        index, value = plane if plane is not None else self.cfg.poincare_plane
        step = self._step_rk4 if method == "rk4" else self._step_euler

        n_total = int(self.cfg.T / self.cfg.dt)
        n_trans = int(self.cfg.transient / self.cfg.dt)

        x = x0.copy()
        section_points = []

        prev_x = x.copy()
        prev_sign = np.sign(prev_x[index] - value)

        for i in range(n_total):
            x = step(x, f)
            if i < n_trans:
                prev_x = x.copy()
                prev_sign = np.sign(prev_x[index] - value)
                continue

            curr_sign = np.sign(x[index] - value)
            # Пересечение плоскости: смена знака (простое условие)
            if curr_sign == 0 or curr_sign != prev_sign:
                # Линейная интерполяция между prev_x и x
                t = (value - prev_x[index]) / (x[index] - prev_x[index] + 1e-15)
                point = prev_x + t * (x - prev_x)
                section_points.append(point.copy())

            prev_x = x.copy()
            prev_sign = curr_sign

        return np.array(section_points)
