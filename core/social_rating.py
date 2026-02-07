# core/social_rating.py
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple


@dataclass
class SocialRatingConfig:
    K: int                   # число уровней иерархии
    N_levels: list           # N_l: число агентов/подсистем на уровне l
    alpha: float = 0.8       # коэффициент затухания по уровням
    lambda0: float = 1.0     # базовый вес Λ_0
    beta: float = 1.0        # чувствительность к разнице рейтингов в конкуренции
    eta_comp: float = 0.1    # интенсивность градиента конкуренции


class SocialRating:
    """
    Социальный рейтинг R^{(a)}(Ψ^{(a)}) и пена конкуренции Φ_dom
    для многоуровневой GRA Мета-обнулёнки.

    Опорные формулы:

        R^{(a)}(Ψ^{(a)}) =
            Σ_{l=0}^K Λ_l ( ||P_{G_l} Ψ^{(a)}||^2 - Φ^{(l)}(Ψ^{(a)}, G_l) )

        Φ_comp^{(l)}(a,b) =
            |<Ψ^{(a)} | P_{G_l} | Ψ^{(b)}> |^2 * exp( -β |R^{(a)} - R^{(b)}| )
    """

    def __init__(
        self,
        config: SocialRatingConfig,
        projector_fn: Callable[[int, np.ndarray], np.ndarray],
        foam_level_fn: Callable[[int, np.ndarray], np.ndarray],
    ):
        """
        projector_fn(l, Psi_level) -> P_G_l @ Psi_level
        foam_level_fn(l, Psi_level) -> Φ^{(l)}(Ψ^{(a)}, G_l) для всех агентов уровня l
        """
        self.cfg = config
        self.projector_fn = projector_fn
        self.foam_level_fn = foam_level_fn

        self.Lambda = np.array(
            [self.cfg.lambda0 * (self.cfg.alpha ** l) for l in range(self.cfg.K + 1)],
            dtype=float,
        )

        # Индексы срезов по уровням в общем векторе/матрице состояния
        self.level_offsets = self._build_level_offsets()

    def _build_level_offsets(self) -> Dict[int, Tuple[int, int]]:
        """
        Создаёт карту {l: (start, end)} индексов для уровня l
        в глобальном массиве агентов.
        """
        offsets = {}
        start = 0
        for l, N_l in enumerate(self.cfg.N_levels):
            end = start + N_l
            offsets[l] = (start, end)
            start = end
        return offsets

    # ---------- Социальный рейтинг ----------

    def compute_R(self, Psi: np.ndarray) -> np.ndarray:
        """
        Вычисляет вектор R^(a) для всех агентов всех уровней.

        Psi: shape (N_total, D)
            N_total = Σ_l N_l, D — размерность состояния агента.
        """
        N_total = Psi.shape[0]
        R = np.zeros(N_total, dtype=float)

        for l in range(self.cfg.K + 1):
            start, end = self.level_offsets[l]
            Psi_l = Psi[start:end]  # shape (N_l, D)

            PGl_Psi = self.projector_fn(l, Psi_l)       # P_G_l Ψ
            Phi_l = self.foam_level_fn(l, Psi_l)        # Φ^{(l)}(Ψ^{(a)}, G_l), shape (N_l,)

            # ||P_G_l Ψ^{(a)}||^2 для каждого агента
            proj_norm_sq = np.sum(PGl_Psi ** 2, axis=1)

            R[start:end] = self.Lambda[l] * (proj_norm_sq - Phi_l)

        return R

    # ---------- Пена конкуренции (между равными) ----------

    def competition_foam_level(
        self,
        l: int,
        Psi: np.ndarray,
        R: np.ndarray,
    ) -> float:
        """
        Φ_comp^{(l)} = Σ_{a≠b} |<Ψ^{(a)}|P_{G_l}|Ψ^{(b)}> |^2 * exp( -β |R^{(a)} - R^{(b)}| )
        для уровня l (скалярная суммарная пена конкуренции).
        """
        start, end = self.level_offsets[l]
        Psi_l = Psi[start:end]  # (N_l, D)
        R_l = R[start:end]      # (N_l,)

        N_l = Psi_l.shape[0]
        PGl_Psi = self.projector_fn(l, Psi_l)  # (N_l, D)

        # Скалярные произведения между всеми парами: M[a,b] = <Ψ_a|PΨ_b>
        M = Psi_l @ PGl_Psi.T  # (N_l, N_l)
        M_sq = np.abs(M) ** 2

        # Матрица разностей рейтингов
        dR = np.abs(R_l[:, None] - R_l[None, :])
        weight = np.exp(-self.cfg.beta * dR)

        # Обнуляем диагональ a == b
        np.fill_diagonal(M_sq, 0.0)
        np.fill_diagonal(weight, 0.0)

        Phi_comp_l = np.sum(M_sq * weight)
        return float(Phi_comp_l)

    def competition_foam_all_levels(
        self,
        Psi: np.ndarray,
        R: np.ndarray,
    ) -> Dict[int, float]:
        """
        Возвращает {l: Φ_comp^{(l)}} для всех уровней.
        """
        phi_comp = {}
        for l in range(self.cfg.K + 1):
            phi_comp[l] = self.competition_foam_level(l, Psi, R)
        return phi_comp

    # ---------- Градиент для живой динамики ----------

    def gradient_live(
        self,
        Psi: np.ndarray,
        R: np.ndarray,
    ) -> np.ndarray:
        """
        Приближённый «градиент живого»:

            dΨ/dt ≈ ∇_Ψ R - η_comp ∇_Ψ Φ_comp

        Здесь ∇_Ψ Φ_comp даём в грубом линейном приближении:
        тянем агента в сторону роста рейтинга и отталкиваем
        от конкурентов с близким рейтингом.
        """
        N_total, D = Psi.shape
        grad = np.zeros_like(Psi)

        # Градиент по рейтингу: ∇_Ψ R ~ масштабируем Ψ к "большему влиянию"
        # (в реальной модели это через Hess(R), здесь — упрощение).
        grad_R = Psi * R[:, None]

        grad_comp = np.zeros_like(Psi)

        for l in range(self.cfg.K + 1):
            start, end = self.level_offsets[l]
            Psi_l = Psi[start:end]
            R_l = R[start:end]

            N_l = Psi_l.shape[0]
            if N_l <= 1:
                continue

            # Матрица разностей рейтингов
            dR = R_l[:, None] - R_l[None, :]
            weight = np.exp(-self.cfg.beta * np.abs(dR))

            # Сигналы конкуренции: тянемся прочь от агентов с близким рейтингом
            # sign(dR) — направление "кто выше"
            sign = np.sign(dR)

            # Собираем вклад: для агента a суммируем векторные разности с b
            for a in range(N_l):
                diff = Psi_l[a][None, :] - Psi_l  # (N_l, D)
                # Весим по weight[a,b] * (1 - |sign|) для "почти равных" рейтингов
                w = weight[a] * (1.0 - np.abs(sign[a]))
                grad_comp[start + a] += (diff * w[:, None]).sum(axis=0)

        grad = grad_R - self.cfg.eta_comp * grad_comp
        return grad
