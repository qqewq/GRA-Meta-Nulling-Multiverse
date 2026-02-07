import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass

@dataclass
class MultiverseState:
    Psi: np.ndarray  # Состояние мультиверса H_multiverse
    R: np.ndarray    # Социальные рейтинги
    Phi: Dict[int, float]  # Пена по уровням

class GRAMultiverse:
    def __init__(self, K: int, N_levels: list, alpha: float = 0.8):
        self.K = K  # Уровней иерархии
        self.N_levels = N_levels  # Конкурентов на уровне
        self.alpha = alpha
        self.Lambda = [1.0 * alpha**l for l in range(K+1)]
        self.dim_M = np.sum(np.array(N_levels) * 2)  # dim(M)
    
    def social_rating(self, Psi: np.ndarray) -> np.ndarray:
        """R^(a)(Psi^(a)) = sum Lambda_l * (||P_Gl Psi||^2 - Phi^(l))"""
        R = np.zeros(self.dim_M)
        for l in range(self.K + 1):
            PGl = self._projector_level(l, Psi)  # Проектор доминирования
            phi_l = self._competition_foam(l, Psi)
            R[l*self.N_levels[l]: (l+1)*self.N_levels[l]] = (
                np.linalg.norm(PGl @ Psi, axis=1)**2 - phi_l
            ) * self.Lambda[l]
        return R
    
    def dynamics_live(self, Psi: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """dPsi/dt = +grad R - eta * grad Phi_comp (живое)"""
        grad_R = self._gradient_R(Psi)
        grad_Phi = self._gradient_competition(Psi)
        return Psi + dt * (grad_R - 0.1 * grad_Phi)
    
    def dynamics_dead(self, Psi: np.ndarray) -> np.ndarray:
        """dPsi/dt = 0 (неживое — статика)"""
        return Psi.copy()
