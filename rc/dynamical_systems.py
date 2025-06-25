from typing import Callable, Optional

import numpy as np


def rk4_solve(f: Callable, x0: np.ndarray, t_fin: int, dt: float):
    t = np.arange(0, t_fin, dt)
    x = np.zeros((len(t) + 1, len(x0))).astype(float)
    x[0] = x0.copy()
    for i in range(len(x) - 1):
        k1 = f(x[i], t[i])
        k2 = f(x[i] + k1 * dt / 2, t[i] + dt / 2)
        k3 = f(x[i] + k2 * dt / 2, t[i] + dt / 2)
        k4 = f(x[i] + k3 * dt, t[i] + dt)
        dx_dt = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        dx = dx_dt * dt
        x[i + 1] = x[i] + dx
    return x


def solve_trajectory(f: Callable, x0: np.ndarray, t_fin: float, dt: float) -> np.ndarray:
    return rk4_solve(f, x0, t_fin, dt)


class LorenzSystem:
    dim = 3

    def __init__(self, sigma: float, rho: float, beta: float):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def derivative(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        dx_dt = np.zeros(self.dim).astype(float)
        dx_dt[0] = self.sigma * (x[1] - x[0])
        dx_dt[1] = x[0] * (self.rho - x[2]) - x[1]
        dx_dt[2] = x[0] * x[1] - self.beta * x[2]
        return dx_dt

    def solve_trajectory(self, *args, **kwargs):
        return solve_trajectory(f=self.derivative, *args, **kwargs)


class RosslerSystem:
    dim = 3

    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c

    def derivative(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        dx_dt = np.zeros(self.dim).astype(float)
        dx_dt[0] = -x[1] - x[2]
        dx_dt[1] = x[0] + self.a * x[1]
        dx_dt[2] = self.b + x[2] * (x[0] - self.c)
        return dx_dt

    def solve_trajectory(self, *args, **kwargs):
        return solve_trajectory(f=self.derivative, *args, **kwargs)
