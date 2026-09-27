"""
American Option Pricing â Longstaff-Schwartz Monte Carlo
Day 15 â options-greeks-pricer/pricing/american_lsm.py

Implements the Longstaff-Schwartz (2001) Least-Squares Monte Carlo
algorithm for pricing American-style options, plus Binomial tree as
a reference.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AmericanOption:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    is_call: bool = False


@dataclass
class LSMResult:
    price: float
    std_error: float
    delta: float
    early_exercise_boundary: List[Tuple[float, float]]
    n_paths: int
    n_steps: int


# ---------------------------------------------------------------------------
# Utility: polynomial basis functions for regression
# ---------------------------------------------------------------------------

def _laguerre_basis(x: float, degree: int = 3) -> List[float]:
    """
    Laguerre polynomials L0..L_{degree} evaluated at x.
    Standard choice in LSM paper.
    """
    L = [0.0] * (degree + 1)
    L[0] = math.exp(-x / 2)
    if degree >= 1:
        L[1] = math.exp(-x / 2) * (1 - x)
    if degree >= 2:
        L[2] = math.exp(-x / 2) * (1 - 2 * x + x ** 2 / 2)
    if degree >= 3:
        L[3] = math.exp(-x / 2) * (1 - 3 * x + 3 * x ** 2 / 2 - x ** 3 / 6)
    return L


def _ols(X: List[List[float]], y: List[float]) -> List[float]:
    """Least-squares: beta = (X^T X)^{-1} X^T y."""
    n = len(X)
    p = len(X[0])
    XtX = [[sum(X[i][j] * X[i][k] for i in range(n))
             for k in range(p)] for j in range(p)]
    Xty = [sum(X[i][j] * y[i] for i in range(n)) for j in range(p)]
    aug = [XtX[j][:] + [Xty[j]] for j in range(p)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        if abs(pv) < 1e-14:
            continue
        aug[col] = [x / pv for x in aug[col]]
        for row in range(p):
            if row != col:
                f = aug[row][col]
                aug[row] = [aug[row][k] - f * aug[col][k] for k in range(p + 1)]
    return [aug[j][p] for j in range(p)]


# ---------------------------------------------------------------------------
# GBM path simulation
# ---------------------------------------------------------------------------

def simulate_gbm_paths(S0: float, r: float, q: float, sigma: float,
                       T: float, n_paths: int, n_steps: int,
                       seed: int = 42) -> List[List[float]]:
    """Simulate n_paths GBM price paths with antithetic variates."""
    rng = random.Random(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol_dt = sigma * math.sqrt(dt)
    paths: List[List[float]] = []
    half = n_paths // 2
    for _ in range(half):
        path = [S0]
        anti_path = [S0]
        for _ in range(n_steps):
            z = rng.gauss(0, 1)
            path.append(path[-1] * math.exp(drift + vol_dt * z))
            anti_path.append(anti_path[-1] * math.exp(drift - vol_dt * z))
        paths.append(path)
        paths.append(anti_path)
    return paths


# ---------------------------------------------------------------------------
# LSM core
# ---------------------------------------------------------------------------

def _intrinsic(S: float, K: float, is_call: bool) -> float:
    return max(S - K, 0.0) if is_call else max(K - S, 0.0)


def lsm_price(option: AmericanOption,
              n_paths: int = 10_000,
              n_steps: int = 100,
              degree: int = 3,
              seed: int = 42) -> LSMResult:
    """Longstaff-Schwartz Least-Squares Monte Carlo for American options."""
    S0, K, T, r, sigma, q = (
        option.S, option.K, option.T, option.r, option.sigma, option.q
    )
    is_call = option.is_call
    dt = T / n_steps
    df = math.exp(-r * dt)
    paths = simulate_gbm_paths(S0, r, q, sigma, T, n_paths, n_steps, seed)
    n_paths_actual = len(paths)
    cashflows = [_intrinsic(paths[i][n_steps], K, is_call)
                 for i in range(n_paths_actual)]
    early_ex_boundary: List[Tuple[float, float]] = []
    for step in range(n_steps - 1, 0, -1):
        t = step * dt
        itm_idx = [i for i in range(n_paths_actual)
                   if _intrinsic(paths[i][step], K, is_call) > 0]
        if not itm_idx:
            cashflows = [cf * df for cf in cashflows]
            continue
        S_itm = [paths[i][step] for i in itm_idx]
        X_reg = [_laguerre_basis(S / K, degree) for S in S_itm]
        Y_reg = [cashflows[i] * df for i in itm_idx]
        coeffs = _ols(X_reg, Y_reg)
        cont_vals = [sum(c * b for c, b in zip(coeffs, _laguerre_basis(S / K, degree)))
                     for S in S_itm]
        critical_S_values = []
        for local_idx, global_idx in enumerate(itm_idx):
            S_t = paths[global_idx][step]
            payoff = _intrinsic(S_t, K, is_call)
            cont = cont_vals[local_idx]
            if payoff >= cont and payoff > 0:
                cashflows[global_idx] = payoff
                critical_S_values.append(S_t)
            else:
                cashflows[global_idx] *= df
        if critical_S_values:
            early_ex_boundary.append((t, sum(critical_S_values) / len(critical_S_values)))
    price = sum(cf * df for cf in cashflows) / n_paths_actual
    prices = [cf * df for cf in cashflows]
    mu = sum(prices) / n_paths_actual
    var = sum((p - mu) ** 2 for p in prices) / (n_paths_actual - 1)
    std_err = math.sqrt(var / n_paths_actual)
    delta = _lsm_delta(option, n_paths, n_steps, degree, seed)
    early_ex_boundary.sort(key=lambda x: x[0])
    return LSMResult(price=price, std_error=std_err, std_error=std_err,
                      delta=delta, early_exercise_boundary=early_ex_boundary,
                      n_paths=n_paths_actual, n_steps=n_steps)


def _lsm_delta(option: AmericanOption, n_paths: int, n_steps: int,
               degree: int, seed: int, dS: float = 0.5) -> float:
    opt_up = AmericanOption(S=option.S + dS, K=option.K, T=option.T,
                             r=option.r, sigma=option.sigma, q=option.q,
                             is_call=option.is_call)
    opt_dn = AmericanOption(S=option.S - dS, K=option.K, T=option.T,
                             r=option.r, sigma=option.sigma, q=option.q,
                             is_call=option.is_call)
    p_up = lsm_price(opt_up, n_paths=max(n_paths // 5, 1000),
                      n_steps=n_steps, degree=degree, seed=seed)
    p_dn = lsm_price(opt_dn, n_paths=max(n_paths // 5, 1000),
                      n_steps=n_steps, degree=degree, seed=seed)
    return (p_up.price - p_dn.price) / (2 * dS)


# ---------------------------------------------------------------------------
# Binomial tree reference
# ---------------------------------------------------------------------------

def binomial_american(option: AmericanOption, n_steps: int = 500) -> float:
    """CRR binomial tree for American options."""
    S, K, T, r, sigma, q = (
        option.S, option.K, option.T, option.r, option.sigma, option.q
    )
    is_call = option.is_call
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    df = math.exp(-r * dt)
    prices = [S * u ** (n_steps - 2 * j) for j in range(n_steps + 1)]
    values = [_intrinsic(p_i, K, is_call) for p_i in prices]
    for step in range(n_steps - 1, -1, -1):
        for j in range(step + 1):
            S_node = S * u ** (step - 2 * j)
            cont = df * (p * values[j] + (1 - p) * values[j + 1])
            ex = _intrinsic(S_node, K, is_call)
            values[j] = max(cont, ex)
    return values[0]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    opt = AmericanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.02,
                         is_call=False)
    ref = binomial_american(opt, n_steps=500)
    print(f"Binomial tree (500 steps): {ref:.4f}")
    result = lsm_price(opt, n_paths=20_000, n_steps=100, degree=3)
    print(f"LSM Monte Carlo: {result.price:.4f} Â± {result.std_error:.4f}")
    print(f"Delta: {result.delta:.4f}")
