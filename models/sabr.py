"""
SABR Stochastic Volatility Model
Day 15 â options-greeks-pricer/models/sabr.py

Implements the SABR model (Hagan et al. 2002):
  dF = Ï F^Î² dW1
  dÏ = Î± Ï dW2
  dW1 dW2 = Ï dt

Provides the Hagan et al. implied volatility approximation and
calibration to a volatility smile.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# SABR parameters
# ---------------------------------------------------------------------------

@dataclass
class SABRParams:
    alpha: float   # initial vol-of-vol
    beta: float    # CEV exponent â [0, 1]
    rho: float     # correlation â (-1, 1)
    nu: float      # vol-of-vol
    F: float       # forward price
    T: float       # time to expiry (years)


# ---------------------------------------------------------------------------
# Hagan et al. implied vol formula
# ---------------------------------------------------------------------------

def sabr_implied_vol(params: SABRParams, K: float) -> float:
    """
    Compute Black implied vol from SABR parameters using Hagan et al. (2002)
    approximation. Valid for K != F and K > 0, F > 0.
    """
    alpha, beta, rho, nu, F, T = (
        params.alpha, params.beta, params.rho,
        params.nu, params.F, params.T
    )

    if T <= 0:
        return max(alpha * (F ** (beta - 1)), 1e-8)

    # ATM case or near-ATM
    if abs(F - K) < 1e-8 * F:
        return _sabr_atm(alpha, beta, rho, nu, F, T)

    FK = F * K
    FK_mid = FK ** ((1 - beta) / 2)
    log_FK = math.log(F / K)

    # z and x(z)
    z = (nu / alpha) * FK_mid * log_FK
    if abs(z) < 1e-6:
        x_z = 1.0
    else:
        disc = math.sqrt(1 - 2 * rho * z + z ** 2)
        arg = (disc + z - rho) / (1 - rho)
        if arg <= 0:
            arg = 1e-10
        x_z = math.log(arg) / z

    # Numerator: Ï_B(K, F)
    one_minus_beta = 1 - beta
    A = alpha / (FK_mid * (1 + (one_minus_beta ** 2 / 24) * log_FK ** 2
                            + (one_minus_beta ** 4 / 1920) * log_FK ** 4))

    B = 1 + T * (
        (one_minus_beta ** 2 / 24) * alpha ** 2 / (FK ** (one_minus_beta))
        + (rho * beta * nu * alpha) / (4 * FK_mid)
        + (2 - 3 * rho ** 2) * nu ** 2 / 24
    )

    return A * (z / x_z) * B


def _sabr_atm(alpha: float, beta: float, rho: float,
              nu: float, F: float, T: float) -> float:
    """SABR implied vol at-the-money (K = F)."""
    one_minus_beta = 1 - beta
    F_beta = F ** (one_minus_beta)

    A = alpha / F_beta

    B = 1 + T * (
        (one_minus_beta ** 2 / 24) * (alpha / F_beta) ** 2
        + (rho * beta * nu * alpha) / (4 * F_beta)
        + (2 - 3 * rho ** 2) * nu ** 2 / 24
    )

    return A * B


# ---------------------------------------------------------------------------
# Black-76 pricing (for calibration)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def black76_call(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-76 call price."""
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0) * math.exp(-r * T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return math.exp(-r * T) * (F * _norm_cdf(d1) - K * _norm_cdf(d2))


def black76_put(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-76 put price."""
    if T <= 0 or sigma <= 0:
        return max(K - F, 0.0) * math.exp(-r * T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return math.exp(-r * T) * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


# ---------------------------------------------------------------------------
# SABR calibration
# ---------------------------------------------------------------------------

def sabr_smile(params: SABRParams, strikes: List[float]) -> List[float]:
    """Compute implied vol smile across strikes."""
    return [sabr_implied_vol(params, K) for K in strikes]


def _sabr_loss(alpha: float, beta: float, rho: float, nu: float,
               F: float, T: float,
               strikes: List[float],
               market_vols: List[float]) -> float:
    """SSE between SABR vols and market vols."""
    params = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu, F=F, T=T)
    total = 0.0
    for K, mv in zip(strikes, market_vols):
        try:
            sv = sabr_implied_vol(params, K)
        except (ValueError, ZeroDivisionError):
            sv = 0.0
        total += (sv - mv) ** 2
    return total


def calibrate_sabr(F: float,
                   T: float,
                   strikes: List[float],
                   market_vols: List[float],
                   beta: float = 0.5,
                   n_grid: int = 10,
                   tol: float = 1e-6,
                   max_iter: int = 100) -> SABRParams:
    """
    Calibrate SABR (alpha, rho, nu) with fixed beta via grid search + Nelder-Mead.

    beta is fixed (typically 0.5 or 1.0) to avoid over-parameterisation.
    """
    # Grid search over (alpha, rho, nu)
    best_loss = math.inf
    best = (0.3, 0.0, 0.3)

    alpha_grid = [0.05 + 0.4 * i / n_grid for i in range(n_grid + 1)]
    rho_grid = [-0.8 + 1.6 * i / n_grid for i in range(n_grid + 1)]
    nu_grid = [0.05 + 0.8 * i / n_grid for i in range(n_grid + 1)]

    for a in alpha_grid:
        for r in rho_grid:
            for v in nu_grid:
                loss = _sabr_loss(a, beta, r, v, F, T, strikes, market_vols)
                if loss < best_loss:
                    best_loss = loss
                    best = (a, r, v)

    # Refine with simple coordinate descent
    alpha, rho, nu = best
    step = 0.05

    for iteration in range(max_iter):
        improved = False
        for dim in range(3):
            params = [alpha, rho, nu]
            for sign in [+1, -1]:
                p = params[:]
                p[dim] += sign * step

                # Clamp to valid ranges
                p[0] = max(1e-4, p[0])
                p[1] = max(-0.999, min(0.999, p[1]))
                p[2] = max(1e-4, p[2])

                loss = _sabr_loss(p[0], beta, p[1], p[2], F, T,
                                   strikes, market_vols)
                if loss < best_loss:
                    best_loss = loss
                    alpha, rho, nu = p
                    improved = True
                    break

        step *= 0.95
        if step < tol:
            break

    return SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu, F=F, T=T)


# ---------------------------------------------------------------------------
# SABR Greeks (numerical)
# ---------------------------------------------------------------------------

def sabr_delta(params: SABRParams, K: float, r: float = 0.0,
               dF: float = 0.001) -> float:
    """Numerical delta via forward bump."""
    p_up = SABRParams(**{**params.__dict__, "F": params.F + dF})
    p_dn = SABRParams(**{**params.__dict__, "F": params.F - dF})
    call_up = black76_call(params.F + dF, K, params.T, r,
                            sabr_implied_vol(p_up, K))
    call_dn = black76_call(params.F - dF, K, params.T, r,
                            sabr_implied_vol(p_dn, K))
    return (call_up - call_dn) / (2 * dF)


def sabr_vega(params: SABRParams, K: float, r: float = 0.0,
              da: float = 0.001) -> float:
    """Vega w.r.t. alpha (vol-of-vol sensitivity)."""
    p_up = SABRParams(**{**params.__dict__, "alpha": params.alpha + da})
    p_dn = SABRParams(**{**params.__dict__, "alpha": params.alpha - da})
    iv_up = sabr_implied_vol(p_up, K)
    iv_dn = sabr_implied_vol(p_dn, K)
    call_up = black76_call(params.F, K, params.T, r, iv_up)
    call_dn = black76_call(params.F, K, params.T, r, iv_dn)
    return (call_up - call_dn) / (2 * da)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    F, T, r = 100.0, 0.5, 0.03

    true_params = SABRParams(alpha=0.30, beta=0.5, rho=-0.30, nu=0.40, F=F, T=T)

    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    market_vols = [sabr_implied_vol(true_params, K) for K in strikes]

    print("Market vol smile (true SABR):")
    print(f"  {'Strike':>8} {'Market IV':>12}")
    for K, mv in zip(strikes, market_vols):
        print(f"  {K:>8.0f}  {mv:>11.4%}")

    calib = calibrate_sabr(F, T, strikes, market_vols, beta=0.5, n_grid=8)
    print(f"\nCalibrated SABR: Î±={calib.alpha:.4f}, Ï={calib.rho:.4f}, Î½={calib.nu:.4f}")

    fitted_vols = sabr_smile(calib, strikes)
    max_err = max(abs(fv - mv) for fv, mv in zip(fitted_vols, market_vols))
    print(f"Max fit error: {max_err:.2e}")

    atm_iv = sabr_implied_vol(calib, F)
    atm_price = black76_call(F, F, T, r, atm_iv)
    delta = sabr_delta(calib, F, r)
    vega = sabr_vega(calib, F, r)
    print(f"\nATM call: price={atm_price:.4f}, delta={delta:.4f}, vega={vega:.4f}")
