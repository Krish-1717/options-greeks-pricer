"""
Dupire Local Volatility Surface
Day 15 â options-greeks-pricer/volatility/local_vol.py

Implements Dupire's (1994) formula to extract the local volatility surface
from observed call prices / implied vols, plus local vol Monte Carlo pricing.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Normal CDF and PDF
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


# ---------------------------------------------------------------------------
# Black-Scholes implied vol (Newton-Raphson)
# ---------------------------------------------------------------------------

def bs_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T)


def implied_vol(S: float, K: float, T: float, r: float, q: float,
                market_price: float, tol: float = 1e-8, max_iter: int = 100) -> float:
    """Newton-Raphson implied vol solver."""
    sigma = 0.30
    for _ in range(max_iter):
        price = bs_call(S, K, T, r, q, sigma)
        vega = bs_vega(S, K, T, r, q, sigma)
        if abs(vega) < 1e-10:
            break
        sigma -= (price - market_price) / vega
        sigma = max(1e-6, sigma)
        if abs(price - market_price) < tol:
            break
    return sigma


# ---------------------------------------------------------------------------
# Implied vol surface (interpolated)
# ---------------------------------------------------------------------------

@dataclass
class ImpliedVolSurface:
    """
    Bilinear interpolated implied vol surface on a (T, K) grid.
    """
    tenors: List[float]      # expiry grid
    strikes: List[float]     # strike grid
    vols: List[List[float]]  # vols[i][j] = Ï(T_i, K_j)
    S0: float
    r: float
    q: float = 0.0

    def vol(self, T: float, K: float) -> float:
        """Bilinearly interpolate (T, K) â Ï."""
        # Clamp to grid
        T = max(self.tenors[0], min(self.tenors[-1], T))
        K = max(self.strikes[0], min(self.strikes[-1], K))

        # Find bracketing indices
        ti = 0
        for i in range(len(self.tenors) - 1):
            if self.tenors[i] <= T <= self.tenors[i + 1]:
                ti = i
                break
        ki = 0
        for j in range(len(self.strikes) - 1):
            if self.strikes[j] <= K <= self.strikes[j + 1]:
                ki = j
                break

        T0, T1 = self.tenors[ti], self.tenors[min(ti + 1, len(self.tenors) - 1)]
        K0, K1 = self.strikes[ki], self.strikes[min(ki + 1, len(self.strikes) - 1)]

        wT = (T - T0) / (T1 - T0 + 1e-12)
        wK = (K - K0) / (K1 - K0 + 1e-12)

        v00 = self.vols[ti][ki]
        v01 = self.vols[ti][min(ki + 1, len(self.strikes) - 1)]
        v10 = self.vols[min(ti + 1, len(self.tenors) - 1)][ki]
        v11 = self.vols[min(ti + 1, len(self.tenors) - 1)][min(ki + 1, len(self.strikes) - 1)]

        v = ((1 - wT) * (1 - wK) * v00
             + (1 - wT) * wK * v01
             + wT * (1 - wK) * v10
             + wT * wK * v11)
        return max(v, 1e-6)

    def call_price(self, T: float, K: float) -> float:
        return bs_call(self.S0, K, T, self.r, self.q, self.vol(T, K))


# ---------------------------------------------------------------------------
# Dupire local volatility
# ---------------------------------------------------------------------------

def dupire_local_vol(surface: ImpliedVolSurface,
                     T: float, K: float,
                     dT: float = 1e-4, dK: Optional[float] = None) -> float:
    """
    Dupire (1994) formula:
      Ï_loc^2(K,T) = [âC/âT + (r-q) K âC/âK + q C]
                     / [0.5 K^2 â^2C/âK^2]

    Computed numerically from the implied vol surface.
    """
    if dK is None:
        dK = max(0.5, K * 0.01)

    C = surface.call_price(T, K)

    # Time derivative
    C_T_up = surface.call_price(T + dT, K)
    C_T_dn = surface.call_price(max(T - dT, 1e-6), K)
    dC_dT = (C_T_up - C_T_dn) / (2 * dT)

    # Strike derivatives
    K_up = min(K + dK, surface.strikes[-1])
    K_dn = max(K - dK, surface.strikes[0])
    C_K_up = surface.call_price(T, K_up)
    C_K_dn = surface.call_price(T, K_dn)
    dC_dK = (C_K_up - C_K_dn) / (K_up - K_dn)
    d2C_dK2 = (C_K_up - 2 * C + C_K_dn) / (dK ** 2)

    r, q = surface.r, surface.q

    numerator = dC_dT + (r - q) * K * dC_dK + q * C
    denominator = 0.5 * K ** 2 * d2C_dK2

    if abs(denominator) < 1e-10 or numerator / denominator < 0:
        # Fall back to implied vol squared
        return surface.vol(T, K) ** 2

    return numerator / denominator


class LocalVolSurface:
    """
    Precomputed local volatility surface on a (T, K) grid.
    """

    def __init__(self, implied: ImpliedVolSurface,
                 tenors: Optional[List[float]] = None,
                 strikes: Optional[List[float]] = None):
        self.tenors = tenors or implied.tenors
        self.strikes = strikes or implied.strikes
        self._implied = implied

        # Precompute
        self._grid: List[List[float]] = [
            [math.sqrt(max(dupire_local_vol(implied, T, K), 1e-8))
             for K in self.strikes]
            for T in self.tenors
        ]

    def local_vol(self, T: float, K: float) -> float:
        """Bilinear interpolation of local vol grid."""
        proxy = ImpliedVolSurface(
            tenors=self.tenors, strikes=self.strikes,
            vols=self._grid, S0=self._implied.S0,
            r=self._implied.r, q=self._implied.q
        )
        return proxy.vol(T, K)


# ---------------------------------------------------------------------------
# Local vol Monte Carlo
# ---------------------------------------------------------------------------

def local_vol_mc(lv_surface: LocalVolSurface,
                 S0: float, K: float, T: float, r: float, q: float,
                 is_call: bool = True,
                 n_paths: int = 10_000, n_steps: int = 100,
                 seed: int = 42) -> Tuple[float, float]:
    """
    Price an option under the local volatility model via Monte Carlo.
    Returns (price, std_error).
    """
    import random as _r
    rng = _r.Random(seed)
    dt = T / n_steps
    payoffs: List[float] = []

    half = n_paths // 2
    for path_i in range(half):
        S = S0
        S_anti = S0
        for step in range(n_steps):
            t = step * dt
            lv = lv_surface.local_vol(t, S)
            lv_anti = lv_surface.local_vol(t, S_anti)
            z = rng.gauss(0, 1)
            S = S * math.exp((r - q - 0.5 * lv ** 2) * dt + lv * math.sqrt(dt) * z)
            S_anti = S_anti * math.exp(
                (r - q - 0.5 * lv_anti ** 2) * dt - lv_anti * math.sqrt(dt) * z
            )

        if is_call:
            payoffs.append(max(S - K, 0.0) * math.exp(-r * T))
            payoffs.append(max(S_anti - K, 0.0) * math.exp(-r * T))
        else:
            payoffs.append(max(K - S, 0.0) * math.exp(-r * T))
            payoffs.append(max(K - S_anti, 0.0) * math.exp(-r * T))

    n = len(payoffs)
    price = sum(payoffs) / n
    std = math.sqrt(sum((p - price) ** 2 for p in payoffs) / (n - 1))
    return price, std / math.sqrt(n)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S0, r, q = 100.0, 0.05, 0.02

    # Build a simple implied vol surface (smile)
    tenors = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    strikes = [70, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130]

    def parametric_iv(T: float, K: float) -> float:
        """Simple smile: ATM vol + smile curvature."""
        atm_vol = 0.20 + 0.02 * math.sqrt(T)
        moneyness = math.log(K / S0) / (atm_vol * math.sqrt(T) + 1e-6)
        smile = 0.03 * moneyness ** 2 - 0.01 * moneyness
        return max(atm_vol + smile, 0.05)

    vols = [[parametric_iv(T, K) for K in strikes] for T in tenors]
    surface = ImpliedVolSurface(tenors=tenors, strikes=strikes,
                                vols=vols, S0=S0, r=r, q=q)

    # Build local vol surface
    lv = LocalVolSurface(surface)

    print("Local volatility surface (selected nodes):")
    print(f"{'T':>6} {'K':>6} {'IV':>8} {'LV':>8}")
    for T_test in [0.5, 1.0]:
        for K_test in [90, 100, 110]:
            iv = surface.vol(T_test, K_test)
            lvol = lv.local_vol(T_test, K_test)
            print(f"{T_test:>6.2f} {K_test:>6.0f} {iv:>7.4f}  {lvol:>7.4f}")

    # Price ATM call
    T_opt, K_opt = 1.0, 100.0
    bs_ref = bs_call(S0, K_opt, T_opt, r, q, surface.vol(T_opt, K_opt))
    mc_price, mc_se = local_vol_mc(lv, S0, K_opt, T_opt, r, q,
                                    is_call=True, n_paths=20_000, n_steps=50)
    print(f"\nATM Call (T=1, K=100):")
    print(f"  BS reference:    {bs_ref:.4f}")
    print(f"  Local vol MC:    {mc_price:.4f} Â± {mc_se:.4f}")
