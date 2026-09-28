"""
Variance Gamma Model â Pricing and Calibration
Day 17 â options-greeks-pricer/models/variance_gamma.py

Implements:
  - Variance Gamma (VG) process via Brownian subordination
  - Characteristic function (Carr-Madan) pricing
  - Monte Carlo VG call/put with antithetic variates
  - Implied volatility extraction from VG prices
  - Calibration to market smile
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Normal / Gamma helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * _norm_pdf(d1) * math.sqrt(T)


def _gamma_sample(shape: float, scale: float, rng: random.Random) -> float:
    """Sample from Gamma(shape, scale) using Marsaglia-Tsang method."""
    if shape < 1.0:
        return _gamma_sample(shape + 1.0, scale, rng) * rng.random() ** (1.0 / shape)
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.gauss(0, 1)
        v = (1.0 + c * x) ** 3
        if v <= 0:
            continue
        u = rng.random()
        if u < 1.0 - 0.0331 * x ** 4:
            return d * v * scale
        if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
            return d * v * scale


# ---------------------------------------------------------------------------
# VG parameters
# ---------------------------------------------------------------------------

@dataclass
class VGParams:
    """
    Variance Gamma model parameters.
    S_t = S_0 * exp((r + omega) t + X(t; sigma, nu, theta))
    where omega = (1/nu) * ln(1 - theta*nu - 0.5*sigma^2*nu) is the martingale correction.
    sigma: vol of the Brownian motion
    nu:    variance rate of the Gamma subordinator (controls kurtosis)
    theta: drift of the Brownian motion (controls skew)
    """
    sigma: float = 0.20
    nu: float = 0.10
    theta: float = -0.10

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        if self.nu <= 0:
            raise ValueError("nu must be > 0")
        # Check condition for risk-neutral measure existence
        if 1 - self.theta * self.nu - 0.5 * self.sigma ** 2 * self.nu <= 0:
            raise ValueError("Parameters violate martingale condition")

    @property
    def omega(self) -> float:
        """Martingale correction term."""
        return math.log(1 - self.theta * self.nu - 0.5 * self.sigma ** 2 * self.nu) / self.nu

    @property
    def skewness(self) -> float:
        """Theoretical skewness of log-returns."""
        m2 = self.sigma ** 2 + self.theta ** 2 * self.nu
        m3 = 2 * self.theta ** 3 * self.nu ** 2 + 3 * self.sigma ** 2 * self.theta * self.nu
        return m3 / m2 ** 1.5

    @property
    def excess_kurtosis(self) -> float:
        """Theoretical excess kurtosis of log-returns."""
        m2 = self.sigma ** 2 + self.theta ** 2 * self.nu
        m4 = (3 * self.sigma ** 4 * self.nu + 12 * self.sigma ** 2 * self.theta ** 2 * self.nu ** 2
              + 6 * self.theta ** 4 * self.nu ** 3)
        return m4 / m2 ** 2


# ---------------------------------------------------------------------------
# Monte Carlo VG pricing
# ---------------------------------------------------------------------------

def vg_path_mc(S0: float, T: float, r: float, params: VGParams,
               n_steps: int = 252, seed: int = 42) -> List[float]:
    """Simulate a single VG price path by subordinating Brownian motion."""
    rng = random.Random(seed)
    dt = T / n_steps
    shape = dt / params.nu   # Gamma shape for each step
    scale = params.nu         # Gamma scale

    path = [S0]
    log_S = math.log(S0)
    for _ in range(n_steps):
        g = _gamma_sample(shape, scale, rng)
        z = rng.gauss(0, 1)
        dX = params.theta * g + params.sigma * math.sqrt(g) * z
        log_S += (r + params.omega) * dt + dX
        path.append(math.exp(log_S))
    return path


def vg_call_mc(S: float, K: float, T: float, r: float, params: VGParams,
               n_paths: int = 100_000, seed: int = 42) -> float:
    """Monte Carlo VG call price with antithetic variates."""
    rng = random.Random(seed)
    shape = T / params.nu
    scale = params.nu
    df = math.exp(-r * T)
    drift = (r + params.omega) * T

    payoffs = []
    half = n_paths // 2
    for _ in range(half):
        g = _gamma_sample(shape, scale, rng)
        z = rng.gauss(0, 1)
        for sign in [1, -1]:
            dX = params.theta * g + params.sigma * math.sqrt(g) * sign * z
            S_T = S * math.exp(drift + dX)
            payoffs.append(max(S_T - K, 0.0))

    return df * sum(payoffs) / len(payoffs)


def vg_put_mc(S: float, K: float, T: float, r: float, params: VGParams,
              n_paths: int = 100_000, seed: int = 42) -> float:
    """Monte Carlo VG put price via put-call parity."""
    call = vg_call_mc(S, K, T, r, params, n_paths, seed)
    return call - S + K * math.exp(-r * T)


# ---------------------------------------------------------------------------
# Characteristic function (Carr-Madan) pricing
# ---------------------------------------------------------------------------

def _cexp(z: complex) -> complex:
    re = z.real
    if re > 700:
        return complex(0, 0)
    return math.exp(re) * complex(math.cos(z.imag), math.sin(z.imag))


def vg_cf(params: VGParams, S: float, T: float, r: float, phi: complex) -> complex:
    """VG characteristic function of log(S_T) under risk-neutral measure."""
    i = complex(0, 1)
    log_S = math.log(S)
    # phi * (log S + (r + omega) T)
    linear = i * phi * (log_S + (r + params.omega) * T)
    # VG part: ln(1 - i*phi*theta*nu - 0.5*sigma^2*nu*phi^2) * (-T/nu)
    inner = 1.0 - i * phi * params.theta * params.nu - 0.5 * params.sigma ** 2 * params.nu * (phi ** 2)
    # Complex log
    ln_inner = complex(math.log(abs(inner)), math.atan2(inner.imag, inner.real))
    vg_part = -(T / params.nu) * ln_inner
    return _cexp(linear + vg_part)


def vg_call_cf(S: float, K: float, T: float, r: float, params: VGParams,
               alpha: float = 1.5, n_points: int = 128, dphi: float = 0.25) -> float:
    """
    Carr-Madan FFT-style call price via characteristic function.
    Uses the dampened call formula: C = e^{-alpha*k} / pi * Re[int e^{-i*phi*k} * psi(phi) dphi]
    where k = log(K/S).
    """
    i = complex(0, 1)
    k = math.log(K)  # log-strike

    def psi(phi: complex) -> complex:
        """Modified CF for Carr-Madan."""
        cf_val = vg_cf(params, S, T, r, phi - i * (alpha + 1))
        denom = alpha ** 2 + alpha - phi ** 2 + i * (2 * alpha + 1) * phi
        if abs(denom) < 1e-14:
            return complex(0, 0)
        return math.exp(-r * T) * cf_val / denom

    integral = complex(0, 0)
    for j in range(1, n_points + 1):
        phi_j = dphi * j
        integral += _cexp(-i * phi_j * k) * psi(phi_j) * dphi

    price = math.exp(-alpha * k) / math.pi * integral.real
    return max(price, 0.0)


# ---------------------------------------------------------------------------
# Implied vol from VG price
# ---------------------------------------------------------------------------

def vg_implied_vol(S: float, K: float, T: float, r: float, params: VGParams,
                   use_mc: bool = False, n_paths: int = 50_000) -> float:
    """Convert VG call price to BS implied vol via Newton-Raphson."""
    if use_mc:
        target = vg_call_mc(S, K, T, r, params, n_paths)
    else:
        target = vg_call_cf(S, K, T, r, params)

    sigma = math.sqrt(params.sigma ** 2 + params.theta ** 2 * params.nu)  # starting guess

    for _ in range(100):
        price = _bs_call(S, K, T, r, sigma)
        vega = _bs_vega(S, K, T, r, sigma)
        if abs(vega) < 1e-10:
            break
        sigma -= (price - target) / vega
        sigma = max(1e-4, min(5.0, sigma))
        if abs(price - target) < 1e-7:
            break
    return sigma


# ---------------------------------------------------------------------------
# Calibration (coordinate descent)
# ---------------------------------------------------------------------------

def _vg_loss(params: VGParams, S: float, r: float,
             market: List[Tuple[float, float, float]]) -> float:
    """RMSE between VG implied vols and market implied vols."""
    total, n = 0.0, 0
    for K, T, mkt_iv in market:
        try:
            vg_iv = vg_implied_vol(S, K, T, r, params)
            total += (vg_iv - mkt_iv) ** 2
            n += 1
        except Exception:
            pass
    return math.sqrt(total / max(n, 1))


def calibrate_vg(S: float, r: float,
                 market_vols: List[Tuple[float, float, float]],
                 max_iter: int = 200,
                 tol: float = 1e-6) -> VGParams:
    """
    Calibrate VG parameters to market (K, T, implied_vol) triples.
    Uses coordinate descent with adaptive step sizes.
    """
    best = VGParams(sigma=0.20, nu=0.10, theta=-0.10)
    best_loss = _vg_loss(best, S, r, market_vols)

    param_names = ["sigma", "nu", "theta"]
    steps = {"sigma": 0.02, "nu": 0.02, "theta": 0.02}
    bounds = {"sigma": (0.01, 2.0), "nu": (0.001, 2.0), "theta": (-2.0, 2.0)}

    for _ in range(max_iter):
        improved = False
        for name in param_names:
            for sign in [1, -1]:
                lo, hi = bounds[name]
                new_val = max(lo, min(hi, getattr(best, name) + sign * steps[name]))
                d = {"sigma": best.sigma, "nu": best.nu, "theta": best.theta}
                d[name] = new_val
                try:
                    candidate = VGParams(**d)
                except ValueError:
                    continue
                loss = _vg_loss(candidate, S, r, market_vols)
                if loss < best_loss - tol:
                    best_loss, best = loss, candidate
                    improved = True
            steps[name] *= 0.95
        if not improved:
            break

    return best


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S, r = 100.0, 0.05
    params = VGParams(sigma=0.20, nu=0.10, theta=-0.15)

    print(f"VG params: Ï={params.sigma}, Î½={params.nu}, Î¸={params.theta}")
    print(f"  Ï (martingale correction): {params.omega:.6f}")
    print(f"  Skewness: {params.skewness:.4f}")
    print(f"  Excess kurtosis: {params.excess_kurtosis:.4f}")
    print()

    strikes = [90, 95, 100, 105, 110]
    tenors = [0.25, 0.5, 1.0]

    print(f"{'K':>6} {'T':>5} {'VG-CF':>8} {'VG-MC':>8} {'IV-CF':>8}")
    print("-" * 40)
    for T in tenors:
        for K in strikes:
            cf_price = vg_call_cf(S, K, T, r, params)
            mc_price = vg_call_mc(S, K, T, r, params, n_paths=50_000)
            iv = vg_implied_vol(S, K, T, r, params)
            print(f"{K:>6} {T:>5.2f} {cf_price:>8.4f} {mc_price:>8.4f} {iv:>8.4%}")

    # Calibration demo
    print("\nCalibration test:")
    true_params = VGParams(sigma=0.18, nu=0.08, theta=-0.12)
    market: List[Tuple[float, float, float]] = []
    for T in [0.25, 0.5, 1.0]:
        for K in [90, 95, 100, 105, 110]:
            iv = vg_implied_vol(S, K, T, r, true_params)
            market.append((K, T, iv))

    calib = calibrate_vg(S, r, market, max_iter=100)
    print(f"  True:  Ï={true_params.sigma:.3f}, Î½={true_params.nu:.3f}, Î¸={true_params.theta:.3f}")
    print(f"  Calib: Ï={calib.sigma:.3f}, Î½={calib.nu:.3f}, Î¸={calib.theta:.3f}")
    print(f"  RMSE (IV): {_vg_loss(calib, S, r, market):.4%}")
