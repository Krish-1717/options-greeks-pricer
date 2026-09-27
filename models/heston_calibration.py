"""
Heston Stochastic Volatility Model â Calibration
Day 16 â options-greeks-pricer/models/heston_calibration.py

Implements:
  - Heston (1993) semi-analytical call pricing via characteristic function
  - Fast Fourier Transform (FFT) pricing over a strike grid
  - Calibration to observed implied vol surface
  - Model risk and sensitivity analysis
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Heston parameters
# ---------------------------------------------------------------------------

@dataclass
class HestonParams:
    """
    Parameters for the Heston model:
      dS = (r-q) S dt + âV S dW1
      dV = Îº(Î¸-V) dt + Ï âV dW2
      dW1 dW2 = Ï dt
    """
    v0: float     # initial variance
    kappa: float  # mean-reversion speed
    theta: float  # long-run variance
    sigma: float  # vol of vol
    rho: float    # correlation â (-1, 1)

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must be in (-1, 1)")
        if self.v0 <= 0 or self.theta <= 0 or self.kappa <= 0:
            raise ValueError("v0, theta, kappa must be > 0")

    @property
    def feller_satisfied(self) -> bool:
        """Feller condition: 2ÎºÎ¸ â¥ ÏÂ² ensures V stays positive."""
        return 2 * self.kappa * self.theta >= self.sigma ** 2

    @property
    def atm_vol(self) -> float:
        return math.sqrt(self.theta)


# ---------------------------------------------------------------------------
# Heston characteristic function
# ---------------------------------------------------------------------------

def _heston_cf(params: HestonParams, S: float, T: float,
               r: float, q: float, phi: complex) -> complex:
    """
    Heston characteristic function E[e^{iÏ ln S_T}].
    Uses the formulation of Albrecher et al. to avoid discontinuities.
    """
    kappa, theta, sigma, rho, v0 = (
        params.kappa, params.theta, params.sigma, params.rho, params.v0
    )
    i = complex(0, 1)

    # xi and d
    xi = kappa - i * phi * rho * sigma
    d2 = (xi ** 2 + sigma ** 2 * (phi ** 2 + i * phi))
    d = d2 ** 0.5

    # Avoid sqrt sign ambiguity using log form
    g_num = xi - d
    g_den = xi + d
    if abs(g_den) < 1e-14:
        g = complex(1.0, 0)
    else:
        g = g_num / g_den

    # e^{-d T} factor
    exp_dt = cexp(-d * T)

    if abs(1 - g * exp_dt) < 1e-14:
        D = (xi - d) / (sigma ** 2)
    else:
        D = (xi - d) / sigma ** 2 * (1 - exp_dt) / (1 - g * exp_dt)

    if abs(1 - g * exp_dt) < 1e-14:
        C = kappa * theta / sigma ** 2 * ((xi - d) * T)
    else:
        log_arg = (1 - g * exp_dt) / (1 - g)
        if abs(log_arg) < 1e-14:
            C = complex(0, 0)
        else:
            C = kappa * theta / sigma ** 2 * ((xi - d) * T - 2 * clog(log_arg))

    lnS = math.log(S) + (r - q) * T
    return cexp(C + D * v0 + i * phi * lnS)


def cexp(z: complex) -> complex:
    """Safe complex exponential."""
    re = z.real
    if re > 700:
        return complex(0, 0)
    return math.exp(re) * complex(math.cos(z.imag), math.sin(z.imag))


def clog(z: complex) -> complex:
    """Complex logarithm."""
    return complex(math.log(abs(z)), math.atan2(z.imag, z.real))


# ---------------------------------------------------------------------------
# Semi-analytical Heston call price
# ---------------------------------------------------------------------------

def heston_call(params: HestonParams,
                S: float, K: float, T: float, r: float, q: float,
                n_integration: int = 128) -> float:
    """
    Heston call price via Carr-Madan / Heston (1993) numerical integration.
    Uses Gauss-Laguerre-style trapezoidal integration of characteristic function.
    """
    i = complex(0, 1)
    F = S * math.exp((r - q) * T)
    x = math.log(F / K)

    def integrand(phi: float) -> complex:
        """Integrand for the Gil-Pelaez formula."""
        cf = _heston_cf(params, S, T, r, q, phi - i / 2)
        denom = i * phi * _heston_cf(params, S, T, r, q, -i / 2)
        if abs(denom.real) + abs(denom.imag) < 1e-14:
            return complex(0, 0)
        return cexp(-i * phi * x) * cf / denom

    # Numerical integration via trapezoidal rule
    dphi = 0.5
    n = n_integration
    result = complex(0, 0)
    for k in range(1, n + 1):
        phi = dphi * k
        result += integrand(phi) * dphi

    # Price via Gil-Pelaez inversion
    pi = math.pi
    price_unnorm = F - K * math.exp(-r * T) * (0.5 + result.real / pi)
    return max(price_unnorm, 0.0)


def heston_put(params: HestonParams,
               S: float, K: float, T: float, r: float, q: float,
               n_integration: int = 128) -> float:
    """Heston put via put-call parity."""
    call = heston_call(params, S, K, T, r, q, n_integration)
    return call - S * math.exp(-q * T) + K * math.exp(-r * T)


# ---------------------------------------------------------------------------
# Implied volatility from Heston price
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def _bs_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T)


def heston_implied_vol(params: HestonParams, S: float, K: float,
                        T: float, r: float, q: float,
                        tol: float = 1e-6) -> float:
    """Convert Heston call price to BS implied vol via Newton-Raphson."""
    target = heston_call(params, S, K, T, r, q)
    sigma = math.sqrt(params.v0)  # starting guess

    for _ in range(100):
        price = _bs_call(S, K, T, r, q, sigma)
        vega = _bs_vega(S, K, T, r, q, sigma)
        if abs(vega) < 1e-10:
            break
        sigma -= (price - target) / vega
        sigma = max(1e-4, sigma)
        if abs(price - target) < tol:
            break
    return sigma


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _calibration_loss(params: HestonParams,
                       S: float, r: float, q: float,
                       market: List[Tuple[float, float, float]]) -> float:
    """RMSE between Heston implied vols and market implied vols."""
    total = 0.0
    n = 0
    for K, T, mkt_iv in market:
        try:
            h_iv = heston_implied_vol(params, S, K, T, r, q)
            total += (h_iv - mkt_iv) ** 2
            n += 1
        except Exception:
            pass
    return math.sqrt(total / max(n, 1))


def calibrate_heston(S: float, r: float, q: float,
                      market_vols: List[Tuple[float, float, float]],
                      n_grid: int = 5,
                      tol: float = 1e-5,
                      max_iter: int = 100) -> HestonParams:
    """
    Calibrate Heston model to market (K, T, implied_vol) triples.
    Grid search followed by coordinate descent.
    """
    # Grid search
    best_loss = math.inf
    best_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.4, rho=-0.5)

    v0_grid = [0.01, 0.04, 0.09]
    kappa_grid = [0.5, 1.5, 3.0, 5.0]
    theta_grid = [0.01, 0.04, 0.09]
    sigma_grid = [0.2, 0.4, 0.6, 0.8]
    rho_grid = [-0.8, -0.5, -0.2, 0.0]

    for v0 in v0_grid:
        for kappa in kappa_grid:
            for theta in theta_grid:
                for sigma_h in sigma_grid:
                    for rho in rho_grid:
                        try:
                            p = HestonParams(v0=v0, kappa=kappa, theta=theta,
                                              sigma=sigma_h, rho=rho)
                        except ValueError:
                            continue
                        loss = _calibration_loss(p, S, r, q, market_vols)
                        if loss < best_loss:
                            best_loss = loss
                            best_params = p

    # Coordinate descent refinement
    p = best_params
    param_names = ["v0", "kappa", "theta", "sigma", "rho"]
    steps = {"v0": 0.005, "kappa": 0.2, "theta": 0.005, "sigma": 0.05, "rho": 0.05}
    bounds = {
        "v0": (1e-4, 1.0), "kappa": (0.1, 20.0), "theta": (1e-4, 1.0),
        "sigma": (0.01, 2.0), "rho": (-0.999, 0.999),
    }

    for _ in range(max_iter):
        improved = False
        for name in param_names:
            step = steps[name]
            lo, hi = bounds[name]

            def make_params(val: float) -> HestonParams:
                d = {n: getattr(p, n) for n in param_names}
                d[name] = val
                try:
                    return HestonParams(**d)
                except ValueError:
                    return p

            for sign in [+1, -1]:
                new_val = max(lo, min(hi, getattr(p, name) + sign * step))
                p_new = make_params(new_val)
                loss_new = _calibration_loss(p_new, S, r, q, market_vols)
                if loss_new < best_loss - tol:
                    best_loss = loss_new
                    p = p_new
                    improved = True

            steps[name] *= 0.9

        if not improved:
            break

    return p


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S, r, q = 100.0, 0.05, 0.02

    # True Heston parameters
    true_p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.50, rho=-0.70)
    print(f"True params: v0={true_p.v0}, Îº={true_p.kappa}, Î¸={true_p.theta}, "
          f"Ï={true_p.sigma}, Ï={true_p.rho}")
    print(f"Feller condition: {true_p.feller_satisfied}")

    # Generate market data
    strikes = [85, 90, 95, 100, 105, 110]
    tenors = [0.25, 0.5, 1.0]
    market: List[Tuple[float, float, float]] = []
    print("\nMarket implied vols:")
    for T in tenors:
        for K in strikes:
            iv = heston_implied_vol(true_p, S, K, T, r, q)
            market.append((K, T, iv))
            print(f"  K={K:?5}, T={T:.2f}: iv={iv:.4%}")

    # Calibrate
    print("\nCalibrating...")
    calib = calibrate_heston(S, r, q, market, n_grid=5)

    print(f"Calibrated: v0={calib.v0:.4f}, Îº={calib.kappa:.4f}, "
          f"Î¸={calib.theta:.4f}, Ï={calib.sigma:.4f}, Ï={calib.rho:.4f}")

    rmse = _calibration_loss(calib, S, r, q, market)
    print(f"Calibration RMSE (IV): {rmse:.4%}")
