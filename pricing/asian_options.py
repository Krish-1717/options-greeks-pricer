"""
Asian Option Pricing
Day 18 â options-greeks-pricer/pricing/asian_options.py

Implements:
  - Geometric Asian call/put: closed-form Black-Scholes-style formula
  - Arithmetic Asian call/put: Monte Carlo with geometric control variate
  - Continuous and discrete monitoring
  - Floating vs fixed strike variants
  - Asian Greeks via finite difference
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------

def _N(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _n(x: float) -> float:
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


# ---------------------------------------------------------------------------
# Asian option data structure
# ---------------------------------------------------------------------------

@dataclass
class AsianOption:
    S: float              # spot price
    K: Optional[float]    # fixed strike (None = floating strike)
    T: float              # time to expiry
    r: float              # risk-free rate
    q: float              # dividend yield
    sigma: float          # volatility
    n_obs: int = 252      # number of observation dates
    is_call: bool = True
    floating_strike: bool = False   # if True, K = average (floating strike)


# ---------------------------------------------------------------------------
# Geometric Asian â closed-form
# ---------------------------------------------------------------------------

def _geo_asian_params(opt: AsianOption) -> Tuple[float, float]:
    """
    Adjusted parameters for geometric average approximation.
    For continuous monitoring:
      sigma_g = sigma / sqrt(3)
      b_g     = 0.5 * (r - q - sigma^2/6)
    For discrete monitoring (Kemna-Vorst):
      sigma_gÂ² = sigmaÂ² * (n+1)(2n+1) / (6nÂ²)
      mu_g    = (b - sigmaÂ²/2) * (n+1)/(2n) + sigmaÂ²_g / 2
    """
    n = opt.n_obs
    b = opt.r - opt.q  # cost of carry

    # Discrete-monitoring adjusted vol (Kemna-Vorst approximation)
    sigma_g_sq = opt.sigma ** 2 * (n + 1) * (2 * n + 1) / (6 * n ** 2)
    sigma_g = math.sqrt(sigma_g_sq)

    # Adjusted drift
    mu_g = (b - 0.5 * opt.sigma ** 2) * (n + 1) / (2 * n) + 0.5 * sigma_g_sq
    return sigma_g, mu_g


def geometric_asian_call(opt: AsianOption) -> float:
    """
    Closed-form geometric Asian call price (Kemna-Vorst 1990, discrete monitoring).
    Fixed strike only.
    """
    assert not opt.floating_strike, "Floating strike not supported for closed-form geo Asian"
    T, r, K = opt.T, opt.r, opt.K
    sigma_g, mu_g = _geo_asian_params(opt)

    if T <= 0 or sigma_g <= 0:
        return max(opt.S - K, 0.0) * math.exp(-r * T)

    F_g = opt.S * math.exp(mu_g * T)
    d1 = (math.log(F_g / K) + 0.5 * sigma_g ** 2 * T) / (sigma_g * math.sqrt(T))
    d2 = d1 - sigma_g * math.sqrt(T)

    return math.exp(-r * T) * (F_g * _N(d1) - K * _N(d2))


def geometric_asian_put(opt: AsianOption) -> float:
    """Closed-form geometric Asian put price."""
    assert not opt.floating_strike
    T, r, K = opt.T, opt.r, opt.K
    sigma_g, mu_g = _geo_asian_params(opt)

    if T <= 0 or sigma_g <= 0:
        return max(K - opt.S, 0.0) * math.exp(-r * T)

    F_g = opt.S * math.exp(mu_g * T)
    d1 = (math.log(F_g / K) + 0.5 * sigma_g ** 2 * T) / (sigma_g * math.sqrt(T))
    d2 = d1 - sigma_g * math.sqrt(T)

    return math.exp(-r * T) * (K * _N(-d2) - F_g * _N(-d1))


def geometric_asian(opt: AsianOption) -> float:
    return geometric_asian_call(opt) if opt.is_call else geometric_asian_put(opt)


# ---------------------------------------------------------------------------
# Arithmetic Asian â Monte Carlo with geometric control variate
# ---------------------------------------------------------------------------

def _simulate_path(S0: float, r: float, q: float, sigma: float,
                   T: float, n_obs: int, rng: random.Random,
                   antithetic: bool = False, z_vals: Optional[List[float]] = None
                   ) -> Tuple[List[float], List[float]]:
    """
    Simulate a GBM path at n_obs equally spaced observation dates.
    Returns (path_orig, path_anti) if antithetic else (path_orig, path_orig).
    """
    dt = T / n_obs
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol_sqrt_dt = sigma * math.sqrt(dt)

    prices_orig, prices_anti = [S0], [S0]
    for _ in range(n_obs):
        z = rng.gauss(0, 1) if z_vals is None else z_vals[_]
        prices_orig.append(prices_orig[-1] * math.exp(drift + vol_sqrt_dt * z))
        if antithetic:
            prices_anti.append(prices_anti[-1] * math.exp(drift - vol_sqrt_dt * z))

    return prices_orig[1:], prices_anti[1:] if antithetic else prices_orig[1:]


def arithmetic_asian_mc(
        opt: AsianOption,
        n_paths: int = 100_000,
        seed: int = 42,
        use_control_variate: bool = True,
) -> Tuple[float, float]:
    """
    Monte Carlo arithmetic Asian option with geometric control variate.

    Returns (price, std_error).
    """
    rng = random.Random(seed)
    S0, K, T, r, q, sigma, n_obs = (
        opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma, opt.n_obs
    )
    dt = T / n_obs
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol_sqrt_dt = sigma * math.sqrt(dt)
    df = math.exp(-r * T)

    # Geometric control variate price (closed-form)
    if use_control_variate and not opt.floating_strike:
        cv_price = geometric_asian(opt)

    payoffs_cv: List[float] = []
    payoffs_raw: List[float] = []

    half = n_paths // 2
    for _ in range(half):
        for sign in [1, -1]:
            # Simulate path
            prices = [S0]
            geo_prices = [S0]
            for _ in range(n_obs):
                z = rng.gauss(0, 1) * sign
                prices.append(prices[-1] * math.exp(drift + vol_sqrt_dt * z))
                geo_prices.append(geo_prices[-1] * math.exp(drift + vol_sqrt_dt * z))

            obs = prices[1:]
            arith_avg = sum(obs) / n_obs
            geo_avg = math.exp(sum(math.log(p) for p in obs) / n_obs)

            if opt.floating_strike:
                # Floating strike: K = A (arithmetic average)
                K_eff = arith_avg
                payoff_arith = max(K_eff - obs[-1], 0.0) if opt.is_call else max(obs[-1] - K_eff, 0.0)
            else:
                payoff_arith = max(arith_avg - K, 0.0) if opt.is_call else max(K - arith_avg, 0.0)
                payoff_geo = max(geo_avg - K, 0.0) if opt.is_call else max(K - geo_avg, 0.0)
                payoffs_raw.append(df * payoff_arith)

                if use_control_variate:
                    payoffs_cv.append(df * (payoff_arith - payoff_geo))

    if opt.floating_strike or not use_control_variate:
        raw_prices = [df * p for p in payoffs_raw] if payoffs_raw else [0.0]
        n = len(raw_prices)
        mu = sum(raw_prices) / n
        var = sum((p - mu) ** 2 for p in raw_prices) / max(n - 1, 1)
        return mu, math.sqrt(var / n)

    # Control variate adjustment
    n = len(payoffs_cv)
    mu_cv = sum(payoffs_cv) / n
    mu_raw = sum(payoffs_raw) / n

    # Optimal beta (covariance / variance of control)
    cov = sum((payoffs_cv[i] - mu_cv) * (payoffs_raw[i] - mu_raw) for i in range(n)) / n
    var_cv = sum((p - mu_cv) ** 2 for p in payoffs_cv) / n
    beta = cov / var_cv if var_cv > 1e-12 else 0.0

    cv_adjusted = [payoffs_raw[i] - beta * (payoffs_cv[i] - 0.0) for i in range(n)]
    # Centre: E[payoff_geo * df] â cv_price
    cv_adjusted = [payoffs_raw[i] - beta * payoffs_cv[i] + beta * cv_price for i in range(n)]

    mu = sum(cv_adjusted) / n
    var = sum((p - mu) ** 2 for p in cv_adjusted) / max(n - 1, 1)
    return mu, math.sqrt(var / n)


# ---------------------------------------------------------------------------
# Asian Greeks (finite difference)
# ---------------------------------------------------------------------------

def asian_delta(opt: AsianOption, dS: float = 0.5, use_mc: bool = False,
                n_paths: int = 50_000) -> float:
    from dataclasses import replace
    up = replace(opt, S=opt.S + dS)
    dn = replace(opt, S=opt.S - dS)
    if use_mc:
        pu = arithmetic_asian_mc(up, n_paths)[0]
        pd = arithmetic_asian_mc(dn, n_paths)[0]
    else:
        pu = geometric_asian(up)
        pd = geometric_asian(dn)
    return (pu - pd) / (2 * dS)


def asian_gamma(opt: AsianOption, dS: float = 0.5) -> float:
    from dataclasses import replace
    up = geometric_asian(replace(opt, S=opt.S + dS))
    mid = geometric_asian(opt)
    dn = geometric_asian(replace(opt, S=opt.S - dS))
    return (up - 2 * mid + dn) / dS ** 2


def asian_vega(opt: AsianOption, dsigma: float = 0.01) -> float:
    from dataclasses import replace
    up = geometric_asian(replace(opt, sigma=opt.sigma + dsigma))
    dn = geometric_asian(replace(opt, sigma=opt.sigma - dsigma))
    return (up - dn) / (2 * dsigma)


def asian_theta(opt: AsianOption, dt: float = 1 / 252) -> float:
    from dataclasses import replace
    if opt.T <= dt:
        return 0.0
    dn = geometric_asian(replace(opt, T=opt.T - dt))
    return (dn - geometric_asian(opt)) / dt


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20
    n_obs = 252

    opt_call = AsianOption(S=S, K=K, T=T, r=r, q=q, sigma=sigma,
                            n_obs=n_obs, is_call=True)
    opt_put = AsianOption(S=S, K=K, T=T, r=r, q=q, sigma=sigma,
                           n_obs=n_obs, is_call=False)

    geo_call = geometric_asian_call(opt_call)
    geo_put = geometric_asian_put(opt_put)
    print(f"Geometric Asian (closed-form):")
    print(f"  Call: {geo_call:.4f}")
    print(f"  Put:  {geo_put:.4f}")
    print(f"  Put-call parity check: {geo_call - geo_put:.4f} (should â S - Ke^{{-rT}} adjusted)")

    print(f"\nArithmetic Asian (MC with geometric control variate, 100k paths):")
    arith_call, se_c = arithmetic_asian_mc(opt_call, n_paths=100_000)
    arith_put, se_p = arithmetic_asian_mc(opt_put, n_paths=100_000)
    print(f"  Call: {arith_call:.4f} Â± {1.96 * se_c:.4f} (95% CI)")
    print(f"  Put:  {arith_put:.4f} Â± {1.96 * se_p:.4f} (95% CI)")

    print(f"\nComparison across moneyness (T=1, geometric closed-form vs arithmetic MC):")
    print(f"{'K':>6} {'Geo Call':>10} {'Arith Call':>12} {'Difference':>12}")
    print("-" * 45)
    for K_val in [85, 90, 95, 100, 105, 110, 115]:
        o = AsianOption(S=S, K=K_val, T=T, r=r, q=q, sigma=sigma,
                        n_obs=n_obs, is_call=True)
        g = geometric_asian_call(o)
        a, _ = arithmetic_asian_mc(o, n_paths=50_000)
        print(f"{K_val:>6} {g:>10.4f} {a:>12.4f} {a - g:>+12.4f}")

    print(f"\nGreeks (geometric closed-form, K=100):")
    delta = asian_delta(opt_call)
    gamma = asian_gamma(opt_call)
    vega = asian_vega(opt_call)
    theta = asian_theta(opt_call)
    print(f"  Delta: {delta:.4f}")
    print(f"  Gamma: {gamma:.6f}")
    print(f"  Vega:  {vega:.4f}")
    print(f"  Theta: {theta:.4f}")
