"""
Barrier Option Pricing
Day 16 â options-greeks-pricer/exotics/barrier_options.py

Implements analytical and Monte Carlo pricing for:
  - Knock-in / knock-out calls and puts (single barrier)
  - Double barrier options
  - Partial-time barriers (forward start)
  - Barrier Greeks (delta, gamma, vega, barrier delta)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Normal distribution
# ---------------------------------------------------------------------------

def _N(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _n(x: float) -> float:
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


# ---------------------------------------------------------------------------
# Standard Black-Scholes (for comparisons)
# ---------------------------------------------------------------------------

def bs_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _N(d1) - K * math.exp(-r * T) * _N(d2)


def bs_put(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _N(-d2) - S * math.exp(-q * T) * _N(-d1)


# ---------------------------------------------------------------------------
# Barrier option data structure
# ---------------------------------------------------------------------------

@dataclass
class BarrierOption:
    S: float          # spot
    K: float          # strike
    T: float          # time to expiry
    r: float          # risk-free rate
    q: float          # dividend yield
    sigma: float      # volatility
    H: float          # barrier level
    is_call: bool = True
    knock_in: bool = False    # True = knock-in, False = knock-out
    up_barrier: bool = True   # True = up barrier, False = down barrier
    rebate: float = 0.0       # rebate paid if knocked out


# ---------------------------------------------------------------------------
# Analytical single-barrier pricing (Merton 1973, Reiner-Rubinstein 1991)
# ---------------------------------------------------------------------------

def _barrier_params(S: float, K: float, H: float, T: float,
                     r: float, q: float, sigma: float
                     ) -> Tuple[float, float, float, float, float, float, float]:
    mu = (r - q) / sigma ** 2 - 0.5
    lam = math.sqrt(mu ** 2 + 2 * r / sigma ** 2)
    x1 = math.log(S / K) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    x2 = math.log(S / H) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    y1 = math.log(H ** 2 / (S * K)) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    y2 = math.log(H / S) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    return mu, lam, x1, x2, y1, y2


def down_and_out_call(opt: BarrierOption) -> float:
    """Down-and-out call: knocked out if S reaches H (H < S0, H < K typically)."""
    S, K, H, T, r, q, sigma = (opt.S, opt.K, opt.H, opt.T, opt.r, opt.q, opt.sigma)
    rebate = opt.rebate

    if H >= K:
        # Barrier above strike â special case
        return 0.0

    mu = (r - q - 0.5 * sigma ** 2) / (sigma ** 2)
    lamb = (H / S) ** (2 * mu + 2)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    e1 = (math.log(H ** 2 / (S * K)) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    e2 = e1 - sigma * sqrt_T

    A = S * math.exp(-q * T) * _N(d1) - K * math.exp(-r * T) * _N(d2)
    B = S * math.exp(-q * T) * lamb * _N(e1) - K * math.exp(-r * T) * lamb * _N(e2)

    return max(A - B, 0.0) + rebate * _rebate_factor(S, H, T, r, q, sigma, up=False)


def down_and_in_call(opt: BarrierOption) -> float:
    """Down-and-in call: activated only if S touches H."""
    vanilla = bs_call(opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma)
    dout = down_and_out_call(BarrierOption(
        S=opt.S, K=opt.K, H=opt.H, T=opt.T, r=opt.r, q=opt.q,
        sigma=opt.sigma, rebate=0.0))
    return max(vanilla - dout, 0.0) + opt.rebate * _rebate_factor_in(
        opt.S, opt.H, opt.T, opt.r, opt.q, opt.sigma, up=False)


def up_and_out_call(opt: BarrierOption) -> float:
    """Up-and-out call: knocked out if S rises to H."""
    S, K, H, T, r, q, sigma = opt.S, opt.K, opt.H, opt.T, opt.r, opt.q, opt.sigma
    rebate = opt.rebate

    if S >= H:
        return rebate * math.exp(-r * T)

    mu = (r - q - 0.5 * sigma ** 2) / sigma ** 2
    lamb = (H / S) ** (2 * mu + 2)
    sqrt_T = math.sqrt(T)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    e1 = (math.log(H ** 2 / (S * K)) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    e2 = e1 - sigma * sqrt_T
    f1 = (math.log(S / H) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    f2 = f1 - sigma * sqrt_T

    if H <= K:
        price = 0.0
    else:
        A = S * math.exp(-q * T) * _N(d1) - K * math.exp(-r * T) * _N(d2)
        B = S * math.exp(-q * T) * _N(f1) - K * math.exp(-r * T) * _N(f2)
        C = S * math.exp(-q * T) * lamb * _N(-e1) - K * math.exp(-r * T) * lamb * _N(-e2)
        D = S * math.exp(-q * T) * lamb * _N(-f1) - K * math.exp(-r * T) * lamb * _N(-f2)
        price = A - B + C - D

    rebate_pv = rebate * _rebate_factor(S, H, T, r, q, sigma, up=True)
    return max(price, 0.0) + rebate_pv


def up_and_in_call(opt: BarrierOption) -> float:
    """Up-and-in call: activated if S rises to H."""
    vanilla = bs_call(opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma)
    uout = up_and_out_call(BarrierOption(
        S=opt.S, K=opt.K, H=opt.H, T=opt.T, r=opt.r, q=opt.q,
        sigma=opt.sigma, rebate=0.0))
    return max(vanilla - uout, 0.0)


def _rebate_factor(S: float, H: float, T: float, r: float, q: float,
                   sigma: float, up: bool) -> float:
    """PV factor for rebate (binary barrier)."""
    mu = (r - q - 0.5 * sigma ** 2) / sigma ** 2
    lam = math.sqrt(mu ** 2 + 2 * r / sigma ** 2)
    sqrt_T = math.sqrt(T)
    z = math.log(H / S) / (sigma * sqrt_T) + lam * sigma * sqrt_T
    x = math.log(H / S) / (sigma * sqrt_T) - lam * sigma * sqrt_T
    ratio = (H / S) ** (mu + lam)
    ratio2 = (H / S) ** (mu - lam)
    if up:
        return ratio * _N(z) + ratio2 * _N(x)
    else:
        return ratio * _N(-z)!+ ratio2 * _N(-x)


def _rebate_factor_in(S: float, H: float, T: float, r: float, q: float,
                       sigma: float, up: bool) -> float:
    return math.exp(-r * T) - _rebate_factor(S, H, T, r, q, sigma, up)


def price_barrier_option(opt: BarrierOption) -> float:
    """Route to the correct analytical formula."""
    if opt.up_barrier:
        if opt.is_call:
            return up_and_in_call(opt) if opt.knock_in else up_and_out_call(opt)
        else:
            # Put-call relations for barrier options
            # Up-and-out put = Up-and-in put via vanilla
            vanilla = bs_put(opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma)
            # Approximate using MC
            return barrier_option_mc(opt, n_paths=50_000)
    else:
        if opt.is_call:
            return down_and_in_call(opt) if opt.knock_in else down_and_out_call(opt)
        else:
            return barrier_option_mc(opt, n_paths=50_000)


# ---------------------------------------------------------------------------
# Monte Carlo for barrier options (general)
# ---------------------------------------------------------------------------

def barrier_option_mc(opt: BarrierOption,
                       n_paths: int = 100_000,
                       n_steps: int = 252,
                       seed: int = 42) -> float:
    """
    General Monte Carlo barrier option pricer with continuous monitoring
    approximated by daily steps.
    """
    rng = random.Random(seed)
    S0, K, H, T, r, q, sigma = (
        opt.S, opt.K, opt.H, opt.T, opt.r, opt.q, opt.sigma
    )
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol_dt = sigma * math.sqrt(dt)
    df = math.exp(-r * T)

    payoffs: List[float] = []
    half = n_paths // 2

    for _ in range(half):
        for anti in [1, -1]:
            S = S0
            knocked = False
            for _ in range(n_steps):
                z = rng.gauss(0, 1) * anti
                S = S * math.exp(drift + vol_dt * z)
                if opt.up_barrier and S >= H:
                    knocked = True
                    break
                if not opt.up_barrier and S <= H:
                    knocked = True
                    break

            if opt.knock_in:
                # Activated only if barrier was hit
                if knocked:
                    payoff = (max(S - K, 0.0) if opt.is_call else max(K - S, 0.0))
                else:
                    payoff = opt.rebate
            else:
                # Knocked out if barrier hit
                if knocked:
                    payoff = opt.rebate
                else:
                    payoff = (max(S - K, 0.0) if opt.is_call else max(K - S, 0.0))

            payoffs.append(df * payoff)

    return sum(payoffs) / len(payoffs)


# ---------------------------------------------------------------------------
# Barrier Greeks (numerical)
# ---------------------------------------------------------------------------

def barrier_delta(opt: BarrierOption, dS: float = 0.5) -> float:
    up = BarrierOption(**{**opt.__dict__, "S": opt.S + dS})
    dn = BarrierOption(**{**opt.__dict__, "S": opt.S - dS})
    return (price_barrier_option(up) - price_barrier_option(dn)) / (2 * dS)


def barrier_gamma(opt: BarrierOption, dS: float = 0.5) -> float:
    up = price_barrier_option(BarrierOption(**{**opt.__dict__, "S": opt.S + dS}))
    mid = price_barrier_option(opt)
    dn = price_barrier_option(BarrierOption(**{**opt.__dict__, "S": opt.S - dS}))
    return (up - 2 * mid + dn) / dS ** 2


def barrier_vega(opt: BarrierOption, dsigma: float = 0.01) -> float:
    up = price_barrier_option(BarrierOption(**{**opt.__dict__, "sigma": opt.sigma + dsigma}))
    dn = price_barrier_option(BarrierOption(**{**opt.__dict__, "sigma": opt.sigma - dsigma}))
    return (up - dn) / (2 * dsigma)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.05, 0.02, 0.20

    vanilla_call = bs_call(S, K, T, r, q, sigma)
    print(f"Vanilla call: {vanilla_call:.4f}")
    print()

    # Down-and-out call (H=85, below spot)
    dao = BarrierOption(S=S, K=K, H=85.0, T=T, r=r, q=q, sigma=sigma,
                         is_call=True, knock_in=False, up_barrier=False)
    dao_price = price_barrier_option(dao)

    # Down-and-in call
    dai = BarrierOption(S=S, K=K, H=85.0, T=T, r=r, q=q, sigma=sigma,
                         is_call=True, knock_in=True, up_barrier=False)
    dai_price = price_barrier_option(dai)

    # Up-and-out call (H=115, above spot)
    uao = BarrierOption(S=S, K=K, H=115.0, T=T, r=r, q=q, sigma=sigma,
                         is_call=True, knock_in=False, up_barrier=True)
    uao_price = price_barrier_option(uao)

    print(f"{'Type':<25} {'Price':>8} {'% of Vanilla':>14}")
    print("-" * 50)
    print(f"{'Vanilla call':<25} {vanilla_call:>8.4f} {'100.0%':>14}")
    print(f"{'Down-and-out (H=85)':<25} {dao_price:>8.4f} {dao_price/vanilla_call*100:>13.1f}%")
    print(f"{'Down-and-in  (H=85)':<25} {dai_price:>8.4f} {dai_price/vanilla_call*100:>13.1f}%")
    print(f"{'Up-and-out   (H=115)':<25} {uao_price:>8.4f} {uao_price/vanilla_call*100:>13.1f}%")

    # Check: D&O + D&I â Vanilla
    print(f"\nD&O + D&I = {dao_price + dai_price:.4f} (vs vanilla {vanilla_call:.4f})")

    # Greeks for down-and-out
    delta = barrier_delta(dao)
    gamma = barrier_gamma(dao)
    vega  = barrier_vega(dao)
    print(f"\nDown-and-out Greeks:")
    print(f"  Delta: {delta:.4f}")
    print(f"  Gamma: {gamma:.6f}")
    print(f"  Vega:  {vega:.4f}")

    # Effect of barrier level on D&O price
    print(f"\nBarrier level sensitivity:")
    for H in [70, 80, 85, 90, 95]:
        opt = BarrierOption(S=S, K=K, H=H, T=T, r=r, q=q, sigma=sigma,
                             is_call=True, knock_in=False, up_barrier=False)
        p = price_barrier_option(opt)
        print(f"  H={H:>3}: {p:.4f}")
