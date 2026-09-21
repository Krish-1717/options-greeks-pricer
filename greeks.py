"""
src/greeks.py -- Black-Scholes option pricing with full Greeks
options-greeks-pricer Day 1 Commit 1
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal

OptionType = Literal["call", "put"]

_SQRT2PI = math.sqrt(2 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


@dataclass
class BSInputs:
    S: float   # spot price
    K: float   # strike
    T: float   # time to expiry (years)
    r: float   # risk-free rate (continuous)
    sigma: float  # implied volatility
    q: float = 0.0  # continuous dividend yield

    def d1(self) -> float:
        return (math.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * math.sqrt(self.T))

    def d2(self) -> float:
        return self.d1() - self.sigma * math.sqrt(self.T)


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float   # per calendar day
    vega: float    # per 1% move in vol
    rho: float     # per 1% move in rate
    option_type: OptionType


def price_option(inputs: BSInputs, option_type: OptionType) -> Greeks:
    """Full Black-Scholes pricing with all first-order Greeks."""
    S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return Greeks(price=intrinsic, delta=float(S > K if option_type == "call" else S < K),
                      gamma=0.0, theta=0.0, vega=0.0, rho=0.0, option_type=option_type)

    d1 = inputs.d1()
    d2 = inputs.d2()
    sqrtT = math.sqrt(T)
    e_qt = math.exp(-q * T)
    e_rt = math.exp(-r * T)
    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    nd1_pdf = _norm_pdf(d1)

    if option_type == "call":
        price = S * e_qt * nd1 - K * e_rt * nd2
        delta = e_qt * nd1
        theta = (-(S * e_qt * nd1_pdf * sigma) / (2 * sqrtT)
                 - r * K * e_rt * nd2
                 + q * S * e_qt * nd1) / 365
        rho = K * T * e_rt * nd2 / 100
    else:
        nd1_neg = _norm_cdf(-d1)
        nd2_neg = _norm_cdf(-d2)
        price = K * e_rt * nd2_neg - S * e_qt * nd1_neg
        delta = -e_qt * nd1_neg
        theta = (-(S * e_qt * nd1_pdf * sigma) / (2 * sqrtT)
                 + r * K * e_rt * nd2_neg
                 - q * S * e_qt * nd1_neg) / 365
        rho = -K * T * e_rt * nd2_neg / 100

    gamma = e_qt * nd1_pdf / (S * sigma * sqrtT)
    vega = S * e_qt * nd1_pdf * sqrtT / 100

    return Greeks(price=price, delta=delta, gamma=gamma,
                  theta=theta, vega=vega, rho=rho, option_type=option_type)


def implied_vol(market_price: float, inputs: BSInputs, option_type: OptionType,
                tol: float = 1e-6, max_iter: int = 100) -> float:
    """Newton-Raphson implied volatility solver."""
    sigma = 0.2
    for _ in range(max_iter):
        inp = BSInputs(inputs.S, inputs.K, inputs.T, inputs.r, sigma, inputs.q)
        g = price_option(inp, option_type)
        diff = g.price - market_price
        if abs(diff) < tol:
            return sigma
        vega_raw = g.vega * 100  # undo the /100 scaling
        if abs(vega_raw) < 1e-10:
            break
        sigma -= diff / vega_raw
        sigma = max(1e-4, min(sigma, 5.0))
    return sigma
