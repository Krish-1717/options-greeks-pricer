"""
Delta Hedging Backtest â P&L Simulation
Day 16 â options-greeks-pricer/hedging/delta_hedge_backtest.py

Simulates dynamic delta hedging of a short options position:
  - Discrete rebalancing at configurable frequency
  - Transaction costs on each rebalance
  - P&L decomposition (theta, gamma, vega P&L)
  - Comparison across hedge frequencies
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return (S * math.exp(-q * T) * _norm_cdf(d1)
            - K * math.exp(-r * T) * _norm_cdf(d2))


def bs_put(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return (K * math.exp(-r * T) * _norm_cdf(-d2)
            - S * math.exp(-q * T) * _norm_cdf(-d1))


def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float,
             is_call: bool = True) -> float:
    if T <= 0:
        return 1.0 if (is_call and S > K) else 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    if is_call:
        return math.exp(-q * T) * _norm_cdf(d1)
    return -math.exp(-q * T) * _norm_cdf(-d1)


def bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return math.exp(-q * T) * _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_theta(S: float, K: float, T: float, r: float, q: float, sigma: float,
             is_call: bool = True) -> float:
    """Theta per day."""
    if T <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    term1 = -S * math.exp(-q * T) * _norm_pdf(d1) * sigma / (2 * math.sqrt(T))
    if is_call:
        theta = (term1 + q * S * math.exp(-q * T) * _norm_cdf(d1)
                 - r * K * math.exp(-r * T) * _norm_cdf(d2))
    else:
        theta = (term1 - q * S * math.exp(-q * T) * _norm_cdf(-d1)
                 + r * K * math.exp(-r * T) * _norm_cdf(-d2))
    return theta / 365  # per calendar day


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Vega per 1% move in vol."""
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T) * 0.01


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class OptionContract:
    K: float
    T_init: float     # initial time to expiry
    r: float
    q: float
    sigma: float      # implied vol at inception
    is_call: bool = True
    position: int = -1         # -1 = short, +1 = long
    n_contracts: float = 1.0   # contract multiplier


@dataclass
class HedgeStep:
    t: float
    S: float
    T_remaining: float
    option_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    hedge_shares: float       # current hedge position
    hedge_cost: float         # cost of rebalancing at this step
    pnl_delta: float          # delta P&L since last step
    pnl_gamma: float          # gamma P&L since last step
    pnl_theta: float          # theta P&L since last step


@dataclass
class HedgeBacktestResult:
    total_pnl: float
    pnl_series: List[float]
    hedge_steps: List[HedgeStep]
    final_option_pnl: float    # option payoff - initial premium
    hedge_pnl: float           # from hedge trades
    transaction_costs: float
    n_rebalances: int
    realised_vol: float


def simulate_gbm(S0: float, r: float, q: float, sigma: float, T: float,
                 n_steps: int, seed: int = 42) -> List[float]:
    """Simulate GBM price path. Returns price at each step including t=0."""
    rng = random.Random(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol_dt = sigma * math.sqrt(dt)
    prices = [S0]
    for _ in range(n_steps):
        z = rng.gauss(0, 1)
        prices.append(prices[-1] * math.exp(drift + vol_dt * z))
    return prices


def delta_hedge_backtest(
        option: OptionContract,
        price_path: List[float],
        hedge_freq: int = 1,         # rebalance every N steps
        transaction_cost: float = 0.001,  # as fraction of trade value
        sigma_hedge: Optional[float] = None,  # vol used for hedging (default = option.sigma)
) -> HedgeBacktestResult:
    """
    Simulate delta hedging of a single option over a given price path.

    price_path: S at each time step (length = n_steps + 1, first is S0)
    """
    sigma_h = sigma_hedge or option.sigma
    n_steps = len(price_path) - 1
    dt = option.T_init / n_steps

    # Initial setup
    S0 = price_path[0]
    T0 = option.T_init
    init_price = (bs_call(S0, option.K, T0, option.r, option.q, option.sigma)
                  if option.is_call else
                  bs_put(S0, option.K, T0, option.r, option.q, option.sigma))

    # Short the option: receive premium
    cash = init_price * option.n_contracts * abs(option.position)
    shares = 0.0   # current hedge shares
    total_tc = 0.0
    pnl_series: List[float] = []
    hedge_steps: List[HedgeStep] = []
    n_rebal = 0

    for step in range(n_steps):
        S = price_path[step]
        T_rem = max(option.T_init - step * dt, 1e-8)

        # Compute Greeks
        delta = bs_delta(S, option.K, T_rem, option.r, option.q, sigma_h,
                          option.is_call)
        gamma = bs_gamma(S, option.K, T_rem, option.r, option.q, sigma_h)
        theta = bs_theta(S, option.K, T_rem, option.r, option.q, sigma_h,
                          option.is_call)
        vega = bs_vega(S, option.K, T_rem, option.r, option.q, sigma_h)
        opt_price = (bs_call(S, option.K, T_rem, option.r, option.q, option.sigma)
                     if option.is_call else
                     bs_put(S, option.K, T_rem, option.r, option.q, option.sigma))

        # Rebalance on schedule
        hedge_cost = 0.0
        if step % hedge_freq == 0:
            target_shares = delta * option.position * option.n_contracts * -1
            trade = target_shares - shares
            hedge_cost = abs(trade) * S * transaction_cost
            cash -= trade * S + hedge_cost
            shares = target_shares
            total_tc += hedge_cost
            n_rebal += 1

        # Carry cash at risk-free rate
        cash *= math.exp(option.r * dt)

        # P&L attribution (approximate, one step)
        S_next = price_path[step + 1]
        dS = S_next - S
        pnl_delta = shares * dS
        pnl_gamma = 0.5 * gamma * dS ** 2 * option.position * -1 * option.n_contracts
        pnl_theta = theta * dt * 365 * option.position * -1 * option.n_contracts

        step_pnl = pnl_delta + pnl_gamma + pnl_theta - hedge_cost
        pnl_series.append(step_pnl)

        hedge_steps.append(HedgeStep(
            t=step * dt, S=S, T_remaining=T_rem,
            option_price=opt_price, delta=delta,
            gamma=gamma, theta=theta, vega=vega,
            hedge_shares=shares, hedge_cost=hedge_cost,
            pnl_delta=pnl_delta, pnl_gamma=pnl_gamma, pnl_theta=pnl_theta,
        ))

    # Expiry settlement
    S_T = price_path[-1]
    payoff = (max(S_T - option.K, 0.0) if option.is_call
              else max(option.K - S_T, 0.0))
    option_pnl = (init_price - payoff) * option.n_contracts  # short position
    hedge_value = shares * S_T + cash
    total_pnl = option_pnl + hedge_value

    # Realised vol from price path
    log_rets = [math.log(price_path[i + 1] / price_path[i])
                for i in range(n_steps)]
    rv_mu = sum(log_rets) / n_steps
    rv_var = sum((r - rv_mu) ** 2 for r in log_rets) / n_steps
    rv = math.sqrt(rv_var * 252)

    return HedgeBacktestResult(
        total_pnl=total_pnl,
        pnl_series=pnl_series,
        hedge_steps=hedge_steps,
        final_option_pnl=option_pnl,
        hedge_pnl=sum(pnl_series),
        transaction_costs=total_tc,
        n_rebalances=n_rebal,
        realised_vol=rv,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random as _rng
    S0, K, T = 100.0, 100.0, 0.25
    r, q, sigma = 0.05, 0.0, 0.20
    n_steps = 63   # one trading day per step

    option = OptionContract(K=K, T_init=T, r=r, q=q, sigma=sigma,
                             is_call=True, position=-1, n_contracts=100)

    # Simulate price path
    path = simulate_gbm(S0, r, q, sigma, T, n_steps, seed=42)

    print(f"S0={S0}, K={K}, T={T:.2f}, Ï={sigma:.0%}")
    print(f"Initial call price: ${bs_call(S0, K, T, r, q, sigma):.4f}")
    print(f"Final spot: ${path[-1]:.2f}")
    print()

    # Compare hedge frequencies
    for freq in [1, 5, 21]:
        res = delta_hedge_backtest(option, path, hedge_freq=freq,
                                    transaction_cost=0.001)
        print(f"Hedge freq = {freq:2d}d: "
              f"Total P&L = ${res.total_pnl:+8.2f}  "
              f"TC = ${res.transaction_costs:.2f}  "
              f"Rebalances = {res.n_rebalances:3d}")

    # Detailed report for daily hedging
    print("\nDetailed daily hedge stats:")
    res = delta_hedge_backtest(option, path, hedge_freq=1)
    print(f"  Realised vol:  {res.realised_vol:.2%}")
    print(f"  Option P&L:    ${res.final_option_pnl:.2f}")
    print(f"  Hedge P&L:     ${res.hedge_pnl:.2f}")
    print(f"  Trans. costs:  ${res.transaction_costs:.2f}")
    print(f"  Net P&L:       ${res.total_pnl:.2f}")

    # Show last 5 steps
    print(f"\n  Last 5 hedge steps:")
    print(f"  {'t':>6} {'S':>7} {'delta':>7} {'gamma':>7} {'pnl_gamma':>10}")
    for hs in res.hedge_steps[-5:]:
        print(f"  {hs.t:>6.3f} {hs.S:>7.2f} {hs.delta:>7.4f} "
              f"{hs.gamma:>7.4f} {hs.pnl_gamma:>10.4f}")
