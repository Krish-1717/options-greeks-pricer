"""
analytics/scenario_matrix.py -- Greeks scenario matrix for options-greeks-pricer
Day 11: computes delta/gamma/vega/theta/rho across spot x vol x expiry grids.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── helpers ──────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Abramowitz & Stegun approximation (error < 7.5e-8)."""
    if x < 0:
        return 1.0 - _norm_cdf(-x)
    k = 1.0 / (1.0 + 0.2316419 * x)
    poly = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))))
    return 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return float('inf') if S > K else float('-inf')
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

def _d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    return _d1(S, K, T, r, sigma, q) - sigma * math.sqrt(T)


# ── Greeks ────────────────────────────────────────────────────────────────────

@dataclass
class GreeksResult:
    price:  float
    delta:  float
    gamma:  float
    vega:   float   # per 1-vol-point (not per 0.01)
    theta:  float   # per calendar day
    rho:    float   # per 1% rate move
    option_type: str

def bsm_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> GreeksResult:
    """Black-Scholes-Merton Greeks for a European option."""
    ot = option_type.lower()
    if T <= 0:
        # Expiry: analytical limits
        intrinsic = max(S - K, 0) if ot == "call" else max(K - S, 0)
        return GreeksResult(price=intrinsic, delta=float(S > K) if ot == "call" else float(S < K),
                            gamma=0.0, vega=0.0, theta=0.0, rho=0.0, option_type=ot)

    d1v = _d1(S, K, T, r, sigma, q)
    d2v = _d2(S, K, T, r, sigma, q)
    sqrtT = math.sqrt(T)
    disc = math.exp(-r * T)
    q_disc = math.exp(-q * T)

    if ot == "call":
        price = S * q_disc * _norm_cdf(d1v) - K * disc * _norm_cdf(d2v)
        delta = q_disc * _norm_cdf(d1v)
        rho   = K * T * disc * _norm_cdf(d2v) / 100.0
        theta = (-S * q_disc * _norm_pdf(d1v) * sigma / (2 * sqrtT)
                 - r * K * disc * _norm_cdf(d2v)
                 + q * S * q_disc * _norm_cdf(d1v)) / 365.0
    else:
        price = K * disc * _norm_cdf(-d2v) - S * q_disc * _norm_cdf(-d1v)
        delta = q_disc * (_norm_cdf(d1v) - 1.0)
        rho   = -K * T * disc * _norm_cdf(-d2v) / 100.0
        theta = (-S * q_disc * _norm_pdf(d1v) * sigma / (2 * sqrtT)
                 + r * K * disc * _norm_cdf(-d2v)
                 - q * S * q_disc * _norm_cdf(-d1v)) / 365.0

    gamma = q_disc * _norm_pdf(d1v) / (S * sigma * sqrtT)
    vega  = S * q_disc * _norm_pdf(d1v) * sqrtT / 100.0

    return GreeksResult(price=price, delta=delta, gamma=gamma,
                        vega=vega, theta=theta, rho=rho, option_type=ot)


# ── Scenario Matrix ───────────────────────────────────────────────────────────

@dataclass
class ScenarioCell:
    spot:  float
    vol:   float
    expiry_days: int
    greeks: GreeksResult

@dataclass
class ScenarioMatrix:
    """Grid of Greeks across spot, vol, and expiry dimensions."""
    K: float
    r: float
    option_type: str
    q: float
    cells: List[ScenarioCell] = field(default_factory=list)

    # ── slice views ──────────────────────────────────────────────────────────
    def by_spot(self, spot: float) -> List[ScenarioCell]:
        return [c for c in self.cells if abs(c.spot - spot) < 1e-9]

    def by_vol(self, vol: float) -> List[ScenarioCell]:
        return [c for c in self.cells if abs(c.vol - vol) < 1e-9]

    def by_expiry(self, days: int) -> List[ScenarioCell]:
        return [c for c in self.cells if c.expiry_days == days]

    # ── extreme finders ──────────────────────────────────────────────────────
    def max_gamma_cell(self) -> Optional[ScenarioCell]:
        return max(self.cells, key=lambda c: c.greeks.gamma) if self.cells else None

    def max_vega_cell(self) -> Optional[ScenarioCell]:
        return max(self.cells, key=lambda c: c.greeks.vega) if self.cells else None

    def min_theta_cell(self) -> Optional[ScenarioCell]:
        return min(self.cells, key=lambda c: c.greeks.theta) if self.cells else None

    # ── summary ──────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        if not self.cells:
            return {}
        deltas  = [c.greeks.delta  for c in self.cells]
        gammas  = [c.greeks.gamma  for c in self.cells]
        prices  = [c.greeks.price  for c in self.cells]
        return {
            "n_cells": len(self.cells),
            "delta_range": (min(deltas), max(deltas)),
            "gamma_range": (min(gammas), max(gammas)),
            "price_range": (min(prices), max(prices)),
            "max_gamma": {"spot": self.max_gamma_cell().spot,
                          "vol": self.max_gamma_cell().vol,
                          "expiry_days": self.max_gamma_cell().expiry_days,
                          "gamma": self.max_gamma_cell().greeks.gamma},
        }

    # ── table printer ────────────────────────────────────────────────────────
    def print_delta_table(self, expiry_days: int, greek: str = "delta") -> None:
        """Print a spot x vol heatmap for a single expiry."""
        slice_ = self.by_expiry(expiry_days)
        if not slice_:
            print(f"No cells for expiry_days={expiry_days}")
            return
        spots = sorted(set(c.spot for c in slice_))
        vols  = sorted(set(c.vol  for c in slice_))
        lookup: dict[Tuple[float, float], float] = {}
        for c in slice_:
            val = getattr(c.greeks, greek)
            lookup[(c.spot, c.vol)] = val

        col_w = 9
        header = f"{'spot/vol':>{col_w}}" + "".join(f"{v*100:>{col_w}.1f}%" for v in vols)
        print(f"\n{greek.upper()} matrix | K={self.K} expiry={expiry_days}d otype={self.option_type}")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for S in spots:
            row = f"{S:>{col_w}.2f}"
            for v in vols:
                val = lookup.get((S, v), float('nan'))
                row += f"{val:>{col_w}.4f}"
            print(row)
        print("-" * len(header))


def build_scenario_matrix(
    K: float,
    r: float,
    spots: List[float],
    vols: List[float],
    expiry_days_list: List[int],
    option_type: str = "call",
    q: float = 0.0,
) -> ScenarioMatrix:
    """Compute Greeks for every combination of spot, vol, and expiry."""
    matrix = ScenarioMatrix(K=K, r=r, option_type=option_type, q=q)
    for days in expiry_days_list:
        T = days / 365.0
        for S in spots:
            for v in vols:
                g = bsm_greeks(S, K, T, r, v, option_type, q)
                matrix.cells.append(ScenarioCell(spot=S, vol=v, expiry_days=days, greeks=g))
    return matrix


# ── CLI demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    K = 100.0
    r = 0.05
    spots = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    vols  = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    expiries = [7, 30, 60, 90, 180]

    for otype in ("call", "put"):
        mat = build_scenario_matrix(K, r, spots, vols, expiries, option_type=otype)
        print(f"\n{'='*60}")
        print(f"Option type : {otype.upper()}")
        print(f"Strike      : {K}  r={r*100:.1f}%  spots={spots[0]}-{spots[-1]}")
        print(json.dumps(mat.summary(), indent=2, default=str))
        mat.print_delta_table(expiry_days=30, greek="delta")
        mat.print_delta_table(expiry_days=30, greek="gamma")
