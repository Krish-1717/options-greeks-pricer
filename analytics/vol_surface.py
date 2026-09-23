"""
analytics/vol_surface.py -- Implied volatility surface for options-greeks-pricer
Day 12: SVI parameterisation, arbitrage-free interpolation, surface metrics
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Normal/inverse-normal helpers (no scipy)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bisect(f, lo: float, hi: float, tol: float = 1e-8, max_iter: int = 60) -> float:
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Black-Scholes implied vol inversion
# ---------------------------------------------------------------------------

def _bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    option_type: str = "call",
    tol: float = 1e-6,
) -> Optional[float]:
    """
    Invert Black-Scholes to find implied volatility via bisection.
    Returns None if the price is below intrinsic value or convergence fails.
    """
    if option_type == "put":
        # put-call parity
        fwd = S * math.exp(r * T)
        market_price = market_price + fwd - K * math.exp(-r * T)

    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    if market_price <= intrinsic:
        return None

    def objective(sigma: float) -> float:
        return _bs_call_price(S, K, T, r, sigma) - market_price

    try:
        return _bisect(objective, 1e-6, 10.0, tol=tol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Volatility smile / surface data structures
# ---------------------------------------------------------------------------

@dataclass
class VolSmile:
    """Implied vols across strikes for a single expiry."""
    expiry: float          # time to expiry in years
    strikes: List[float]   # absolute strikes
    ivols: List[float]     # implied vols (decimal)
    forward: float         # forward price for this expiry

    def __post_init__(self) -> None:
        if len(self.strikes) != len(self.ivols):
            raise ValueError("strikes and ivols must have equal length")

    @property
    def log_moneyness(self) -> List[float]:
        """log(K/F) for each strike."""
        return [math.log(k / self.forward) for k in self.strikes]

    def vol_at(self, strike: float) -> float:
        """Linear interpolation between grid strikes; flat extrapolation."""
        xs, ys = self.strikes, self.ivols
        if strike <= xs[0]:
            return ys[0]
        if strike >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= strike <= xs[i + 1]:
                t = (strike - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + t * (ys[i + 1] - ys[i])
        return ys[-1]

    @property
    def atm_vol(self) -> float:
        return self.vol_at(self.forward)

    @property
    def skew(self) -> float:
        """25-delta put skew (25P IV - ATM IV) in vol points."""
        k25p = self.forward * math.exp(-0.5 * self.atm_vol ** 2 * self.expiry
                                       - 0.674 * self.atm_vol * math.sqrt(self.expiry))
        return self.vol_at(k25p) - self.atm_vol

    @property
    def kurtosis_proxy(self) -> float:
        """Wing spread: avg(25P + 25C) - ATM, a proxy for realized kurtosis."""
        k25c = self.forward * math.exp(-0.5 * self.atm_vol ** 2 * self.expiry
                                       + 0.674 * self.atm_vol * math.sqrt(self.expiry))
        k25p = self.forward * math.exp(-0.5 * self.atm_vol ** 2 * self.expiry
                                       - 0.674 * self.atm_vol * math.sqrt(self.expiry))
        return 0.5 * (self.vol_at(k25p) + self.vol_at(k25c)) - self.atm_vol


# ---------------------------------------------------------------------------
# SVI (Stochastic Volatility Inspired) parameterisation
# ---------------------------------------------------------------------------

@dataclass
class SVIParams:
    """
    Raw SVI: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    where k = log(K/F) and w = sigma_implied^2 * T.
    """
    a: float      # vertical translation
    b: float      # slope; >= 0
    rho: float    # correlation; in (-1, 1)
    m: float      # horizontal translation
    sigma: float  # curvature; > 0

    def total_variance(self, log_moneyness: float) -> float:
        k = log_moneyness - self.m
        return self.a + self.b * (self.rho * k + math.sqrt(k ** 2 + self.sigma ** 2))

    def implied_vol(self, log_moneyness: float, expiry: float) -> float:
        w = self.total_variance(log_moneyness)
        return math.sqrt(max(w, 0.0) / expiry)

    def is_arbitrage_free(self) -> bool:
        """Necessary (not sufficient) conditions for absence of butterfly arb."""
        if self.b < 0 or self.sigma <= 0:
            return False
        if abs(self.rho) >= 1:
            return False
        # Durrleman condition: a + b*sigma*sqrt(1-rho^2) >= 0
        if self.a + self.b * self.sigma * math.sqrt(1.0 - self.rho ** 2) < 0:
            return False
        return True


def fit_svi(smile: VolSmile, initial: Optional[SVIParams] = None) -> SVIParams:
    """
    Minimise MSE between SVI total variance and market total variance.
    Uses a simple coordinate descent with fine grid search per parameter.
    """
    T = smile.expiry
    ks = smile.log_moneyness
    ws_market = [iv ** 2 * T for iv in smile.ivols]

    if initial is None:
        atm_w = smile.atm_vol ** 2 * T
        p = SVIParams(a=atm_w * 0.5, b=0.5, rho=-0.3, m=0.0, sigma=0.2)
    else:
        p = SVIParams(initial.a, initial.b, initial.rho, initial.m, initial.sigma)

    def mse(params: SVIParams) -> float:
        err = 0.0
        for k, w_mkt in zip(ks, ws_market):
            w_model = params.total_variance(k)
            err += (w_model - w_mkt) ** 2
        return err / len(ks)

    # Coordinate descent: 3 passes over all params
    param_ranges = {
        "a": (0.0, 0.5, 20),
        "b": (0.01, 2.0, 20),
        "rho": (-0.95, 0.95, 20),
        "m": (-0.5, 0.5, 20),
        "sigma": (0.01, 1.0, 20),
    }
    for _ in range(3):
        for name, (lo, hi, n) in param_ranges.items():
            best_val = getattr(p, name)
            best_err = mse(p)
            for i in range(n + 1):
                val = lo + (hi - lo) * i / n
                test = SVIParams(
                    a=val if name == "a" else p.a,
                    b=val if name == "b" else p.b,
                    rho=val if name == "rho" else p.rho,
                    m=val if name == "m" else p.m,
                    sigma=val if name == "sigma" else p.sigma,
                )
                err = mse(test)
                if err < best_err:
                    best_err = err
                    best_val = val
            setattr(p, name, best_val)
    return p


# ---------------------------------------------------------------------------
# Volatility surface
# ---------------------------------------------------------------------------

@dataclass
class VolSurface:
    """Collection of smiles across expiries."""
    smiles: List[VolSmile]
    svi_params: Dict[float, SVIParams] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.smiles.sort(key=lambda s: s.expiry)

    def fit_svi(self) -> None:
        """Fit SVI to each smile independently."""
        prev = None
        for smile in self.smiles:
            p = fit_svi(smile, initial=prev)
            self.svi_params[smile.expiry] = p
            prev = p

    def vol_at(self, strike: float, expiry: float) -> float:
        """Bilinear interpolation across strikes and expiries."""
        expiries = [s.expiry for s in self.smiles]
        if expiry <= expiries[0]:
            return self.smiles[0].vol_at(strike)
        if expiry >= expiries[-1]:
            return self.smiles[-1].vol_at(strike)
        for i in range(len(expiries) - 1):
            if expiries[i] <= expiry <= expiries[i + 1]:
                t = (expiry - expiries[i]) / (expiries[i + 1] - expiries[i])
                v0 = self.smiles[i].vol_at(strike)
                v1 = self.smiles[i + 1].vol_at(strike)
                # Interpolate total variance, not vol, for time-consistency
                w0 = v0 ** 2 * expiries[i]
                w1 = v1 ** 2 * expiries[i + 1]
                w = w0 + t * (w1 - w0)
                return math.sqrt(max(w, 0.0) / expiry)
        return self.smiles[-1].vol_at(strike)

    def term_structure(self) -> List[Tuple[float, float]]:
        """ATM vol for each expiry: [(expiry, atm_vol), ...]"""
        return [(s.expiry, s.atm_vol) for s in self.smiles]

    def print_surface(self) -> None:
        print("=" * 65)
        print("  VOLATILITY SURFACE")
        print("=" * 65)
        print(f"  {'Expiry (Y)':>12}  {'[ATM Vol':>9}  {'25P Skew':>10}  {'Kurt proxy':>12}  {'ArbFree':>8}")
        print("  " + "-" * 57)
        for smile in self.smiles:
            svi = self.svi_params.get(smile.expiry)
            arb_free = svi.is_arbitrage_free() if svi else "N/A"
            print(
                f"  {smile.expiry:>12.4f}  "
                f"{smile.atm_vol*100:>8.2f}%  "
                f"{smile.skew*100:>+9.2f}%  "
                f"{mile.kurtosis_proxy*100:>+11.2f}%  "
                f"{'[Yes' if arb_free else 'No':>8}"
            )
        print("=" * 65)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S = 100.0
    r = 0.05

    smiles = [
        VolSmile(
            expiry=0.25,
            strikes=[85, 90, 95, 100, 105, 110, 115],
            ivols=  [0.32, 0.28, 0.24, 0.20, 0.21, 0.23, 0.26],
            forward=S * math.exp(r * 0.25),
        ),
        VolSmile(
            expiry=0.50,
            strikes=[80, 88, 94, 100, 106, 112, 120],
            ivols=  [0.30, 0.26, 0.23, 0.20, 0.21, 0.22, 0.25],
            forward=S * math.exp(r * 0.50),
        ),
        VolSmile(
            expiry=1.00,
            strikes=[75, 85, 92, 100, 108, 115, 125],
            ivols=  [0.28, 0.25, 0.22, 0.20, 0.205, 0.215, 0.24],
            forward=S * math.exp(r * 1.00),
        ),
        VolSmile(
            expiry=2.00,
            strikes=[65, 80, 90, 100, 110, 120, 135],
            ivols=  [0.27, 0.24, 0.22, 0.20, 0.205, 0.21, 0.23],
            forward=S * math.exp(r * 2.00),
        ),
    ]

    surf = VolSurface(smiles=smiles)
    surf.fit_svi()
    surf.print_surface()

    print()
    print("Spot vol interpolation: strike=97, expiry=0.75")
    print(f"  => {surf.vol_at(97.0, 0.75)*100:.2f}%")
