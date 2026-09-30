# Options Greeks Pricer

A 30-day deep-dive into options pricing, Greeks, volatility surfaces, and exotic derivatives — built entirely in pure Python (stdlib only, no numpy/pandas/scipy).

## What's Inside

This repo covers the full stack of modern options theory, from closed-form Black-Scholes all the way to stochastic volatility models, exotic payoffs, and interest rate derivatives.

### Days 1–19 — Core Foundations

| Module | Topic |
|--------|-------|
| `greeks.py` | Black-Scholes pricing, delta/gamma/vega/theta/rho |
| `analytics/` | Implied vol inversion, put-call parity, moneyness |
| `calibration/` | Heston & SABR calibration via nonlinear least squares |
| `data/` | Tick data ingestion, OHLCV processing |
| `exotics/` | Barrier, Asian, lookback, digital options |
| `hedging/` | Delta hedging simulation, P&L attribution |
| `models/` | Binomial trees, trinomial lattices |
| `pricing/` | Monte Carlo with variance reduction |
| `risk/` | VaR, CVaR, scenario Greeks |
| `volatility/` | Term structure, SABR smile |

### Days 20–30 — Advanced Topics (`quant_code/`)

| File | Topic |
|------|-------|
| `options_day20_smile_interpolation.py` | Smile interpolation: SVI, SABR, cubic spline |
| `options_day21_multi_asset.py` | Multi-asset options: spread, basket, quanto |
| `options_day22_scenario_analysis.py` | Greeks P&L scenario grids, ladder analysis |
| `options_day23_jump_diffusion.py` | Merton jump-diffusion, Kou double-exponential |
| `options_day24_tail_risk_hedging.py` | Tail risk hedging with OTM puts, skew trading |
| `options_day25_local_volatility.py` | CEV model, Dupire local vol via finite differences |
| `options_day26_stochastic_vol.py` | Heston CF + Gil-Pelaez, SABR Hagan 2002, full-truncation Euler MC |
| `options_day27_exotic_options.py` | Barrier (Reiner-Rubinstein), Asian (geo + arithmetic MC), lookback |
| `options_day28_vol_surface_svi.py` | SVI calibration, butterfly-free check via Dupire g(k) |
| `options_day29_interest_rate_derivs.py` | Yield curve bootstrap, Black's caplets/floorlets, swaptions |
| `options_day30_greeks_book.py` | Vanna/volga/charm/speed/zomma, P&L Taylor attribution, delta hedge sim |

## Key Concepts

- **Black-Scholes** — closed-form pricing and full Greeks suite
- **Implied volatility** — Newton-Raphson inversion, smile construction
- **Local volatility** — Dupire equation, CEV model, finite-difference surface
- **Stochastic volatility** — Heston (characteristic function + FFT), SABR (Hagan closed-form)
- **Jump-diffusion** — Merton (infinite series), Kou (double-exponential)
- **Exotic options** — Barrier (6 types, Brownian bridge MC), Asian (geo closed-form + control variate), Lookback, Forward-starting
- **Volatility surface** — SVI parameterisation, arbitrage-free checks
- **Interest rate derivatives** — LIBOR bootstrapping, cap/floor, swaption Black model
- **Greeks P&L** — Second-order Taylor attribution, vanna/volga/charm

## Running the Code

```bash
# Run any day file directly
python quant_code/options_day20_smile_interpolation.py

# Run all Day 20-30 demos
python run_demos.py
```

## Requirements

Pure Python 3.8+ standard library only — no external packages required.

```
python >= 3.8
# No pip install needed
```

## Architecture

```
options-greeks-pricer/
├── greeks.py                    # Entry point: BS pricing + Greeks
├── analytics/                   # Implied vol, moneyness helpers
├── calibration/                 # Model calibration routines
├── models/                      # Tree models (binomial, trinomial)
├── pricing/                     # Monte Carlo engines
├── volatility/                  # SABR, term structure
├── hedging/                     # Delta hedging P&L
├── exotics/                     # Barrier, Asian, digital
├── risk/                        # VaR, scenario analysis
├── quant_code/                  # Days 20-30 advanced modules
└── run_demos.py                 # Demo runner for all modules
```
