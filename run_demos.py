"""
run_demos.py — Options Greeks Pricer
Runs all Day 20-30 demo modules in sequence.
Pure Python stdlib only — no external dependencies.
"""
import subprocess
import sys
import os
import time

MODULES = [
    "quant_code/options_day20_smile_interpolation.py",
    "quant_code/options_day21_multi_asset.py",
    "quant_code/options_day22_scenario_analysis.py",
    "quant_code/options_day23_jump_diffusion.py",
    "quant_code/options_day24_tail_risk_hedging.py",
    "quant_code/options_day25_local_volatility.py",
    "quant_code/options_day26_stochastic_vol.py",
    "quant_code/options_day27_exotic_options.py",
    "quant_code/options_day28_vol_surface_svi.py",
    "quant_code/options_day29_interest_rate_derivs.py",
    "quant_code/options_day30_greeks_book.py",
]

def run_module(path: str) -> bool:
    name = os.path.basename(path)
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, path], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  ✓ {name} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ✗ {name} failed (exit code {result.returncode})")
        return False

if __name__ == "__main__":
    print("Options Greeks Pricer — Day 20-30 Demo Runner")
    print("=" * 60)

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    passed, failed = 0, 0
    for module in MODULES:
        if run_module(module):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{len(MODULES)} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
