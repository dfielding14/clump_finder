#!/usr/bin/env python
"""
Fit and compare clump volume distribution models using unbinned MLE.

Models compared (same shapes as prior least-squares workflow, but MLE):
  (a) DPL - all parameters free (5 params)
  (b) DPL with gamma=0, beta=-4/9 fixed, m free (3 params)
  (c) Single power law with slope=4/9 fixed (0 params)
  (d) Single power law with free slope (1 param)
  (e) Zipf's law (slope=0) (0 params)
  (f) Beuermann with gamma=0, beta=-4/9 fixed, s free (4 params)
  (g) Beuermann - all parameters free (6 params)

Unbinned MLE fits the PDF directly from individual clump volumes. For models
specified as f(V) = V * dN/dlogV, the PDF in log-space is proportional to
f(V)/V; normalization is handled by integrating over the data range.

Usage:
  python fit_dpl_leastsq.py [sweep_name]

  sweep_name: n320_sweep, n640_sweep, n1280_sweep, n2560_sweep,
              n5120_sweep, or n10240_sweep (default: n10240_sweep).
"""
import glob
import os
import re
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt

# ============================================================================
# Parse command-line arguments
# ============================================================================

clump_out_base = "/lustre/orion/ast207/proj-shared/mpturb/clump_find_drummond/clump_out"

if len(sys.argv) > 1:
    sweep_name = sys.argv[1]
else:
    sweep_name = "n10240_sweep"

base_dir = f"{clump_out_base}/{sweep_name}"

if not os.path.isdir(base_dir):
    print(f"ERROR: Directory not found: {base_dir}")
    sys.exit(1)

match = re.match(r"n(\d+)_sweep", sweep_name)
if match:
    resolution = int(match.group(1))
else:
    print(f"ERROR: Could not parse resolution from sweep name: {sweep_name}")
    sys.exit(1)

print(f"Processing: {sweep_name} (resolution={resolution})")

# ============================================================================
# Auto-discover snapshots with step > 30
# ============================================================================

min_step = 30

all_dirs = sorted(os.listdir(base_dir))
snapshot_dirs = []

for d in all_dirs:
    step_match = re.search(r"step(\d+)", d)
    if step_match:
        step_num = int(step_match.group(1))
        if step_num > min_step:
            stitched_files = glob.glob(f"{base_dir}/{d}/*stitched.npz")
            if stitched_files:
                snapshot_dirs.append(d)

print(f"Found {len(snapshot_dirs)} snapshots with step > {min_step}:")
for d in snapshot_dirs:
    print(f"  {d}")

if len(snapshot_dirs) == 0:
    print("ERROR: No valid snapshots found!")
    sys.exit(1)

# Grid spacing
dx = 1.0 / resolution
dx3 = dx**3

# ============================================================================
# Model functions (return f(V) = V * dN/dlogV)
# ============================================================================

def dpl_exp_cut(V, log_V0, gamma, beta, log_V_cut, m):
    """Double power law with small-V exponential cutoff (A=1)."""
    V = np.asarray(V, dtype=np.float64)
    V0 = 10.0**log_V0
    V_cut = 10.0**log_V_cut
    eps = np.finfo(np.float64).tiny
    core = np.power(V / V0, -gamma) * np.power(1.0 + V / V0, (gamma - beta))
    cut = np.exp(-np.power(V_cut / np.clip(V, eps, None), m))
    return core * cut


def dpl_fixed(V, log_V0, log_V_cut, m):
    """DPL with gamma=0, beta=-4/9 fixed (A=1)."""
    return dpl_exp_cut(V, log_V0, 0.0, -4.0 / 9.0, log_V_cut, m)


def power_law(V, alpha):
    """Single power law for V*dN/dlogV (A=1)."""
    V = np.asarray(V, dtype=np.float64)
    return np.power(V, alpha)


def beuermann_exp_cut(V, log_V0, gamma, beta, s, log_V_cut, m):
    """Beuermann smoothly broken power law with small-V cutoff (A=1)."""
    V = np.asarray(V, dtype=np.float64)
    V0 = 10.0**log_V0
    V_cut = 10.0**log_V_cut
    eps = np.finfo(np.float64).tiny

    ratio = V / V0
    ratio_s = np.clip(np.power(ratio, s), 0.0, 1e100)
    core = np.power(ratio, -gamma) * np.power(1.0 + ratio_s, (gamma - beta) / s)
    cut = np.exp(-np.power(V_cut / np.clip(V, eps, None), m))
    return core * cut


def beuermann_fixed(V, log_V0, s, log_V_cut, m):
    """Beuermann with gamma=0, beta=-4/9 fixed (A=1)."""
    return beuermann_exp_cut(V, log_V0, 0.0, -4.0 / 9.0, s, log_V_cut, m)


# ============================================================================
# MLE utilities
# ============================================================================

_NORM_N = 256
_NORM_NODES, _NORM_WEIGHTS = np.polynomial.legendre.leggauss(_NORM_N)


def _logspace_norm(model_func, params, vmin, vmax):
    """Fast log-space normalization via Gauss-Legendre quadrature."""
    log_vmin = np.log10(vmin)
    log_vmax = np.log10(vmax)
    half = 0.5 * (log_vmax - log_vmin)
    mid = 0.5 * (log_vmax + log_vmin)
    log_v = half * _NORM_NODES + mid
    v = 10.0**log_v
    integrand = model_func(v, *params) / v
    return half * np.sum(_NORM_WEIGHTS * integrand)


def nll_generic(model_func, params, V_data, vmin=None, vmax=None):
    if V_data.size == 0:
        return np.inf
    if vmin is None:
        vmin = V_data.min()
    if vmax is None:
        vmax = V_data.max()

    model_vals = model_func(V_data, *params)
    if np.any(model_vals <= 0):
        return np.inf

    norm = _logspace_norm(model_func, params, vmin, vmax)
    if not np.isfinite(norm) or norm <= 0:
        return np.inf

    log_p = np.log(model_vals) - np.log(V_data) - np.log(norm)
    return -np.sum(log_p)


def powerlaw_negloglik(alpha, V_data):
    """Analytic NLL for power law: f(V) = V^alpha => p(logV) ∝ 10^((alpha-1)x)."""
    x = np.log10(V_data)
    n = len(x)
    x_min, x_max = x.min(), x.max()

    beta = alpha - 1.0
    ln10 = np.log(10.0)

    if abs(beta) < 1e-12:
        return n * np.log(x_max - x_min)

    log_terms = np.array([beta * ln10 * x_max, beta * ln10 * x_min])
    if beta > 0:
        log_norm_denom = log_terms[0] + np.log(1.0 - np.exp(log_terms[1] - log_terms[0]))
    else:
        log_norm_denom = log_terms[1] + np.log(1.0 - np.exp(log_terms[0] - log_terms[1]))

    log_norm = np.log(abs(beta) * ln10) - log_norm_denom
    log_p = log_norm + beta * ln10 * x
    return -np.sum(log_p)


# ============================================================================
# Load and combine snapshots
# ============================================================================

print("=" * 70)
print("LOADING SNAPSHOTS")
print("=" * 70)

all_volumes = []
snapshot_info = []

for snap_dir in snapshot_dirs:
    files = glob.glob(f"{base_dir}/{snap_dir}/*stitched.npz")
    if not files:
        print(f"  WARNING: No stitched file found in {snap_dir}")
        continue

    data = np.load(files[0])
    vols = data["volume"] / dx3
    total_vol = vols.sum() * dx3

    all_volumes.append(vols)
    snapshot_info.append({
        "name": snap_dir,
        "n_clumps": len(vols),
        "total_vol": total_vol,
        "vols": vols,
    })

    print(f"  {snap_dir}: {len(vols):,} clumps, total_vol={total_vol:.4e}")

print(f"\nTotal snapshots loaded: {len(snapshot_info)}")

# Combine volumes for MLE
V_all = np.concatenate([info["vols"] for info in snapshot_info])
V_all = V_all[np.isfinite(V_all)]
V_all = V_all[V_all > 0]

print()
print(f"Combined clumps: {V_all.size:,}")
print(f"Volume range: [{V_all.min():.2e}, {V_all.max():.2e}] dx^3")

# For consistency with previous workflow, use V > 1 for fitting
V_fit = V_all[V_all > 1.0]
if V_fit.size == 0:
    print("ERROR: No clumps with V > 1.")
    sys.exit(1)

print(f"Fitting range: V > 1 (N={V_fit.size:,})")

# Optimization controls (reduce full-data iterations for speed)
MAX_OPT_POINTS = 300_000
OPT_SEED = 12345
OPT_MAXITER = 80
FULL_MAXITER = 15

if V_fit.size > MAX_OPT_POINTS:
    rng = np.random.default_rng(OPT_SEED)
    sel = rng.choice(V_fit.size, size=MAX_OPT_POINTS, replace=False)
    V_opt = V_fit[sel]
    print(f"Optimization sample: {V_opt.size:,} clumps (seed={OPT_SEED})")
    print(f"Refining on full data with maxiter={FULL_MAXITER}")
else:
    V_opt = V_fit


def _multistart_opt(model_func, initial_guesses, bounds, V_data, maxiter):
    best_fun = np.inf
    best_x = None
    for x0 in initial_guesses:
        res = minimize(lambda p: nll_generic(model_func, p, V_data), x0,
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter})
        if res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x
    return best_fun, best_x


def _refine_full(model_func, params, bounds, V_data, maxiter):
    res = minimize(lambda p: nll_generic(model_func, p, V_data), params,
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter})
    return res.x, res.fun

# ============================================================================
# Fit models using unbinned MLE
# ============================================================================

print()
print("=" * 70)
print("FITTING MODELS WITH UNBINNED MLE")
print("=" * 70)

# --- Model (a): DPL free ---
print("\nMODEL (a): DPL - all parameters free (5 params)")

bounds_a = [(0.0, 12.0), (-3.0, 3.0), (-3.0, -0.25), (0.0, 10.0), (0.1, 10.0)]
initial_guesses_a = [
    [2.0, -0.5, -0.4, 1.0, 2.0],
    [3.0, -0.7, -0.3, 1.5, 2.5],
    [1.5, -0.3, -0.6, 0.5, 1.5],
]

best_nll_a, best_params_a = _multistart_opt(
    dpl_exp_cut, initial_guesses_a, bounds_a, V_opt, OPT_MAXITER
)
if V_opt is not V_fit:
    best_params_a, best_nll_a = _refine_full(
        dpl_exp_cut, best_params_a, bounds_a, V_fit, FULL_MAXITER
    )

log_V0_a, gamma_a, beta_a, log_Vcut_a, m_a = best_params_a
print(f"  V0=10^{log_V0_a:.2f}, gamma={gamma_a:.3f}, beta={beta_a:.3f}, V_cut=10^{log_Vcut_a:.2f}, m={m_a:.3f}")

# --- Model (b): DPL fixed slopes ---
print("\nMODEL (b): DPL gamma=0, beta=-4/9 fixed, m free (3 params)")

bounds_b = [(0.0, 12.0), (0.0, 10.0), (0.1, 10.0)]
initial_guesses_b = [
    [2.0, 1.0, 2.0],
    [3.0, 1.5, 2.5],
    [1.5, 0.5, 1.5],
]

best_nll_b, best_params_b = _multistart_opt(
    dpl_fixed, initial_guesses_b, bounds_b, V_opt, OPT_MAXITER
)
if V_opt is not V_fit:
    best_params_b, best_nll_b = _refine_full(
        dpl_fixed, best_params_b, bounds_b, V_fit, FULL_MAXITER
    )

log_V0_b, log_Vcut_b, m_b = best_params_b
print(f"  V0=10^{log_V0_b:.2f}, V_cut=10^{log_Vcut_b:.2f}, m={m_b:.3f}")

# --- Model (c): Power law slope=4/9 ---
print("\nMODEL (c): Power law slope=4/9 fixed (0 params)")

nll_c = powerlaw_negloglik(4.0 / 9.0, V_fit)

# --- Model (d): Power law free slope ---
print("\nMODEL (d): Power law free slope (1 param)")

res_d = minimize_scalar(lambda a: powerlaw_negloglik(a, V_fit), bounds=(-2.0, 2.0), method="bounded")
alpha_d = res_d.x
nll_d = res_d.fun
print(f"  slope={alpha_d:.4f}")

# --- Model (e): Zipf ---
print("\nMODEL (e): Zipf's law slope=0 (0 params)")

nll_e = powerlaw_negloglik(0.0, V_fit)

# --- Model (f): Beuermann fixed slopes ---
print("\nMODEL (f): Beuermann gamma=0, beta=-4/9 fixed, s free (4 params)")

bounds_f = [(0.0, 12.0), (0.1, 50.0), (0.0, 10.0), (0.1, 10.0)]
initial_guesses_f = [
    [2.0, 1.0, 1.0, 2.0],
    [3.0, 2.0, 1.5, 2.5],
    [1.5, 0.5, 0.5, 1.5],
]

best_nll_f, best_params_f = _multistart_opt(
    beuermann_fixed, initial_guesses_f, bounds_f, V_opt, OPT_MAXITER
)
if V_opt is not V_fit:
    best_params_f, best_nll_f = _refine_full(
        beuermann_fixed, best_params_f, bounds_f, V_fit, FULL_MAXITER
    )

log_V0_f, s_f, log_Vcut_f, m_f = best_params_f
print(f"  V0=10^{log_V0_f:.2f}, s={s_f:.3f}, V_cut=10^{log_Vcut_f:.2f}, m={m_f:.3f}")

# --- Model (g): Beuermann free ---
print("\nMODEL (g): Beuermann - all parameters free (6 params)")

bounds_g = [(0.0, 12.0), (-3.0, 3.0), (-3.0, -0.25), (0.1, 50.0), (0.0, 10.0), (0.1, 10.0)]
initial_guesses_g = [
    [2.0, -0.5, -0.4, 1.0, 1.0, 2.0],
    [3.0, -0.7, -0.3, 1.5, 1.5, 2.5],
    [1.5, -0.3, -0.6, 0.8, 0.5, 1.5],
]

best_nll_g, best_params_g = _multistart_opt(
    beuermann_exp_cut, initial_guesses_g, bounds_g, V_opt, OPT_MAXITER
)
if V_opt is not V_fit:
    best_params_g, best_nll_g = _refine_full(
        beuermann_exp_cut, best_params_g, bounds_g, V_fit, FULL_MAXITER
    )

log_V0_g, gamma_g, beta_g, s_g, log_Vcut_g, m_g = best_params_g
print(f"  V0=10^{log_V0_g:.2f}, gamma={gamma_g:.3f}, beta={beta_g:.3f}, s={s_g:.3f}, V_cut=10^{log_Vcut_g:.2f}, m={m_g:.3f}")

# ============================================================================
# Compute AIC for each model over three V ranges
# ============================================================================

models = [
    ("(a) DPL free", dpl_exp_cut, best_params_a, 5),
    ("(b) DPL fixed", dpl_fixed, best_params_b, 3),
    ("(c) PL 4/9", power_law, [4.0 / 9.0], 0),
    ("(d) PL free", power_law, [alpha_d], 1),
    ("(e) Zipf", power_law, [0.0], 0),
    ("(f) Beuermann fix", beuermann_fixed, best_params_f, 4),
    ("(g) Beuermann free", beuermann_exp_cut, best_params_g, 6),
]

V_cuts = [1.0, 1e3, 1e6]
V_cut_labels = ["V > 1", "V > 10^3", "V > 10^6"]

print()
print("=" * 70)
print("AIC COMPARISON ACROSS VOLUME RANGES (UNBINNED MLE)")
print(f"(Combined from {len(snapshot_info)} snapshots)")
print("=" * 70)

aic_results = {label: {} for label in V_cut_labels}

for V_min, label in zip(V_cuts, V_cut_labels):
    V_sub = V_fit[V_fit > V_min]
    if V_sub.size == 0:
        print(f"\n{label}: No data")
        continue

    print(f"\n{label} (N={V_sub.size} clumps):")
    print("-" * 60)

    aic_values = []
    for name, model_func, params, k in models:
        if model_func is power_law:
            alpha = params[0]
            nll = powerlaw_negloglik(alpha, V_sub)
        else:
            nll = nll_generic(model_func, params, V_sub, vmin=V_sub.min(), vmax=V_sub.max())
        aic = 2 * k + 2 * nll
        aic_values.append((name, aic, nll, k))
        aic_results[label][name] = aic

    min_aic = min(av[1] for av in aic_values)

    print(f"  {'Model':<18} {'AIC':>10} {'Delta':>10} {'NLL':>12} {'k':>5}")
    for name, aic, nll, k in sorted(aic_values, key=lambda x: x[1]):
        delta_aic = aic - min_aic
        print(f"  {name:<18} {aic:>10.2f} {delta_aic:>10.2f} {nll:>12.2f} {k:>5}")

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("\nDelta AIC interpretation:")
print("  0-2:   Substantial support")
print("  2-4:   Some support")
print("  4-7:   Considerably less support")
print("  >10:   Essentially no support")

# ============================================================================
# Plot (binned for visualization; MLE for model shapes)
# ============================================================================

print()
print("Generating plot...")

V_plot = V_fit
logV = np.log10(V_plot)

n_bins = 60
bin_edges = np.linspace(np.log10(1.0), np.log10(V_plot.max() * 1.1), n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
V_centers = 10**bin_centers
hist, _ = np.histogram(logV, bins=bin_edges)
y_data = hist * V_centers
mask_data = hist > 0

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

colors = {
    "(a) DPL free": "red",
    "(b) DPL fixed": "blue",
    "(c) PL 4/9": "green",
    "(d) PL free": "magenta",
    "(e) Zipf": "orange",
    "(f) Beuermann fix": "cyan",
    "(g) Beuermann free": "brown",
}

linestyles = {
    "(a) DPL free": "-",
    "(b) DPL fixed": "--",
    "(c) PL 4/9": ":",
    "(d) PL free": "-.",
    "(e) Zipf": ":",
    "(f) Beuermann fix": "--",
    "(g) Beuermann free": "-",
}

for ax, (V_min, label) in zip(axes, zip(V_cuts, V_cut_labels)):
    V_sub = V_plot[V_plot > V_min]
    if V_sub.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        continue

    # Plot binned data for reference
    ax.scatter(V_centers[mask_data], y_data[mask_data], s=20, c="black", label="Data")

    V_model = np.logspace(np.log10(V_sub.min()), np.log10(V_sub.max()), 200)

    # Model curves scaled by N/norm for visualization
    label_aic = aic_results.get(label, {})
    min_aic = min(label_aic.values()) if label_aic else 0.0

    for name, model_func, params, _k in models:
        if model_func is power_law:
            alpha = params[0]
            g_vals = power_law(V_model, alpha)
            norm = _logspace_norm(power_law, [alpha], V_sub.min(), V_sub.max())
        else:
            g_vals = model_func(V_model, *params)
            norm = _logspace_norm(model_func, params, V_sub.min(), V_sub.max())
        y_model = V_sub.size * g_vals / norm

        delta_aic = label_aic.get(name, 0.0) - min_aic
        lw = 2.5 if delta_aic < 2.0 else 1.5
        alpha = 1.0 if delta_aic < 4.0 else 0.5
        ax.plot(V_model, y_model, color=colors[name], ls=linestyles[name],
                lw=lw, alpha=alpha, label=f"{name}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$V$ [$\Delta x^3$]", fontsize=12)
    ax.set_ylabel(r"$V \times dN/d\log V$", fontsize=12)
    ax.set_title(f"{label}", fontsize=12)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f"MLE Model Comparison: n{resolution}, conn6, T=0.02 (combined {len(snapshot_info)} snapshots)\n"
    "Unbinned MLE over V > 1; curves scaled for visualization",
    fontsize=11,
    y=1.05,
)
plt.tight_layout()

plot_path = f"{base_dir}/aic_model_comparison_mle_combined.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to: {plot_path}")
plt.close()

print()
print("Done!")
