#!/usr/bin/env python
"""
Compare power-law vs Zipf vs double-power-law models for clump volume distribution.
Uses unbinned MLE with AIC/BIC to determine best fit for V * dN/dlogV ~ V^alpha.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt

# Load data
npz_path = "/lustre/orion/ast207/proj-shared/mpturb/clump_find_drummond/clump_out/n10240_sweep/conn6_T0p02_final_step00037/clumps_stitched.npz"
data = np.load(npz_path)
volumes = data["volume"]

# Grid spacing for n10240: box size 1, Nres=10240
dx = 1.0 / 10240
dx3 = dx**3

# Normalize volumes to units of dx^3
V_norm = volumes / dx3
V_norm = V_norm[np.isfinite(V_norm)]
V_norm = V_norm[V_norm > 0]

print(f"Total clumps: {len(volumes)}")
print(f"Volume range: [{V_norm.min():.2e}, {V_norm.max():.2e}] dx^3")
print()

# ============================================================================
# Model definitions (return f(V) = V * dN/dlogV)
# ============================================================================

def power_law(V, alpha):
    V = np.asarray(V, dtype=np.float64)
    return np.power(V, alpha)


def powerlaw_negloglik(alpha, V_data):
    """Analytic NLL for power law in log space."""
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


def dpl_exp_cut(V, log_V0, gamma, beta, log_V_cut, m=2.0):
    """Double power law with small-V exponential cutoff (A=1)."""
    V = np.asarray(V, dtype=np.float64)
    V0 = 10.0**log_V0
    V_cut = 10.0**log_V_cut
    eps = np.finfo(np.float64).tiny
    core = np.power(V / V0, -gamma) * np.power(1.0 + V / V0, (gamma - beta))
    cut = np.exp(-np.power(V_cut / np.clip(V, eps, None), m))
    return core * cut


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


def _model_vn_gt(model_func, params, v_grid, vmin, vmax, n_total):
    """Compute V * N(>V) for a model defined as f(V)=V*dN/dlogV."""
    log_min = np.log10(vmin)
    log_max = np.log10(vmax)
    log_grid = np.linspace(log_min, log_max, 2048)
    v_full = 10.0**log_grid
    g = model_func(v_full, *params) / v_full
    g = np.where(np.isfinite(g) & (g > 0), g, 0.0)

    dlog = np.diff(log_grid)
    seg = 0.5 * (g[:-1] + g[1:]) * dlog
    tail = np.empty_like(g)
    tail[-1] = 0.0
    tail[:-1] = np.cumsum(seg[::-1])[::-1]
    norm = tail[0]
    if not np.isfinite(norm) or norm <= 0:
        return np.full_like(v_grid, np.nan)

    tail_interp = np.interp(np.log10(v_grid), log_grid, tail)
    n_gt = n_total * tail_interp / norm
    return v_grid * n_gt


def _data_vn_gt(volumes, v_grid):
    vols = np.sort(volumes)
    idx = np.searchsorted(vols, v_grid, side="left")
    n_gt = vols.size - idx
    return v_grid * n_gt


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


def fit_dpl_multistart(V_data, initial_guesses, bounds):
    """Fit DPL with multiple starting points, return best."""
    best_nll = np.inf
    best_params = None

    for x0 in initial_guesses:
        res = minimize(lambda p: nll_generic(dpl_exp_cut, p, V_data), x0,
                       method="L-BFGS-B", bounds=bounds)
        if res.fun < best_nll:
            best_nll = res.fun
            best_params = res.x

    return best_nll, best_params


def fit_dpl_fixed_slopes(V_data, gamma_fixed, beta_fixed, initial_guesses_2d, bounds_2d):
    """Fit DPL with fixed gamma and beta, optimizing only V0 and V_cut."""
    best_nll = np.inf
    best_params = None

    def dpl_fixed(V, log_V0, log_V_cut):
        return dpl_exp_cut(V, log_V0, gamma_fixed, beta_fixed, log_V_cut)

    for x0 in initial_guesses_2d:
        res = minimize(lambda p: nll_generic(dpl_fixed, p, V_data), x0,
                       method="L-BFGS-B", bounds=bounds_2d)
        if res.fun < best_nll:
            best_nll = res.fun
            best_params = res.x

    return best_nll, best_params


# ============================================================================
# ANALYSIS 1: Full range comparison (V > 1 dx^3)
# ============================================================================
print("=" * 70)
print("ANALYSIS 1: FULL RANGE (V > 1 dx^3)")
print("=" * 70)

V_full = V_norm[V_norm > 1]
n_full = len(V_full)
print(f"N clumps: {n_full}")
print(f"Volume range: [{V_full.min():.2e}, {V_full.max():.2e}] dx^3")
print()

# --- Model 1: Power law (free alpha) ---
res_pl = minimize_scalar(lambda a: powerlaw_negloglik(a, V_full), bounds=(-2, 2), method="bounded")
alpha_fit_full = res_pl.x
nll_pl_full = res_pl.fun

# --- Model 2: Zipf (alpha=0) ---
nll_zipf_full = powerlaw_negloglik(0.0, V_full)

# --- Model 3: Power law with alpha = 4/9 fixed ---
nll_49_full = powerlaw_negloglik(4.0 / 9.0, V_full)

# --- Model 4: DPL with free parameters ---
print("Fitting double power law (free params)...")
# Restrict beta <= -0.25 to keep large-V slope increasing/flat
bounds_dpl_constrained = [(0, 10), (-3, 0), (-3, -0.25), (-2, 5)]

initial_guesses_full = [
    [2.0, -0.5, -0.5, 0.0],
    [2.5, -0.7, -0.3, 0.5],
    [3.0, -0.8, -0.4, 0.5],
    [1.5, -0.4, -0.6, -0.5],
    [np.log10(np.median(V_full)), -alpha_fit_full, -0.5, 0.0],
]

nll_dpl_full, params_dpl_full = fit_dpl_multistart(V_full, initial_guesses_full, bounds_dpl_constrained)

if params_dpl_full is not None:
    log_V0_fit, gamma_fit, beta_fit, log_Vcut_fit = params_dpl_full
else:
    log_V0_fit, gamma_fit, beta_fit, log_Vcut_fit = np.nan, np.nan, np.nan, np.nan
    nll_dpl_full = np.inf

# --- Model 5: DPL with fixed slopes (gamma=0, beta=-4/9) ---
print("Fitting double power law (fixed slopes: 0 at small V, 4/9 at large V)...")
gamma_fixed = 0.0
beta_fixed = -4.0 / 9.0

bounds_2d = [(0, 10), (-2, 5)]
initial_guesses_2d = [
    [2.0, 0.0],
    [3.0, 0.5],
    [1.5, -0.5],
    [2.5, 1.0],
]

nll_dpl_fixed_full, params_dpl_fixed_full = fit_dpl_fixed_slopes(
    V_full, gamma_fixed, beta_fixed, initial_guesses_2d, bounds_2d
)

if params_dpl_fixed_full is not None:
    log_V0_fixed, log_Vcut_fixed = params_dpl_fixed_full
else:
    log_V0_fixed, log_Vcut_fixed = np.nan, np.nan
    nll_dpl_fixed_full = np.inf

# --- Compute AIC/BIC ---
k_pl, k_zipf, k_49, k_dpl, k_dpl_fixed = 1, 0, 0, 4, 2

aic_pl = 2 * k_pl + 2 * nll_pl_full
aic_zipf = 2 * k_zipf + 2 * nll_zipf_full
aic_49 = 2 * k_49 + 2 * nll_49_full
aic_dpl = 2 * k_dpl + 2 * nll_dpl_full
aic_dpl_fixed = 2 * k_dpl_fixed + 2 * nll_dpl_fixed_full

bic_pl = k_pl * np.log(n_full) + 2 * nll_pl_full
bic_zipf = k_zipf * np.log(n_full) + 2 * nll_zipf_full
bic_49 = k_49 * np.log(n_full) + 2 * nll_49_full
bic_dpl = k_dpl * np.log(n_full) + 2 * nll_dpl_full
bic_dpl_fixed = k_dpl_fixed * np.log(n_full) + 2 * nll_dpl_fixed_full

print()
print(f"{'Model':<40} {'k':>3} {'NLL':>14} {'AIC':>14} {'BIC':>14}")
print("-" * 90)
print(f"{'Power law (free alpha)':<40} {k_pl:>3} {nll_pl_full:>14.2f} {aic_pl:>14.2f} {bic_pl:>14.2f}")
print(f"{'Zipf (alpha=0)':<40} {k_zipf:>3} {nll_zipf_full:>14.2f} {aic_zipf:>14.2f} {bic_zipf:>14.2f}")
print(f"{'Power law (alpha=4/9 fixed)':<40} {k_49:>3} {nll_49_full:>14.2f} {aic_49:>14.2f} {bic_49:>14.2f}")
print(f"{'DPL + exp cutoff (free)':<40} {k_dpl:>3} {nll_dpl_full:>14.2f} {aic_dpl:>14.2f} {bic_dpl:>14.2f}")
print(f"{'DPL (slope=0 small, 4/9 large)':<40} {k_dpl_fixed:>3} {nll_dpl_fixed_full:>14.2f} {aic_dpl_fixed:>14.2f} {bic_dpl_fixed:>14.2f}")

print()
print(f"Best-fit power law: alpha = {alpha_fit_full:.4f}")
if not np.isnan(log_V0_fit):
    print(f"Best-fit DPL (free): V0=10^{log_V0_fit:.2f}, gamma={gamma_fit:.3f}, beta={beta_fit:.3f}, V_cut=10^{log_Vcut_fit:.2f}")
    print(f"  -> slope at V<<V0: {-gamma_fit:.3f}, slope at V>>V0: {-beta_fit:.3f}")
if not np.isnan(log_V0_fixed):
    print(f"DPL (fixed slopes): V0=10^{log_V0_fixed:.2f}, V_cut=10^{log_Vcut_fixed:.2f}")

# Delta AIC relative to best
best_aic = min(aic_pl, aic_zipf, aic_49, aic_dpl, aic_dpl_fixed)
print()
print("Delta AIC relative to best:")
print(f"  Power law (free alpha):           Delta AIC = {aic_pl - best_aic:+.1f}")
print(f"  Zipf (alpha=0):                   Delta AIC = {aic_zipf - best_aic:+.1f}")
print(f"  Power law (alpha=4/9):            Delta AIC = {aic_49 - best_aic:+.1f}")
print(f"  DPL (free):                       Delta AIC = {aic_dpl - best_aic:+.1f}")
print(f"  DPL (slope=0 small, 4/9 large):   Delta AIC = {aic_dpl_fixed - best_aic:+.1f}")

# ============================================================================
# ANALYSIS 2: Zipf vs Power law at increasing V_min cutoffs
# ============================================================================
print()
print("=" * 70)
print("ANALYSIS 2: ZIPF vs POWER LAW AT INCREASING V_min CUTOFFS")
print("=" * 70)
print(f"{'log10(V_min)':<12} {'N_clumps':>10} {'alpha_fit':>10} {'NLL_PL':>12} {'NLL_Zipf':>12} {'DeltaAIC(Zipf-PL)':>18}")
print("-" * 80)

log_vmin_values = np.arange(0, 9, 1)
results_cutoff = []

for log_vmin in log_vmin_values:
    V_min = 10.0**log_vmin
    V_sub = V_norm[V_norm > V_min]
    n_sub = len(V_sub)

    if n_sub < 20:
        continue

    res = minimize_scalar(lambda a: powerlaw_negloglik(a, V_sub), bounds=(-2, 2), method="bounded")
    alpha_sub = res.x
    nll_pl = res.fun

    nll_zipf = powerlaw_negloglik(0.0, V_sub)

    aic_pl = 2 * 1 + 2 * nll_pl
    aic_zipf = 2 * 0 + 2 * nll_zipf
    delta_aic = aic_zipf - aic_pl

    results_cutoff.append((log_vmin, n_sub, alpha_sub, nll_pl, nll_zipf, delta_aic))
    print(f"{log_vmin:<12.0f} {n_sub:>10} {alpha_sub:>10.4f} {nll_pl:>12.2f} {nll_zipf:>12.2f} {delta_aic:>+18.1f}")

print()
print("Interpretation: Delta AIC > 0 means power law is preferred; Delta AIC < 0 means Zipf is preferred")

# ============================================================================
# ANALYSIS 3: Five-way comparison at V > 1e3
# ============================================================================
print()
print("=" * 70)
print("ANALYSIS 3: FIVE-WAY COMPARISON (V > 10^3 dx^3)")
print("=" * 70)

V_cut3 = V_norm[V_norm > 1e3]
n_cut3 = len(V_cut3)
print(f"N clumps: {n_cut3}")
print()

res_pl3 = minimize_scalar(lambda a: powerlaw_negloglik(a, V_cut3), bounds=(-2, 2), method="bounded")
alpha_fit3 = res_pl3.x
nll_pl3 = res_pl3.fun

nll_zipf3 = powerlaw_negloglik(0.0, V_cut3)

nll_49_3 = powerlaw_negloglik(4.0 / 9.0, V_cut3)

print("Fitting DPL (free)...")
initial_guesses_3 = [
    [4.0, -0.1, -0.5, 2.0],
    [5.0, -0.2, -0.3, 2.5],
    [3.5, -0.05, -0.6, 1.5],
    [np.log10(np.median(V_cut3)), -alpha_fit3, -0.5, 2.0],
]

nll_dpl3, params_dpl3 = fit_dpl_multistart(V_cut3, initial_guesses_3, bounds_dpl_constrained)
if params_dpl3 is not None:
    log_V0_3, gamma_3, beta_3, log_Vcut_3 = params_dpl3
else:
    log_V0_3, gamma_3, beta_3, log_Vcut_3 = np.nan, np.nan, np.nan, np.nan
    nll_dpl3 = np.inf

print("Fitting DPL (fixed slopes)...")
initial_guesses_3_2d = [
    [4.0, 2.0],
    [5.0, 2.5],
    [3.5, 1.5],
    [6.0, 3.0],
]

nll_dpl_fixed3, params_dpl_fixed3 = fit_dpl_fixed_slopes(
    V_cut3, gamma_fixed, beta_fixed, initial_guesses_3_2d, bounds_2d
)

if params_dpl_fixed3 is not None:
    log_V0_fixed3, log_Vcut_fixed3 = params_dpl_fixed3
else:
    log_V0_fixed3, log_Vcut_fixed3 = np.nan, np.nan
    nll_dpl_fixed3 = np.inf

# AIC/BIC
aic_pl3 = 2 * 1 + 2 * nll_pl3
aic_zipf3 = 2 * 0 + 2 * nll_zipf3
aic_49_3 = 2 * 0 + 2 * nll_49_3
aic_dpl3 = 2 * 4 + 2 * nll_dpl3
aic_dpl_fixed3 = 2 * 2 + 2 * nll_dpl_fixed3

bic_pl3 = 1 * np.log(n_cut3) + 2 * nll_pl3
bic_zipf3 = 0 * np.log(n_cut3) + 2 * nll_zipf3
bic_49_3 = 0 * np.log(n_cut3) + 2 * nll_49_3
bic_dpl3 = 4 * np.log(n_cut3) + 2 * nll_dpl3
bic_dpl_fixed3 = 2 * np.log(n_cut3) + 2 * nll_dpl_fixed3

print()
print(f"{'Model':<40} {'k':>3} {'NLL':>14} {'AIC':>14} {'BIC':>14}")
print("-" * 90)
print(f"{'Power law (alpha={:.4f})':<40} {1:>3} {nll_pl3:>14.2f} {aic_pl3:>14.2f} {bic_pl3:>14.2f}".format(alpha_fit3))
print(f"{'Zipf (alpha=0)':<40} {0:>3} {nll_zipf3:>14.2f} {aic_zipf3:>14.2f} {bic_zipf3:>14.2f}")
print(f"{'Power law (alpha=4/9 fixed)':<40} {0:>3} {nll_49_3:>14.2f} {aic_49_3:>14.2f} {bic_49_3:>14.2f}")
print(f"{'DPL + exp cutoff (free)':<40} {4:>3} {nll_dpl3:>14.2f} {aic_dpl3:>14.2f} {bic_dpl3:>14.2f}")
print(f"{'DPL (slope=0 small, 4/9 large)':<40} {2:>3} {nll_dpl_fixed3:>14.2f} {aic_dpl_fixed3:>14.2f} {bic_dpl_fixed3:>14.2f}")

if not np.isnan(log_V0_3):
    print()
    print(f"DPL (free) params: V0=10^{log_V0_3:.2f}, gamma={gamma_3:.3f}, beta={beta_3:.3f}, V_cut=10^{log_Vcut_3:.2f}")
    print(f"  -> slope at V<<V0: {-gamma_3:.3f}, slope at V>>V0: {-beta_3:.3f}")
if not np.isnan(log_V0_fixed3):
    print(f"DPL (fixed) params: V0=10^{log_V0_fixed3:.2f}, V_cut=10^{log_Vcut_fixed3:.2f}")

best_aic3 = min(aic_pl3, aic_zipf3, aic_49_3, aic_dpl3, aic_dpl_fixed3)
print()
print("Delta AIC relative to best:")
print(f"  Power law (free alpha):           Delta AIC = {aic_pl3 - best_aic3:+.1f}")
print(f"  Zipf (alpha=0):                   Delta AIC = {aic_zipf3 - best_aic3:+.1f}")
print(f"  Power law (alpha=4/9):            Delta AIC = {aic_49_3 - best_aic3:+.1f}")
print(f"  DPL (free):                       Delta AIC = {aic_dpl3 - best_aic3:+.1f}")
print(f"  DPL (slope=0 small, 4/9 large):   Delta AIC = {aic_dpl_fixed3 - best_aic3:+.1f}")

# ============================================================================
# Diagnostic Plot
# ============================================================================
print()
print("=" * 70)
print("Generating diagnostic plot...")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left panel: Full range ---
ax = axes[0]

V_plot_data = V_norm[V_norm > 1]
vmin = V_plot_data.min()
vmax = V_plot_data.max()
V_model = np.logspace(np.log10(vmin), np.log10(vmax), 240)

y_data = _data_vn_gt(V_plot_data, V_model)
ax.plot(V_model, y_data, "k.", ms=3, label="Data (V*N(>V))")

ax.plot(V_model,
        _model_vn_gt(power_law, [alpha_fit_full], V_model, vmin, vmax, V_plot_data.size),
        "b-", lw=2, label=f"Power law (alpha={alpha_fit_full:.3f})")

ax.plot(V_model,
        _model_vn_gt(power_law, [0.0], V_model, vmin, vmax, V_plot_data.size),
        "g:", lw=2, label="Zipf (alpha=0)")

ax.plot(V_model,
        _model_vn_gt(power_law, [4.0 / 9.0], V_model, vmin, vmax, V_plot_data.size),
        "c--", lw=2, label="Power law (alpha=4/9)")

if not np.isnan(log_V0_fit):
    ax.plot(V_model,
            _model_vn_gt(dpl_exp_cut, [log_V0_fit, gamma_fit, beta_fit, log_Vcut_fit],
                         V_model, vmin, vmax, V_plot_data.size),
            color="tab:orange", lw=2, label="DPL free (best)")
if not np.isnan(log_V0_fixed):
    ax.plot(V_model,
            _model_vn_gt(lambda v, log_V0, log_V_cut: dpl_exp_cut(v, log_V0, gamma_fixed, beta_fixed, log_V_cut),
                         [log_V0_fixed, log_Vcut_fixed], V_model, vmin, vmax, V_plot_data.size),
            color="tab:purple", lw=2, label="DPL fixed")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$V$ [$\Delta x^3$]", fontsize=12)
ax.set_ylabel(r"$V \times N(>V)$", fontsize=12)
ax.set_title("Full range (V > 1)", fontsize=12)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.3)

# --- Right panel: V > 10^3 ---
ax = axes[1]

vmin3 = V_cut3.min()
vmax3 = V_cut3.max()
V_model3 = np.logspace(np.log10(vmin3), np.log10(vmax3), 240)

y_data3 = _data_vn_gt(V_cut3, V_model3)
ax.plot(V_model3, y_data3, "k.", ms=3, label="Data (V*N(>V))")

ax.plot(V_model3,
        _model_vn_gt(power_law, [alpha_fit3], V_model3, vmin3, vmax3, V_cut3.size),
        "b-", lw=2, label=f"Power law (alpha={alpha_fit3:.3f})")

ax.plot(V_model3,
        _model_vn_gt(power_law, [0.0], V_model3, vmin3, vmax3, V_cut3.size),
        "g:", lw=2, label="Zipf (alpha=0)")

ax.plot(V_model3,
        _model_vn_gt(power_law, [4.0 / 9.0], V_model3, vmin3, vmax3, V_cut3.size),
        "c--", lw=2, label="Power law (alpha=4/9)")

if not np.isnan(log_V0_3):
    ax.plot(V_model3,
            _model_vn_gt(dpl_exp_cut, [log_V0_3, gamma_3, beta_3, log_Vcut_3],
                         V_model3, vmin3, vmax3, V_cut3.size),
            color="tab:orange", lw=2, label="DPL free")
if not np.isnan(log_V0_fixed3):
    ax.plot(V_model3,
            _model_vn_gt(lambda v, log_V0, log_V_cut: dpl_exp_cut(v, log_V0, gamma_fixed, beta_fixed, log_V_cut),
                         [log_V0_fixed3, log_Vcut_fixed3], V_model3, vmin3, vmax3, V_cut3.size),
            color="tab:purple", lw=2, label="DPL fixed")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$V$ [$\Delta x^3$]", fontsize=12)
ax.set_ylabel(r"$V \times N(>V)$", fontsize=12)
ax.set_title(r"V > $10^3$ $\Delta x^3$", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)

plt.suptitle("Clump Volume Distribution (unbinned MLE, cumulative)", fontsize=14, y=1.02)
plt.tight_layout()

plot_path = "/lustre/orion/ast207/proj-shared/mpturb/clump_find_drummond/clump_out/n10240_sweep/conn6_T0p02_final_step00037/volume_dist_model_comparison_mle.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to: {plot_path}")
plt.close()

# ============================================================================
# Additional plot: Delta AIC vs V_min
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

log_vmins = [r[0] for r in results_cutoff]
delta_aics = [r[5] for r in results_cutoff]

ax.plot(log_vmins, delta_aics, "ko-", lw=2, markersize=8)
ax.axhline(0, color="gray", ls="--", lw=1)
ax.axhline(2, color="red", ls=":", lw=1, alpha=0.7)
ax.axhline(-2, color="red", ls=":", lw=1, alpha=0.7)

ax.fill_between(log_vmins, -2, 2, color="gray", alpha=0.2, label="Essentially equivalent")
ax.fill_between(log_vmins, 2, max(delta_aics) + 10, color="blue", alpha=0.1, label="Power law preferred")
ax.fill_between(log_vmins, min(delta_aics) - 10, -2, color="green", alpha=0.1, label="Zipf preferred")

ax.set_xlabel(r"$\log_{10}(V_{\rm min} / \Delta x^3)$", fontsize=12)
ax.set_ylabel(r"Delta AIC (Zipf - Power law)", fontsize=12)
ax.set_title("Model preference vs. minimum volume cutoff", fontsize=12)
ax.legend(loc="best", fontsize=10)
ax.set_xlim(0, 8)
ax.grid(True, alpha=0.3)

plot_path2 = "/lustre/orion/ast207/proj-shared/mpturb/clump_find_drummond/clump_out/n10240_sweep/conn6_T0p02_final_step00037/aic_vs_vmin_mle.png"
plt.savefig(plot_path2, dpi=150, bbox_inches="tight")
print(f"Saved plot to: {plot_path2}")
plt.close()

print()
print("Done!")
