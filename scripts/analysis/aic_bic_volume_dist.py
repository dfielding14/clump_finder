#!/usr/bin/env python
"""
Compare power-law vs Zipf vs double-power-law models for clump volume distribution.
Uses AIC/BIC to determine best fit for V * dN/dlogV ~ V^alpha.
"""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Load data
npz_path = "/lustre/orion/ast207/proj-shared/mpturb/clump_find_drummond/clump_out/n10240_sweep/conn6_T0p02_final_step00037/n10240_conn6_T0p02_step00037_clumps_stitched.npz"
data = np.load(npz_path)
volumes = data['volume']

# Grid spacing for n10240: box size 1, Nres=10240
dx = 1.0 / 10240
dx3 = dx**3

# Normalize volumes to units of dx^3
V_norm = volumes / dx3

print(f"Total clumps: {len(volumes)}")
print(f"Volume range: [{V_norm.min():.2e}, {V_norm.max():.2e}] dx^3")
print()

# ============================================================================
# Model definitions
# ============================================================================

def powerlaw_negloglik(alpha, V_data):
    """Negative log-likelihood for power law: V * dN/dlogV ~ V^alpha.

    The PDF of x = log10(V) is: p(x) ∝ 10^((alpha-1)*x)
    """
    x = np.log10(V_data)
    n = len(x)
    x_min, x_max = x.min(), x.max()

    beta = alpha - 1
    ln10 = np.log(10)

    if abs(beta) < 1e-10:  # beta ≈ 0, uniform in log space
        return n * np.log(x_max - x_min)

    # For numerical stability, compute in log space
    log_terms = np.array([beta * ln10 * x_max, beta * ln10 * x_min])
    if beta > 0:
        log_norm_denom = log_terms[0] + np.log(1 - np.exp(log_terms[1] - log_terms[0]))
    else:
        log_norm_denom = log_terms[1] + np.log(1 - np.exp(log_terms[0] - log_terms[1]))

    log_norm = np.log(np.abs(beta) * ln10) - log_norm_denom
    log_p = log_norm + beta * ln10 * x
    return -np.sum(log_p)


def dpl_exp_cut(V, A, V0, gamma, beta, alpha=1.0, V_cut=1.0, m=2.0):
    """
    Double power law with *small-V* exponential cutoff: as V->0, f->0.

    This function describes V * dN/dlogV:
      f(V) ~ V^(-gamma) at V << V0 (after cutoff kicks in)
      f(V) ~ V^(-beta) at V >> V0

    So effective slopes of V*dN/dlogV are:
      slope_small = -gamma
      slope_large = -beta

    For INCREASING behavior, need gamma < 0 (small V) and beta < 0 (large V).
    """
    V = np.asarray(V, dtype=float)
    eps = np.finfo(float).tiny
    core = A * np.power(V/V0, -gamma) * np.power(1.0 + np.power(V/V0, alpha), (gamma - beta)/alpha)
    cut = np.exp(-np.power(V_cut/np.clip(V, eps, None), m))
    return core * cut


def dpl_negloglik(params, V_data, fixed_gamma=None, fixed_beta=None):
    """Negative log-likelihood for double power law with exponential cutoff.

    V * dN/dlogV = f(V), so dN/dlogV = f(V)/V
    PDF of log10(V): p(x) ∝ f(10^x) / 10^x = f(V) / V

    params can be [log_V0, gamma, beta, log_V_cut] if no fixed values,
    or [log_V0, log_V_cut] if gamma and beta are fixed.
    """
    if fixed_gamma is not None and fixed_beta is not None:
        log_V0, log_V_cut = params
        gamma = fixed_gamma
        beta = fixed_beta
    else:
        log_V0, gamma, beta, log_V_cut = params

    V0 = 10**log_V0
    V_cut = 10**log_V_cut

    x = np.log10(V_data)
    n = len(x)
    x_min, x_max = x.min(), x.max()

    # f(V) / V for each data point (unnormalized PDF in log space)
    f_over_V = dpl_exp_cut(V_data, 1.0, V0, gamma, beta, V_cut=V_cut) / V_data

    # Handle zeros/negatives
    if np.any(f_over_V <= 0):
        return 1e10

    # Compute normalization by numerical integration over [x_min, x_max]
    def integrand(log_v):
        v = 10**log_v
        return dpl_exp_cut(v, 1.0, V0, gamma, beta, V_cut=V_cut) / v

    norm, _ = quad(integrand, x_min, x_max, limit=100)
    if norm <= 0:
        return 1e10

    # Log-likelihood
    log_p = np.log(f_over_V) - np.log(norm)
    return -np.sum(log_p)


def fit_dpl_multistart(V_data, initial_guesses, bounds):
    """Fit DPL with multiple starting points, return best."""
    best_nll = np.inf
    best_params = None

    for x0 in initial_guesses:
        try:
            res = minimize(lambda p: dpl_negloglik(p, V_data), x0,
                          method='L-BFGS-B', bounds=bounds)
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except:
            pass

    return best_nll, best_params


def fit_dpl_fixed_slopes(V_data, gamma_fixed, beta_fixed, initial_guesses_2d, bounds_2d):
    """Fit DPL with fixed gamma and beta, optimizing only V0 and V_cut."""
    best_nll = np.inf
    best_params = None

    for x0 in initial_guesses_2d:
        try:
            res = minimize(lambda p: dpl_negloglik(p, V_data, fixed_gamma=gamma_fixed, fixed_beta=beta_fixed),
                          x0, method='L-BFGS-B', bounds=bounds_2d)
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except:
            pass

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

# --- Model 1: Power law (free α) ---
res_pl = minimize_scalar(lambda a: powerlaw_negloglik(a, V_full), bounds=(-2, 2), method='bounded')
alpha_fit_full = res_pl.x
nll_pl_full = res_pl.fun

# --- Model 2: Zipf (α=0) ---
nll_zipf_full = powerlaw_negloglik(0, V_full)

# --- Model 3: Power law with α = 4/9 fixed ---
nll_49_full = powerlaw_negloglik(4/9, V_full)

# --- Model 4: DPL with free parameters ---
print("Fitting double power law (free params)...")
# For INCREASING V*dN/dlogV, need negative gamma and beta
# Restrict beta <= 0 to ensure increasing/flat behavior at large V
bounds_dpl = [(0, 10), (-3, 3), (-3, 3), (-2, 5)]  # log_V0, gamma, beta, log_V_cut
bounds_dpl_constrained = [(0, 10), (-3, 0), (-1, 0), (-2, 5)]  # Force gamma <= 0, beta <= 0

initial_guesses_full = [
    [2.0, -0.5, -0.1, 0.0],
    [2.0, -0.6, -0.2, 0.5],
    [3.0, -0.8, -0.05, 0.5],
    [1.5, -0.5, -0.05, -0.5],
    [2.5, -0.7, -0.1, 0.0],
    [1.0, -0.4, -0.08, -1.0],
    [np.log10(np.median(V_full)), -alpha_fit_full, -0.1, 0.0],
    [2.0, -0.55, -0.08, 0.0],  # Target: steep small, flat large
]

# Use constrained bounds to force physically sensible behavior
nll_dpl_full, params_dpl_full = fit_dpl_multistart(V_full, initial_guesses_full, bounds_dpl_constrained)

if params_dpl_full is not None:
    log_V0_fit, gamma_fit, beta_fit, log_Vcut_fit = params_dpl_full
else:
    log_V0_fit, gamma_fit, beta_fit, log_Vcut_fit = np.nan, np.nan, np.nan, np.nan
    nll_dpl_full = np.inf

# --- Model 5: DPL with fixed slopes (γ=0 at small V, β=-4/9 at large V) ---
# slope_small = -gamma = 0 → gamma = 0
# slope_large = -beta = 4/9 → beta = -4/9
print("Fitting double power law (fixed slopes: 0 at small V, 4/9 at large V)...")
gamma_fixed = 0.0
beta_fixed = -4/9

bounds_2d = [(0, 10), (-2, 5)]  # log_V0, log_V_cut
initial_guesses_2d = [
    [2.0, 0.0],
    [3.0, 0.5],
    [1.5, -0.5],
    [2.5, 1.0],
    [4.0, 0.0],
]

nll_dpl_fixed_full, params_dpl_fixed_full = fit_dpl_fixed_slopes(
    V_full, gamma_fixed, beta_fixed, initial_guesses_2d, bounds_2d)

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
print(f"{'Power law (free α)':<40} {k_pl:>3} {nll_pl_full:>14.2f} {aic_pl:>14.2f} {bic_pl:>14.2f}")
print(f"{'Zipf (α=0)':<40} {k_zipf:>3} {nll_zipf_full:>14.2f} {aic_zipf:>14.2f} {bic_zipf:>14.2f}")
print(f"{'Power law (α=4/9 fixed)':<40} {k_49:>3} {nll_49_full:>14.2f} {aic_49:>14.2f} {bic_49:>14.2f}")
print(f"{'DPL + exp cutoff (free)':<40} {k_dpl:>3} {nll_dpl_full:>14.2f} {aic_dpl:>14.2f} {bic_dpl:>14.2f}")
print(f"{'DPL (slope=0 small, 4/9 large)':<40} {k_dpl_fixed:>3} {nll_dpl_fixed_full:>14.2f} {aic_dpl_fixed:>14.2f} {bic_dpl_fixed:>14.2f}")

print()
print(f"Best-fit power law: α = {alpha_fit_full:.4f}")
if not np.isnan(log_V0_fit):
    print(f"Best-fit DPL (free): V0=10^{log_V0_fit:.2f}, γ={gamma_fit:.3f}, β={beta_fit:.3f}, V_cut=10^{log_Vcut_fit:.2f}")
    print(f"  -> slope at V<<V0: {-gamma_fit:.3f}, slope at V>>V0: {-beta_fit:.3f}")
if not np.isnan(log_V0_fixed):
    print(f"DPL (fixed slopes): V0=10^{log_V0_fixed:.2f}, V_cut=10^{log_Vcut_fixed:.2f}")

# Delta AIC relative to best
best_aic = min(aic_pl, aic_zipf, aic_49, aic_dpl, aic_dpl_fixed)
print()
print("ΔAIC relative to best:")
print(f"  Power law (free α):              ΔAIC = {aic_pl - best_aic:+.1f}")
print(f"  Zipf (α=0):                      ΔAIC = {aic_zipf - best_aic:+.1f}")
print(f"  Power law (α=4/9):               ΔAIC = {aic_49 - best_aic:+.1f}")
print(f"  DPL (free):                      ΔAIC = {aic_dpl - best_aic:+.1f}")
print(f"  DPL (slope=0 small, 4/9 large):  ΔAIC = {aic_dpl_fixed - best_aic:+.1f}")

# ============================================================================
# ANALYSIS 2: Zipf vs Power law at increasing V_min cutoffs
# ============================================================================
print()
print("=" * 70)
print("ANALYSIS 2: ZIPF vs POWER LAW AT INCREASING V_min CUTOFFS")
print("=" * 70)
print(f"{'log10(V_min)':<12} {'N_clumps':>10} {'α_fit':>10} {'NLL_PL':>12} {'NLL_Zipf':>12} {'ΔAIC(Zipf-PL)':>15}")
print("-" * 75)

log_vmin_values = np.arange(0, 9, 1)  # 1 to 1e8 in 1 dex steps
results_cutoff = []

for log_vmin in log_vmin_values:
    V_min = 10**log_vmin
    V_sub = V_norm[V_norm > V_min]
    n_sub = len(V_sub)

    if n_sub < 20:
        continue

    # Fit power law
    res = minimize_scalar(lambda a: powerlaw_negloglik(a, V_sub), bounds=(-2, 2), method='bounded')
    alpha_sub = res.x
    nll_pl = res.fun

    # Zipf
    nll_zipf = powerlaw_negloglik(0, V_sub)

    # AIC comparison (k=1 for free power law, k=0 for Zipf)
    aic_pl = 2 * 1 + 2 * nll_pl
    aic_zipf = 2 * 0 + 2 * nll_zipf
    delta_aic = aic_zipf - aic_pl

    results_cutoff.append((log_vmin, n_sub, alpha_sub, nll_pl, nll_zipf, delta_aic))
    print(f"{log_vmin:<12.0f} {n_sub:>10} {alpha_sub:>10.4f} {nll_pl:>12.2f} {nll_zipf:>12.2f} {delta_aic:>+15.1f}")

print()
print("Interpretation: ΔAIC > 0 means power law is preferred; ΔAIC < 0 means Zipf is preferred")

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

# Power law (free)
res_pl3 = minimize_scalar(lambda a: powerlaw_negloglik(a, V_cut3), bounds=(-2, 2), method='bounded')
alpha_fit3 = res_pl3.x
nll_pl3 = res_pl3.fun

# Zipf
nll_zipf3 = powerlaw_negloglik(0, V_cut3)

# Power law α=4/9
nll_49_3 = powerlaw_negloglik(4/9, V_cut3)

# DPL free (use constrained bounds for physically sensible behavior)
print("Fitting DPL (free)...")
initial_guesses_3 = [
    [4.0, -0.1, -0.1, 2.0],
    [5.0, -0.2, -0.05, 2.5],
    [3.5, -0.05, -0.1, 1.5],
    [6.0, -0.1, -0.08, 2.0],
    [np.log10(np.median(V_cut3)), -alpha_fit3, -0.05, 2.0],
    [4.5, -0.15, -0.1, 2.0],
]
nll_dpl3, params_dpl3 = fit_dpl_multistart(V_cut3, initial_guesses_3, bounds_dpl_constrained)
if params_dpl3 is not None:
    log_V0_3, gamma_3, beta_3, log_Vcut_3 = params_dpl3
else:
    log_V0_3, gamma_3, beta_3, log_Vcut_3 = np.nan, np.nan, np.nan, np.nan
    nll_dpl3 = np.inf

# DPL fixed slopes
print("Fitting DPL (fixed slopes)...")
initial_guesses_3_2d = [
    [4.0, 2.0],
    [5.0, 2.5],
    [3.5, 1.5],
    [6.0, 3.0],
]
nll_dpl_fixed3, params_dpl_fixed3 = fit_dpl_fixed_slopes(
    V_cut3, gamma_fixed, beta_fixed, initial_guesses_3_2d, bounds_2d)
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
print(f"{'Power law (α={:.4f})':<40} {1:>3} {nll_pl3:>14.2f} {aic_pl3:>14.2f} {bic_pl3:>14.2f}".format(alpha_fit3))
print(f"{'Zipf (α=0)':<40} {0:>3} {nll_zipf3:>14.2f} {aic_zipf3:>14.2f} {bic_zipf3:>14.2f}")
print(f"{'Power law (α=4/9 fixed)':<40} {0:>3} {nll_49_3:>14.2f} {aic_49_3:>14.2f} {bic_49_3:>14.2f}")
print(f"{'DPL + exp cutoff (free)':<40} {4:>3} {nll_dpl3:>14.2f} {aic_dpl3:>14.2f} {bic_dpl3:>14.2f}")
print(f"{'DPL (slope=0 small, 4/9 large)':<40} {2:>3} {nll_dpl_fixed3:>14.2f} {aic_dpl_fixed3:>14.2f} {bic_dpl_fixed3:>14.2f}")

if not np.isnan(log_V0_3):
    print()
    print(f"DPL (free) params: V0=10^{log_V0_3:.2f}, γ={gamma_3:.3f}, β={beta_3:.3f}, V_cut=10^{log_Vcut_3:.2f}")
    print(f"  -> slope at V<<V0: {-gamma_3:.3f}, slope at V>>V0: {-beta_3:.3f}")
if not np.isnan(log_V0_fixed3):
    print(f"DPL (fixed) params: V0=10^{log_V0_fixed3:.2f}, V_cut=10^{log_Vcut_fixed3:.2f}")

best_aic3 = min(aic_pl3, aic_zipf3, aic_49_3, aic_dpl3, aic_dpl_fixed3)
print()
print("ΔAIC relative to best:")
print(f"  Power law (free α):              ΔAIC = {aic_pl3 - best_aic3:+.1f}")
print(f"  Zipf (α=0):                      ΔAIC = {aic_zipf3 - best_aic3:+.1f}")
print(f"  Power law (α=4/9):               ΔAIC = {aic_49_3 - best_aic3:+.1f}")
print(f"  DPL (free):                      ΔAIC = {aic_dpl3 - best_aic3:+.1f}")
print(f"  DPL (slope=0 small, 4/9 large):  ΔAIC = {aic_dpl_fixed3 - best_aic3:+.1f}")

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
x_plot = np.log10(V_plot_data)
x_min_plot, x_max_plot = x_plot.min(), x_plot.max()

# Histogram
n_bins = 60
hist, bin_edges = np.histogram(x_plot, bins=n_bins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
V_centers = 10**bin_centers
V_dN_dlogV = hist * V_centers

mask_plot = hist > 0
V_pts = V_centers[mask_plot]
y_pts = V_dN_dlogV[mask_plot]

ax.scatter(V_pts, y_pts, s=40, c='black', zorder=5, label='Data')

# For reference lines, anchor at a specific point in the data
# Use the point at V ~ 10^5 as anchor
anchor_idx = np.argmin(np.abs(np.log10(V_pts) - 5))
V_anchor = V_pts[anchor_idx]
y_anchor = y_pts[anchor_idx]

V_model = np.logspace(x_min_plot, x_max_plot, 300)

# Power law (best fit) - anchored
y_pl = y_anchor * (V_model / V_anchor)**alpha_fit_full
ax.plot(V_model, y_pl, 'b-', lw=2, label=f'Power law (α={alpha_fit_full:.3f})')

# Zipf (α=0) - anchored
y_zipf = y_anchor * (V_model / V_anchor)**0
ax.plot(V_model, y_zipf, 'g:', lw=2, label='Zipf (α=0)')

# Power law α=4/9 - anchored
y_49 = y_anchor * (V_model / V_anchor)**(4/9)
ax.plot(V_model, y_49, 'c--', lw=2, label='Power law (α=4/9)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$V$ [$\Delta x^3$]', fontsize=12)
ax.set_ylabel(r'$V \times dN/d\log V$', fontsize=12)
ax.set_title('Full range (V > 1)', fontsize=12)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

# --- Right panel: V > 10^3 ---
ax = axes[1]

x_plot3 = np.log10(V_cut3)
x_min3, x_max3 = x_plot3.min(), x_plot3.max()

hist3, bin_edges3 = np.histogram(x_plot3, bins=50)
bin_centers3 = 0.5 * (bin_edges3[:-1] + bin_edges3[1:])
V_centers3 = 10**bin_centers3
V_dN_dlogV3 = hist3 * V_centers3

mask3 = hist3 > 0
V_pts3 = V_centers3[mask3]
y_pts3 = V_dN_dlogV3[mask3]

ax.scatter(V_pts3, y_pts3, s=40, c='black', zorder=5, label='Data')

# Anchor at V ~ 10^5
anchor_idx3 = np.argmin(np.abs(np.log10(V_pts3) - 5))
V_anchor3 = V_pts3[anchor_idx3]
y_anchor3 = y_pts3[anchor_idx3]

V_model3 = np.logspace(x_min3, x_max3, 300)

# Power law (best fit) - anchored
y_pl3 = y_anchor3 * (V_model3 / V_anchor3)**alpha_fit3
ax.plot(V_model3, y_pl3, 'b-', lw=2, label=f'Power law (α={alpha_fit3:.3f})')

# Zipf (α=0) - anchored
y_zipf3 = y_anchor3 * (V_model3 / V_anchor3)**0
ax.plot(V_model3, y_zipf3, 'g:', lw=2, label='Zipf (α=0)')

# Power law α=4/9 - anchored
y_49_3 = y_anchor3 * (V_model3 / V_anchor3)**(4/9)
ax.plot(V_model3, y_49_3, 'c--', lw=2, label='Power law (α=4/9)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$V$ [$\Delta x^3$]', fontsize=12)
ax.set_ylabel(r'$V \times dN/d\log V$', fontsize=12)
ax.set_title(r'V > $10^3$ $\Delta x^3$', fontsize=12)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

plt.suptitle('Clump Volume Distribution: n10240, step 37, conn6, T=0.02', fontsize=14, y=1.02)
plt.tight_layout()

plot_path = "/lustre/orion/ast207/proj-shared/mpturb/clump_find_drummond/clump_out/n10240_sweep/conn6_T0p02_final_step00037/volume_dist_model_comparison.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {plot_path}")
plt.close()

# ============================================================================
# Additional plot: ΔAIC vs V_min
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

log_vmins = [r[0] for r in results_cutoff]
delta_aics = [r[5] for r in results_cutoff]

ax.plot(log_vmins, delta_aics, 'ko-', lw=2, markersize=8)
ax.axhline(0, color='gray', ls='--', lw=1)
ax.axhline(2, color='red', ls=':', lw=1, alpha=0.7)
ax.axhline(-2, color='red', ls=':', lw=1, alpha=0.7)

ax.fill_between(log_vmins, -2, 2, color='gray', alpha=0.2, label='Essentially equivalent')
ax.fill_between(log_vmins, 2, max(delta_aics)+10, color='blue', alpha=0.1, label='Power law preferred')
ax.fill_between(log_vmins, min(delta_aics)-10, -2, color='green', alpha=0.1, label='Zipf preferred')

ax.set_xlabel(r'$\log_{10}(V_{\rm min} / \Delta x^3)$', fontsize=12)
ax.set_ylabel(r'$\Delta$AIC (Zipf $-$ Power law)', fontsize=12)
ax.set_title('Model preference vs. minimum volume cutoff', fontsize=12)
ax.legend(loc='best', fontsize=10)
ax.set_xlim(0, 8)
ax.grid(True, alpha=0.3)

plot_path2 = "/lustre/orion/ast207/proj-shared/mpturb/clump_find_drummond/clump_out/n10240_sweep/conn6_T0p02_final_step00037/aic_vs_vmin.png"
plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {plot_path2}")
plt.close()

print()
print("Done!")
