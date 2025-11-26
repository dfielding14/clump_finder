#!/usr/bin/env python
"""
Fit and compare models for clump volume distribution using least squares
on binned V × dN/dlogV data, combining multiple snapshots.

Models compared:
  (a) DPL - all parameters free (6 params)
  (b) DPL with gamma=0, beta=-4/9 fixed, m free (4 params)
  (c) Single power law with slope=4/9 fixed (1 param)
  (d) Single power law with free slope (2 params)
  (e) Zipf's law (slope=0) (1 param)
  (f) Beuermann with gamma=0, beta=-4/9 fixed, s free (5 params)
  (g) Beuermann - all parameters free (7 params)

The Beuermann function adds a sharpness parameter 's' to control the
transition width between power-law regimes:
  f(V) ~ (V/V0)^(-γ) × (1 + (V/V0)^s)^((γ-β)/s) × exp(-(V_cut/V)^m)
  - s=1: standard DPL (~1 decade transition)
  - s>1: sharper transition
  - s<1: broader transition

Snapshots are normalized by total volume before averaging.

Usage:
  python fit_dpl_leastsq.py [sweep_name]

  sweep_name: n320_sweep, n640_sweep, n1280_sweep, n2560_sweep, n5120_sweep, or n10240_sweep
              Defaults to n10240_sweep if not specified.
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import glob
import re
import sys
import os

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

# Extract resolution from sweep name (e.g., n10240_sweep -> 10240)
match = re.match(r'n(\d+)_sweep', sweep_name)
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

# Find all snapshot directories
all_dirs = sorted(os.listdir(base_dir))
snapshot_dirs = []

for d in all_dirs:
    # Match patterns like conn6_T0p02_step00035 or conn6_T0p02_final_step00035
    step_match = re.search(r'step(\d+)', d)
    if step_match:
        step_num = int(step_match.group(1))
        if step_num > min_step:
            # Check if it has a stitched file
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
# Model functions
# ============================================================================

def dpl_exp_cut(V, A, log_V0, gamma, beta, log_V_cut, m):
    """Double power law with small-V exponential cutoff."""
    V = np.asarray(V, dtype=float)
    V0 = 10**log_V0
    V_cut = 10**log_V_cut
    eps = np.finfo(float).tiny
    core = A * np.power(V/V0, -gamma) * np.power(1.0 + V/V0, (gamma - beta))
    cut = np.exp(-np.power(V_cut/np.clip(V, eps, None), m))
    return core * cut


def dpl_fixed(V, A, log_V0, log_V_cut, m):
    """DPL with gamma=0, beta=-4/9 fixed, m free."""
    return dpl_exp_cut(V, A, log_V0, 0.0, -4.0/9.0, log_V_cut, m)


def power_law(V, A, alpha):
    """Single power law."""
    return A * np.power(V, alpha)


def zipf_law(V, A):
    """Zipf's law (slope=0)."""
    return A * np.ones_like(V)


def beuermann_exp_cut(V, A, log_V0, gamma, beta, s, log_V_cut, m):
    """
    Beuermann (smoothly broken power law) with small-V exponential cutoff.

    The sharpness parameter 's' controls the transition width:
      - s=1: standard DPL (~1 decade transition)
      - s>1: sharper transition
      - s<1: broader transition
    """
    V = np.asarray(V, dtype=float)
    V0 = 10**log_V0
    V_cut = 10**log_V_cut
    eps = np.finfo(float).tiny

    # Beuermann form: (V/V0)^(-gamma) * (1 + (V/V0)^s)^((gamma-beta)/s)
    ratio = V / V0
    # Clip ratio^s to avoid overflow
    ratio_s = np.clip(np.power(ratio, s), 0, 1e100)
    core = A * np.power(ratio, -gamma) * np.power(1.0 + ratio_s, (gamma - beta) / s)
    cut = np.exp(-np.power(V_cut / np.clip(V, eps, None), m))
    return core * cut


def beuermann_fixed(V, A, log_V0, s, log_V_cut, m):
    """Beuermann with gamma=0, beta=-4/9 fixed, s free."""
    return beuermann_exp_cut(V, A, log_V0, 0.0, -4.0/9.0, s, log_V_cut, m)


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
    vols = data['volume'] / dx3  # Convert to dx^3 units
    total_vol = vols.sum() * dx3  # Total volume in box units

    all_volumes.append(vols)
    snapshot_info.append({
        'name': snap_dir,
        'n_clumps': len(vols),
        'total_vol': total_vol,
        'vols': vols,
    })

    print(f"  {snap_dir}: {len(vols):,} clumps, total_vol={total_vol:.4e}")

print(f"\nTotal snapshots loaded: {len(snapshot_info)}")


# ============================================================================
# Create common bins and compute normalized histograms
# ============================================================================

print()
print("=" * 70)
print("COMPUTING NORMALIZED HISTOGRAMS")
print("=" * 70)

# Find global V range
V_min_global = min(v.min() for v in all_volumes)
V_max_global = max(v.max() for v in all_volumes)
print(f"Global V range: [{V_min_global:.1e}, {V_max_global:.1e}] dx^3")

# Common bin edges
n_bins = 60
bin_edges = np.linspace(np.log10(1.0), np.log10(V_max_global * 1.1), n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
V_centers = 10**bin_centers

# First pass: compute raw histograms to find a good anchor point
raw_histograms = []
for info in snapshot_info:
    vols = info['vols']
    vols_cut = vols[vols > 1]
    hist, _ = np.histogram(np.log10(vols_cut), bins=bin_edges)
    y = hist * V_centers  # V × dN/dlogV
    raw_histograms.append(y)

# Find anchor point: prefer V ~ 10^5, but ensure all snapshots have data there
# Use the bin closest to 10^5 where all snapshots have non-zero counts
target_log_V = min(5.0, np.log10(V_max_global) - 0.5)  # Don't go beyond data
y_stack_raw = np.array(raw_histograms)
valid_anchor_mask = np.all(y_stack_raw > 0, axis=0)

if not valid_anchor_mask.any():
    # Fallback: use bin where at least one snapshot has data
    valid_anchor_mask = np.any(y_stack_raw > 0, axis=0)

valid_indices = np.where(valid_anchor_mask)[0]
idx_anchor = valid_indices[np.argmin(np.abs(bin_centers[valid_indices] - target_log_V))]
print(f"Anchor point: V = 10^{bin_centers[idx_anchor]:.2f} dx^3")

# Second pass: normalize using the anchor point
y_normalized_list = []
for y in raw_histograms:
    y_norm = y / y[idx_anchor]
    y_normalized_list.append(y_norm)

# Stack and compute mean/std
y_stack = np.array(y_normalized_list)
y_mean = np.mean(y_stack, axis=0)
y_std = np.std(y_stack, axis=0)

# Mask bins with data (non-zero in at least one snapshot)
mask_any = np.any(y_stack > 0, axis=0)
# For fitting, use bins where ALL snapshots have data
mask_all = np.all(y_stack > 0, axis=0)

print(f"Bins with any data: {mask_any.sum()}")
print(f"Bins with data in ALL snapshots: {mask_all.sum()}")


# ============================================================================
# Fit models to averaged data
# ============================================================================

V_fit = V_centers[mask_all]
y_fit = y_mean[mask_all]
y_err = y_std[mask_all]

print()
print("=" * 70)
print("FITTING MODELS TO AVERAGED DATA")
print("=" * 70)

# Initial guesses
A0 = np.median(y_fit)
log_V0_0 = np.log10(np.median(V_fit))
log_Vcut0 = max(1.0, np.log10(V_fit.min()))  # V_cut >= 10

# --- Model (a): DPL free ---
print("\nMODEL (a): DPL - all parameters free (6 params)")

def log_dpl_full(V, A, log_V0, gamma, beta, log_V_cut, m):
    model = dpl_exp_cut(V, A, log_V0, gamma, beta, log_V_cut, m)
    return np.log10(np.clip(model, 1e-30, None))

p0_a = [A0, log_V0_0, -0.5, -0.4, log_Vcut0, 2.0]
bounds_a = ([0, 0, -3, -3, 1, 0.1], [np.inf, 12, 3, -0.25, 10, 10])  # V_cut >= 10, beta <= -0.25

popt_a, _ = curve_fit(log_dpl_full, V_fit, np.log10(y_fit), p0=p0_a, bounds=bounds_a, maxfev=10000)
A_a, log_V0_a, gamma_a, beta_a, log_Vcut_a, m_a = popt_a
print(f"  A={A_a:.2e}, V0=10^{log_V0_a:.2f}, γ={gamma_a:.3f}, β={beta_a:.3f}, V_cut=10^{log_Vcut_a:.2f}, m={m_a:.3f}")

# --- Model (b): DPL fixed slopes ---
print("\nMODEL (b): DPL γ=0, β=-4/9 fixed, m free (4 params)")

def log_dpl_fixed(V, A, log_V0, log_V_cut, m):
    model = dpl_fixed(V, A, log_V0, log_V_cut, m)
    return np.log10(np.clip(model, 1e-30, None))

p0_b = [A0, log_V0_0, log_Vcut0, 2.0]
bounds_b = ([0, 0, 1, 0.1], [np.inf, 12, 10, 10])  # V_cut >= 10

popt_b, _ = curve_fit(log_dpl_fixed, V_fit, np.log10(y_fit), p0=p0_b, bounds=bounds_b, maxfev=10000)
A_b, log_V0_b, log_Vcut_b, m_b = popt_b
print(f"  A={A_b:.2e}, V0=10^{log_V0_b:.2f}, V_cut=10^{log_Vcut_b:.2f}, m={m_b:.3f}")

# --- Model (c): Power law slope=4/9 ---
print("\nMODEL (c): Power law slope=4/9 fixed (1 param)")

def log_pl_49(V, A):
    model = power_law(V, A, 4.0/9.0)
    return np.log10(np.clip(model, 1e-30, None))

popt_c, _ = curve_fit(log_pl_49, V_fit, np.log10(y_fit), p0=[1.0], bounds=([0], [np.inf]), maxfev=10000)
A_c = popt_c[0]
print(f"  A={A_c:.2e}")

# --- Model (d): Power law free slope ---
print("\nMODEL (d): Power law free slope (2 params)")

def log_pl_free(V, A, alpha):
    model = power_law(V, A, alpha)
    return np.log10(np.clip(model, 1e-30, None))

popt_d, _ = curve_fit(log_pl_free, V_fit, np.log10(y_fit), p0=[1.0, 0.5], bounds=([0, -2], [np.inf, 2]), maxfev=10000)
A_d, alpha_d = popt_d
print(f"  A={A_d:.2e}, slope={alpha_d:.4f}")

# --- Model (e): Zipf ---
print("\nMODEL (e): Zipf's law slope=0 (1 param)")

def log_zipf(V, A):
    model = zipf_law(V, A)
    return np.log10(np.clip(model, 1e-30, None))

popt_e, _ = curve_fit(log_zipf, V_fit, np.log10(y_fit), p0=[A0], bounds=([0], [np.inf]), maxfev=10000)
A_e = popt_e[0]
print(f"  A={A_e:.2e}")

# --- Model (f): Beuermann fixed slopes ---
print("\nMODEL (f): Beuermann γ=0, β=-4/9 fixed, s free (5 params)")

def log_beuermann_fixed(V, A, log_V0, s, log_V_cut, m):
    model = beuermann_fixed(V, A, log_V0, s, log_V_cut, m)
    return np.log10(np.clip(model, 1e-30, None))

p0_f = [A0, log_V0_0, 1.0, log_Vcut0, 2.0]
bounds_f = ([0, 0, 0.1, 1, 0.1], [np.inf, 12, 50, 10, 10])  # V_cut >= 10

popt_f, _ = curve_fit(log_beuermann_fixed, V_fit, np.log10(y_fit), p0=p0_f, bounds=bounds_f, maxfev=10000)
A_f, log_V0_f, s_f, log_Vcut_f, m_f = popt_f
print(f"  A={A_f:.2e}, V0=10^{log_V0_f:.2f}, s={s_f:.3f}, V_cut=10^{log_Vcut_f:.2f}, m={m_f:.3f}")

# --- Model (g): Beuermann free ---
print("\nMODEL (g): Beuermann - all parameters free (7 params)")

def log_beuermann_full(V, A, log_V0, gamma, beta, s, log_V_cut, m):
    model = beuermann_exp_cut(V, A, log_V0, gamma, beta, s, log_V_cut, m)
    return np.log10(np.clip(model, 1e-30, None))

p0_g = [A0, log_V0_0, -0.5, -0.4, 1.0, log_Vcut0, 2.0]
bounds_g = ([0, 0, -3, -3, 0.1, 1, 0.1], [np.inf, 12, 3, -0.25, 50, 10, 10])  # V_cut >= 10, beta <= -0.25

popt_g, _ = curve_fit(log_beuermann_full, V_fit, np.log10(y_fit), p0=p0_g, bounds=bounds_g, maxfev=10000)
A_g, log_V0_g, gamma_g, beta_g, s_g, log_Vcut_g, m_g = popt_g
print(f"  A={A_g:.2e}, V0=10^{log_V0_g:.2f}, γ={gamma_g:.3f}, β={beta_g:.3f}, s={s_g:.3f}, V_cut=10^{log_Vcut_g:.2f}, m={m_g:.3f}")


# ============================================================================
# Compute AIC for each model over three V ranges
# ============================================================================

def compute_aic(V_pts, y_pts, model_func, params, k):
    """AIC = n * ln(RSS/n) + 2k"""
    log_y = np.log10(y_pts)
    log_model = model_func(V_pts, *params)
    residuals = log_y - log_model
    rss = np.sum(residuals**2)
    n = len(y_pts)
    aic = n * np.log(rss / n) + 2 * k
    return aic, rss, n


def bin_averaged_data(V_min):
    """Get averaged data for V > V_min."""
    mask = (V_centers > V_min) & mask_all
    return V_centers[mask], y_mean[mask], y_std[mask], mask.sum()


models = [
    ("(a) DPL free", log_dpl_full, popt_a, 6),
    ("(b) DPL fixed", log_dpl_fixed, popt_b, 4),
    ("(c) PL 4/9", log_pl_49, popt_c, 1),
    ("(d) PL free", log_pl_free, popt_d, 2),
    ("(e) Zipf", log_zipf, popt_e, 1),
    ("(f) Beuermann fix", log_beuermann_fixed, popt_f, 5),
    ("(g) Beuermann free", log_beuermann_full, popt_g, 7),
]

V_cuts = [1, 1e3, 1e6]
V_cut_labels = ["V > 1", "V > 10^3", "V > 10^6"]

print()
print("=" * 70)
print("AIC COMPARISON ACROSS VOLUME RANGES")
print(f"(Combined from {len(snapshot_info)} snapshots)")
print("=" * 70)

aic_results = {label: {} for label in V_cut_labels}

for V_min, label in zip(V_cuts, V_cut_labels):
    V_pts, y_pts, y_err_pts, n_pts = bin_averaged_data(V_min)

    if n_pts == 0:
        print(f"\n{label}: No data")
        continue

    print(f"\n{label} ({n_pts} bins):")
    print("-" * 50)

    aic_values = []
    for name, model_func, params, k in models:
        aic, rss, n = compute_aic(V_pts, y_pts, model_func, params, k)
        aic_values.append((name, aic, rss, k))
        aic_results[label][name] = aic

    min_aic = min(av[1] for av in aic_values)

    print(f"  {'Model':<18} {'AIC':>10} {'ΔAIC':>10} {'RSS':>10} {'k':>5}")
    for name, aic, rss, k in sorted(aic_values, key=lambda x: x[1]):
        delta_aic = aic - min_aic
        print(f"  {name:<18} {aic:>10.2f} {delta_aic:>10.2f} {rss:>10.4f} {k:>5}")


print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("\nΔAIC interpretation:")
print("  0-2:   Substantial support")
print("  2-4:   Some support")
print("  4-7:   Considerably less support")
print("  >10:   Essentially no support")


# ============================================================================
# Plot
# ============================================================================

print()
print("Generating plot...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

colors = {
    "(a) DPL free": 'red',
    "(b) DPL fixed": 'blue',
    "(c) PL 4/9": 'green',
    "(d) PL free": 'magenta',
    "(e) Zipf": 'orange',
    "(f) Beuermann fix": 'cyan',
    "(g) Beuermann free": 'brown',
}

linestyles = {
    "(a) DPL free": '-',
    "(b) DPL fixed": '--',
    "(c) PL 4/9": ':',
    "(d) PL free": '-.',
    "(e) Zipf": ':',
    "(f) Beuermann fix": '--',
    "(g) Beuermann free": '-',
}

for ax, (V_min, label) in zip(axes, zip(V_cuts, V_cut_labels)):
    V_pts, y_pts, y_err_pts, n_pts = bin_averaged_data(V_min)

    if n_pts == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        continue

    # Data with error bars
    ax.errorbar(V_pts, y_pts, yerr=y_err_pts, fmt='ko', ms=5, capsize=2,
                elinewidth=1, zorder=5, label='Data (mean ± std)')

    # Model curves
    V_model = np.logspace(np.log10(V_pts.min()), np.log10(V_pts.max()), 200)

    aic_dict = aic_results[label]
    min_aic = min(aic_dict.values()) if aic_dict else 0

    y_a = dpl_exp_cut(V_model, *popt_a)
    y_b = dpl_fixed(V_model, *popt_b)
    y_c = power_law(V_model, A_c, 4.0/9.0)
    y_d = power_law(V_model, A_d, alpha_d)
    y_e = zipf_law(V_model, A_e)
    y_f = beuermann_fixed(V_model, *popt_f)
    y_g = beuermann_exp_cut(V_model, *popt_g)

    labels_dict = {
        "(a) DPL free": f"(a) DPL: γ={gamma_a:.2f}, β={beta_a:.2f}, m={m_a:.2f}",
        "(b) DPL fixed": f"(b) DPL: γ=0, β=-4/9, m={m_b:.2f}",
        "(c) PL 4/9": f"(c) PL: slope=4/9",
        "(d) PL free": f"(d) PL: slope={alpha_d:.3f}",
        "(e) Zipf": f"(e) Zipf: slope=0",
        "(f) Beuermann fix": f"(f) Beuermann: s={s_f:.2f}",
        "(g) Beuermann free": f"(g) Beuermann: γ={gamma_g:.2f}, β={beta_g:.2f}, s={s_g:.2f}",
    }

    model_curves = [
        ("(a) DPL free", y_a),
        ("(b) DPL fixed", y_b),
        ("(c) PL 4/9", y_c),
        ("(d) PL free", y_d),
        ("(e) Zipf", y_e),
        ("(f) Beuermann fix", y_f),
        ("(g) Beuermann free", y_g),
    ]

    for name, y_model in model_curves:
        delta_aic = aic_dict.get(name, 0) - min_aic
        lw = 2.5 if delta_aic < 2 else 1.5
        alpha = 1.0 if delta_aic < 4 else 0.5
        ax.plot(V_model, y_model, color=colors[name], ls=linestyles[name],
                lw=lw, alpha=alpha, label=f'{labels_dict[name]} (Δ={delta_aic:.1f})')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$V$ [$\Delta x^3$]', fontsize=12)
    ax.set_ylabel(r'$V \times dN/d\log V$ (normalized)', fontsize=12)
    ax.set_title(f'{label} ({n_pts} bins)', fontsize=12)
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.suptitle(f'AIC Model Comparison: n{resolution}, conn6, T=0.02 (combined {len(snapshot_info)} snapshots)\n' +
             r'AIC = $n \ln(\mathrm{RSS}/n) + 2k$  ($n$ = # bins, RSS = residual sum of squares, $k$ = # params)' + '\n' +
             r'$\Delta$AIC = AIC $-$ AIC$_{\mathrm{best}}$  (lower $\Delta$ = better)',
             fontsize=11, y=1.08)
plt.tight_layout()

plot_path = f"{base_dir}/aic_model_comparison_combined.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {plot_path}")
plt.close()

# ============================================================================
# Save combined data to npz
# ============================================================================

# Collect raw (unnormalized) histograms for each snapshot
y_raw_list = []
for info in snapshot_info:
    vols = info['vols']
    vols_cut = vols[vols > 1]
    hist, _ = np.histogram(np.log10(vols_cut), bins=bin_edges)
    y = hist * V_centers  # V × dN/dlogV
    y_raw_list.append(y)

y_raw_stack = np.array(y_raw_list)

# Save to npz
npz_path = f"{base_dir}/combined_volume_distributions.npz"
np.savez(npz_path,
         # Bin information
         bin_edges=bin_edges,
         bin_centers=bin_centers,
         V_centers=V_centers,
         # Snapshot names
         snapshot_names=np.array([info['name'] for info in snapshot_info]),
         # Raw V × dN/dlogV for each snapshot (shape: n_snapshots × n_bins)
         y_raw=y_raw_stack,
         # Normalized V × dN/dlogV for each snapshot (shape: n_snapshots × n_bins)
         y_normalized=y_stack,
         # Combined normalized distribution (mean and std)
         y_mean=y_mean,
         y_std=y_std,
         # Snapshot metadata
         n_clumps=np.array([info['n_clumps'] for info in snapshot_info]),
         total_vol=np.array([info['total_vol'] for info in snapshot_info]),
         # Normalization info
         anchor_idx=idx_anchor,
         anchor_V=V_centers[idx_anchor],
         )
print(f"Saved combined data to: {npz_path}")

print()
print("Done!")
