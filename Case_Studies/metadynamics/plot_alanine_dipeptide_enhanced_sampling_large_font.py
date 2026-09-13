#!/usr/bin/env python3
"""Generate publication-style plots for alanine-dipeptide Orb-OMOL
enhanced-sampling results.

Changes relative to the original script:
- Larger font sizes throughout.
- Square individual figures.
- Square plotting panels in the combined figure.
- Higher-resolution PNG output.
- PDF output for publication.
- New filenames to preserve the original plots.
- No bold text.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

WORKSPACE = Path(
    "/scratch/kenko/repos/potente/User_Cases_El_Agente_Potente/metadynamics"
)

SUMMARY_JSON = (
    WORKSPACE
    / "alanine_dipeptide_orb_omol_enhanced_sampling_summary.json"
)

PLOT_DIR = (
    WORKSPACE
    / "alanine_dipeptide_orb_omol_2d_es_plots"
)

PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Global plotting style
# ============================================================

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 21,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.titlesize": 22,
    "axes.linewidth": 1.5,
})


# ============================================================
# Load data
# ============================================================

summary = json.loads(SUMMARY_JSON.read_text())

fes_path = Path(
    summary["outputs"][0]["enhanced_sampling"]
    ["free_energy_surfaces"][0]["artifact_path"]
)

df = pd.read_csv(fes_path)


# ============================================================
# Construct 2D grids
# ============================================================

fe_grid = (
    df.pivot(
        index="psi_degree",
        columns="phi_degree",
        values="free_energy_kj_mol",
    )
    .sort_index()
    .sort_index(axis=1)
)

samp_grid = (
    df.pivot(
        index="psi_degree",
        columns="phi_degree",
        values="samples",
    )
    .sort_index()
    .sort_index(axis=1)
)

phi_vals = fe_grid.columns.to_numpy(dtype=float)
psi_vals = fe_grid.index.to_numpy(dtype=float)

PHI, PSI = np.meshgrid(phi_vals, psi_vals)

FE = fe_grid.to_numpy(dtype=float)
SAMPLES = samp_grid.to_numpy(dtype=float)


# ============================================================
# Locate global free-energy minimum
# ============================================================

min_idx = np.nanargmin(FE)
min_psi_i, min_phi_i = np.unravel_index(min_idx, FE.shape)

min_phi = phi_vals[min_phi_i]
min_psi = psi_vals[min_psi_i]
min_fe = FE[min_psi_i, min_phi_i]


# ============================================================
# Common plot settings
# ============================================================

xlim = (-180, 180)
ylim = (-180, 180)

ticks = np.arange(-180, 181, 60)

# Clip high free energies for visualization
clip_max = 20.0
FE_plot = np.clip(FE, 0.0, clip_max)

# Free-energy contour levels
levels = np.arange(
    0.0,
    clip_max + 0.001,
    2.0,
)

# Sampling visualization
sqrt_samples = np.sqrt(SAMPLES)


# ============================================================
# 1. Sampling distribution
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.0, 8.0),
    constrained_layout=True,
)

pm = ax.pcolormesh(
    phi_vals,
    psi_vals,
    sqrt_samples,
    shading="auto",
    cmap="magma",
)

cbar = fig.colorbar(
    pm,
    ax=ax,
    pad=0.03,
    fraction=0.05,
)

cbar.set_label(
    r"$\sqrt{\mathrm{samples\ per\ bin}}$",
    fontsize=19,
    labelpad=14,
)

cbar.ax.tick_params(
    labelsize=15,
    width=1.3,
)

ax.contour(
    PHI,
    PSI,
    SAMPLES,
    levels=[1, 10, 50, 100, 150, 200],
    colors="white",
    linewidths=0.65,
    alpha=0.70,
)

ax.set_title(
    "Alanine dipeptide sampling distribution\n"
    "Orb-OMOL, 2D metadynamics",
    fontsize=21,
    pad=16,
)

ax.set_xlabel(
    r"$\phi$ / degrees",
    fontsize=22,
    labelpad=10,
)

ax.set_ylabel(
    r"$\psi$ / degrees",
    fontsize=22,
    labelpad=10,
)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)

ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.tick_params(
    axis="both",
    labelsize=16,
    width=1.4,
    length=6,
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.grid(
    alpha=0.15,
    linewidth=0.7,
)

sampling_png = (
    PLOT_DIR
    / "sampling_distribution_phi_psi_square_largefont_regular.png"
)

sampling_pdf = (
    PLOT_DIR
    / "sampling_distribution_phi_psi_square_largefont_regular.pdf"
)

fig.savefig(
    sampling_png,
    dpi=400,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    sampling_pdf,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# 2. Free-energy surface
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.0, 8.0),
    constrained_layout=True,
)

pm = ax.pcolormesh(
    phi_vals,
    psi_vals,
    FE_plot,
    shading="auto",
    cmap="viridis_r",
    vmin=0.0,
    vmax=clip_max,
)

cs = ax.contour(
    PHI,
    PSI,
    FE,
    levels=levels,
    colors="k",
    linewidths=0.60,
    alpha=0.55,
)

ax.clabel(
    cs,
    inline=True,
    fontsize=10,
    fmt="%.0f",
)

# Mark global free-energy minimum
ax.plot(
    min_phi,
    min_psi,
    marker="*",
    color="red",
    markersize=18,
    markeredgecolor="white",
    markeredgewidth=1.2,
    label=(
        f"Min: ({min_phi:.1f}°, "
        f"{min_psi:.1f}°)"
    ),
    zorder=10,
)

cbar = fig.colorbar(
    pm,
    ax=ax,
    pad=0.03,
    fraction=0.05,
)

cbar.set_label(
    r"Free energy / kJ mol$^{-1}$",
    fontsize=19,
    labelpad=14,
)

cbar.ax.tick_params(
    labelsize=15,
    width=1.3,
)

ax.set_title(
    "Alanine dipeptide free-energy surface\n"
    "Orb-OMOL, 2D metadynamics",
    fontsize=21,
    pad=16,
)

ax.set_xlabel(
    r"$\phi$ / degrees",
    fontsize=22,
    labelpad=10,
)

ax.set_ylabel(
    r"$\psi$ / degrees",
    fontsize=22,
    labelpad=10,
)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)

ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.tick_params(
    axis="both",
    labelsize=16,
    width=1.4,
    length=6,
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.legend(
    loc="upper right",
    fontsize=14,
    frameon=True,
)

ax.grid(
    alpha=0.15,
    linewidth=0.7,
)

fes_png = (
    PLOT_DIR
    / "free_energy_surface_phi_psi_square_largefont_regular.png"
)

fes_pdf = (
    PLOT_DIR
    / "free_energy_surface_phi_psi_square_largefont_regular.pdf"
)

fig.savefig(
    fes_png,
    dpi=400,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    fes_pdf,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# 3. Combined side-by-side figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15.5, 7.2),
    constrained_layout=True,
)


# ------------------------------------------------------------
# Left: sampling distribution
# ------------------------------------------------------------

ax = axes[0]

pm0 = ax.pcolormesh(
    phi_vals,
    psi_vals,
    sqrt_samples,
    shading="auto",
    cmap="magma",
)

cbar0 = fig.colorbar(
    pm0,
    ax=ax,
    pad=0.025,
    fraction=0.05,
)

cbar0.set_label(
    r"$\sqrt{\mathrm{samples\ per\ bin}}$",
    fontsize=17,
    labelpad=10,
)

cbar0.ax.tick_params(
    labelsize=14,
)

ax.contour(
    PHI,
    PSI,
    SAMPLES,
    levels=[1, 10, 50, 100, 150, 200],
    colors="white",
    linewidths=0.55,
    alpha=0.65,
)

ax.set_title(
    "Sampling distribution",
    fontsize=20,
    pad=12,
)

ax.set_xlabel(
    r"$\phi$ / degrees",
    fontsize=20,
)

ax.set_ylabel(
    r"$\psi$ / degrees",
    fontsize=20,
)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)

ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.tick_params(
    axis="both",
    labelsize=15,
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.grid(alpha=0.15)


# ------------------------------------------------------------
# Right: free-energy surface
# ------------------------------------------------------------

ax = axes[1]

pm1 = ax.pcolormesh(
    phi_vals,
    psi_vals,
    FE_plot,
    shading="auto",
    cmap="viridis_r",
    vmin=0.0,
    vmax=clip_max,
)

ax.contour(
    PHI,
    PSI,
    FE,
    levels=levels,
    colors="k",
    linewidths=0.45,
    alpha=0.45,
)

ax.plot(
    min_phi,
    min_psi,
    marker="*",
    color="red",
    markersize=16,
    markeredgecolor="white",
    markeredgewidth=1.0,
    zorder=10,
)

cbar1 = fig.colorbar(
    pm1,
    ax=ax,
    pad=0.025,
    fraction=0.05,
)

cbar1.set_label(
    r"Free energy / kJ mol$^{-1}$",
    fontsize=17,
    labelpad=10,
)

cbar1.ax.tick_params(
    labelsize=14,
)

ax.set_title(
    "Free-energy surface",
    fontsize=20,
    pad=12,
)

ax.set_xlabel(
    r"$\phi$ / degrees",
    fontsize=20,
)

ax.set_ylabel(
    r"$\psi$ / degrees",
    fontsize=20,
)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)

ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.tick_params(
    axis="both",
    labelsize=15,
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.grid(alpha=0.15)


combined_png = (
    PLOT_DIR
    / "sampling_and_fes_phi_psi_largefont_regular.png"
)

combined_pdf = (
    PLOT_DIR
    / "sampling_and_fes_phi_psi_largefont_regular.pdf"
)

fig.savefig(
    combined_png,
    dpi=400,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    combined_pdf,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# Save plot summary
# ============================================================

plot_summary = {
    "input_free_energy_csv": str(fes_path),
    "plot_directory": str(PLOT_DIR),

    "sampling_distribution_png": str(sampling_png),
    "sampling_distribution_pdf": str(sampling_pdf),

    "free_energy_surface_png": str(fes_png),
    "free_energy_surface_pdf": str(fes_pdf),

    "combined_png": str(combined_png),
    "combined_pdf": str(combined_pdf),

    "free_energy_minimum": {
        "phi_degree": float(min_phi),
        "psi_degree": float(min_psi),
        "free_energy_kj_mol": float(min_fe),
        "samples": float(
            SAMPLES[min_psi_i, min_phi_i]
        ),
    },

    "visited_bins": int(
        (SAMPLES > 0).sum()
    ),

    "total_bins": int(
        SAMPLES.size
    ),

    "total_samples": int(
        SAMPLES.sum()
    ),
}


plot_summary_path = (
    PLOT_DIR
    / "plot_summary_square_largefont_regular.json"
)

plot_summary_path.write_text(
    json.dumps(
        plot_summary,
        indent=2,
    )
)

print(
    json.dumps(
        plot_summary,
        indent=2,
    )
)
