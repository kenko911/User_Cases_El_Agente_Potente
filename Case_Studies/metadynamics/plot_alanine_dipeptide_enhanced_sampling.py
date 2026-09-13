#!/usr/bin/env python3
"""Generate plots for alanine-dipeptide Orb-OMOL enhanced-sampling results."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE = Path('/scratch/kenko/akg4pyscf/.graphchat-workspaces/kenko_room')
SUMMARY_JSON = WORKSPACE / 'alanine_dipeptide_orb_omol_enhanced_sampling_summary.json'
PLOT_DIR = WORKSPACE / 'alanine_dipeptide_orb_omol_2d_es_plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

summary = json.loads(SUMMARY_JSON.read_text())
fes_path = Path(summary['outputs'][0]['enhanced_sampling']['free_energy_surfaces'][0]['artifact_path'])
df = pd.read_csv(fes_path)

# Build 2D grids. CSV ordering is arbitrary-safe via pivot.
fe_grid = df.pivot(index='psi_degree', columns='phi_degree', values='free_energy_kj_mol').sort_index().sort_index(axis=1)
samp_grid = df.pivot(index='psi_degree', columns='phi_degree', values='samples').sort_index().sort_index(axis=1)
phi_vals = fe_grid.columns.to_numpy(dtype=float)
psi_vals = fe_grid.index.to_numpy(dtype=float)
PHI, PSI = np.meshgrid(phi_vals, psi_vals)
FE = fe_grid.to_numpy(dtype=float)
SAMPLES = samp_grid.to_numpy(dtype=float)

# Report minima and basic statistics.
min_idx = np.nanargmin(FE)
min_psi_i, min_phi_i = np.unravel_index(min_idx, FE.shape)
min_phi = phi_vals[min_phi_i]
min_psi = psi_vals[min_psi_i]
min_fe = FE[min_psi_i, min_phi_i] if False else FE[min_psi_i if False else min_psi_i, min_phi_i]
# correct min from unravel values
min_fe = FE[min_psi_i, min_phi_i]

# 1) Sampling distribution heatmap with sqrt color normalization via transformed counts.
sqrt_samples = np.sqrt(SAMPLES)
fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
pm = ax.pcolormesh(phi_vals, psi_vals, sqrt_samples, shading='auto', cmap='magma')
cbar = fig.colorbar(pm, ax=ax)
cbar.set_label(r'$\sqrt{\mathrm{samples\ per\ bin}}$')
ax.contour(PHI, PSI, SAMPLES, levels=[1, 10, 50, 100, 150, 200], colors='white', linewidths=0.45, alpha=0.65)
ax.set_title('Alanine dipeptide sampling distribution\nOrb-OMOL, 2D metadynamics')
ax.set_xlabel(r'$\phi$ / degrees')
ax.set_ylabel(r'$\psi$ / degrees')
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_aspect('equal', adjustable='box')
ax.grid(alpha=0.15)
sampling_png = PLOT_DIR / 'sampling_distribution_phi_psi.png'
fig.savefig(sampling_png, dpi=220)
plt.close(fig)

# 2) Free-energy surface heatmap/contours. Clip high FE for visual readability but keep colorbar label explicit.
clip_max = 20.0
FE_plot = np.clip(FE, 0.0, clip_max)
levels = np.arange(0.0, clip_max + 0.001, 2.0)
fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
pm = ax.pcolormesh(phi_vals, psi_vals, FE_plot, shading='auto', cmap='viridis_r', vmin=0.0, vmax=clip_max)
cs = ax.contour(PHI, PSI, FE, levels=levels, colors='k', linewidths=0.45, alpha=0.55)
ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f')
ax.plot(min_phi, min_psi, marker='*', color='red', markersize=12, markeredgecolor='white', markeredgewidth=0.7,
        label=f'Min: ({min_phi:.1f}°, {min_psi:.1f}°)')
cbar = fig.colorbar(pm, ax=ax)
cbar.set_label('Free energy / kJ mol$^{-1}$')
ax.set_title('Alanine dipeptide free-energy surface\nOrb-OMOL, 2D metadynamics')
ax.set_xlabel(r'$\phi$ / degrees')
ax.set_ylabel(r'$\psi$ / degrees')
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='upper right', frameon=True)
ax.grid(alpha=0.15)
fes_png = PLOT_DIR / 'free_energy_surface_phi_psi.png'
fig.savefig(fes_png, dpi=220)
plt.close(fig)

# 3) Combined side-by-side figure.
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), constrained_layout=True)
ax = axes[0]
pm0 = ax.pcolormesh(phi_vals, psi_vals, sqrt_samples, shading='auto', cmap='magma')
fig.colorbar(pm0, ax=ax, label=r'$\sqrt{\mathrm{samples\ per\ bin}}$')
ax.set_title('Sampling distribution')
ax.set_xlabel(r'$\phi$ / degrees')
ax.set_ylabel(r'$\psi$ / degrees')
ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.set_aspect('equal', adjustable='box')
ax.grid(alpha=0.15)

ax = axes[1]
pm1 = ax.pcolormesh(phi_vals, psi_vals, FE_plot, shading='auto', cmap='viridis_r', vmin=0.0, vmax=clip_max)
ax.contour(PHI, PSI, FE, levels=levels, colors='k', linewidths=0.35, alpha=0.45)
ax.plot(min_phi, min_psi, marker='*', color='red', markersize=11, markeredgecolor='white', markeredgewidth=0.7)
fig.colorbar(pm1, ax=ax, label='Free energy / kJ mol$^{-1}$')
ax.set_title('Free-energy surface')
ax.set_xlabel(r'$\phi$ / degrees')
ax.set_ylabel(r'$\psi$ / degrees')
ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.set_aspect('equal', adjustable='box')
ax.grid(alpha=0.15)
combined_png = PLOT_DIR / 'sampling_and_fes_phi_psi.png'
fig.savefig(combined_png, dpi=220)
plt.close(fig)

plot_summary = {
    'input_free_energy_csv': str(fes_path),
    'plot_directory': str(PLOT_DIR),
    'sampling_distribution_png': str(sampling_png),
    'free_energy_surface_png': str(fes_png),
    'combined_png': str(combined_png),
    'free_energy_minimum': {
        'phi_degree': float(min_phi),
        'psi_degree': float(min_psi),
        'free_energy_kj_mol': float(min_fe),
        'samples': float(SAMPLES[min_psi_i, min_phi_i]),
    },
    'visited_bins': int((SAMPLES > 0).sum()),
    'total_bins': int(SAMPLES.size),
    'total_samples': int(SAMPLES.sum()),
}
plot_summary_path = PLOT_DIR / 'plot_summary.json'
plot_summary_path.write_text(json.dumps(plot_summary, indent=2))

print(json.dumps(plot_summary, indent=2))