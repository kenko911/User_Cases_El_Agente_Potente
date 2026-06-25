#!/usr/bin/env python3
"""
Benchmark MLFF/MLIP elasticity predictions for CIFs in sample_1000/cifs.

For each requested model (default: MatterSim, Orb-OMAT, MACE-OMAT), this script
runs the MLFF workflow on all CIF files in the input directory, extracts VRH bulk
and shear moduli, converts predictions from eV/Å^3 to GPa, writes one prediction
CSV per model, and generates comparison plots/metrics against elasticity_1000.csv.

Default paths are set for:
/scratch/kenko/repos/potente/User_Cases_El_Agente_Potente/User_Cases/High_Throughput/Elasticity/sample_1000/
"""

from __future__ import annotations

import os
# Avoid Orb/MatterSim/PyTorch interaction: FX tracing a dynamo-optimized function.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Required request instrumentation for MLFF/MLIP runs.
# Some deployments do not install every optional OpenTelemetry integration; keep
# the benchmark runnable while still enabling each integration when available.
import logfire
logfire.configure()
try:
    logfire.instrument_requests()
except Exception as exc:  # pragma: no cover - environment-dependent optional extra
    logfire.debug("Skipping logfire requests instrumentation: {exc}", exc=str(exc))
try:
    logfire.instrument_pydantic_ai()
except Exception as exc:  # pragma: no cover - environment-dependent optional extra
    logfire.debug("Skipping logfire pydantic-ai instrumentation: {exc}", exc=str(exc))

import argparse
import asyncio
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grafico.deps import GraficoDeps
from domains.mlffs.graph import mlffs_graph
from domains.mlffs.graph.mlffs_graph import (
    MLFFElasticityConfig,
    MLFFWorkflowRequest,
    RouterDecision,
)

EV_A3_TO_GPA = 160.21766208

DEFAULT_BASE_DIR = Path(
    "/scratch/kenko/repos/potente/User_Cases_El_Agente_Potente/User_Cases/"
    "High_Throughput/Elasticity/sample_1000"
)
DEFAULT_CIF_DIR = DEFAULT_BASE_DIR / "cifs"
DEFAULT_GT_CSV = DEFAULT_BASE_DIR / "elasticity_1000.csv"
DEFAULT_OUTDIR = Path.cwd() / "mlff_elasticity_benchmark"
DEFAULT_MODELS = ("MatterSim", "Orb-OMAT", "MACE-OMAT")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).replace("-", "_")


def build_ctx(workspace_dir: Path, tool_call_id: str):
    """Build the lightweight context expected by run_mlff_workflow(ctx, request)."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    deps = GraficoDeps(
        ws_url=os.getenv("GRAPHCHAT_AGENT_WS_URL")
        or os.getenv("VITE_WS_URL", "ws://graphchat:3000"),
        room=os.getenv("GRAPHCHAT_ROOM", "room"),
        sparql_endpoint=os.getenv(
            "SPARQL_ENDPOINT", "http://blazegraph:8080/blazegraph/namespace/kb/sparql"
        ),
        workspace_path_override=str(workspace_dir),
    )
    return SimpleNamespace(deps=deps, tool_call_id=tool_call_id)


def force_elasticity_router() -> None:
    """
    Force the workflow router to select only MLFFElasticity.

    This keeps the benchmark deterministic and avoids invoking unrelated property
    calculations for 1000 structures. The rest of the MLFF workflow remains intact:
    CIF loading -> relaxation -> elasticity -> persisted outputs.
    """

    async def _elasticity_only(state, structure):  # noqa: ANN001 - workflow callback signature
        return RouterDecision(
            next_nodes=[
                MLFFElasticityConfig(
                    reasoning="Benchmark requires only VRH bulk/shear elasticity.",
                    confidence=1.0,
                )
            ],
            reasoning="Forced elasticity-only routing for benchmark.",
            confidence=1.0,
        )

    mlffs_graph.mlff_dynamic_routing_multi = _elasticity_only


def maybe_make_limited_cif_dir(cif_dir: Path, limit: int | None, outdir: Path) -> Path:
    """Optionally create a temporary directory with symlinks to the first N CIFs."""
    if limit is None:
        return cif_dir
    paths = sorted(cif_dir.glob("*.cif"))[:limit]
    if not paths:
        raise RuntimeError(f"No CIF files found in {cif_dir}")
    tmp = outdir / f"cifs_first_{limit}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for p in paths:
        (tmp / p.name).symlink_to(p)
    return tmp


async def run_one_model(
    model_name: str,
    cif_dir: Path,
    outdir: Path,
    *,
    prefer_gpu: bool,
    relax_cell: bool,
    relax_fmax: float,
    relax_steps: int,
) -> pd.DataFrame:
    """Run one MLFF workflow for one model and return a normalized GPa DataFrame."""
    model_safe = safe_name(model_name)
    workspace_dir = outdir / "workflow_runs" / model_safe
    ctx = build_ctx(workspace_dir, tool_call_id=f"elasticity_{model_safe}")

    request = MLFFWorkflowRequest(
        summarised_user_query=(
            "High-throughput benchmark: compute only the elasticity tensor / VRH "
            "bulk modulus and VRH shear modulus for every input CIF. Do not run "
            "MD, phonons, DOS, adsorption, GCMC, Widom insertion, or NEB."
        ),
        identifier_type="from_path",
        identifier=str(cif_dir),
        model_name=model_name,
        prefer_gpu=prefer_gpu,
        relax_cell=relax_cell,
        relax_fmax=relax_fmax,
        relax_steps=relax_steps,
        update_graph=False,
    )

    logfire.info("Starting MLFF elasticity benchmark for {model}", model=model_name)
    outputs = await mlffs_graph.run_mlff_workflow(ctx, request)
    logfire.info("Finished MLFF elasticity benchmark for {model}", model=model_name)

    # Preferred extraction path: workflow-written property_summary.csv includes
    # source_identifier, which is the CIF stem set by the from_path loader.
    summary_paths = [getattr(o, "summary_csv_path", None) for o in outputs]
    summary_path = next((Path(p) for p in summary_paths if p), None)

    if summary_path and summary_path.exists():
        raw = pd.read_csv(summary_path)
        pred = normalize_prediction_summary(raw, model_name=model_name)
    else:
        # Fallback: preserve sorted CIF order if no summary CSV was persisted.
        mp_ids = [p.stem for p in sorted(cif_dir.glob("*.cif"))]
        rows = []
        for i, output in enumerate(outputs):
            elast = getattr(output, "elasticity", None)
            rows.append(
                {
                    "mp_id": mp_ids[i] if i < len(mp_ids) else f"unknown_{i}",
                    "bulk_modulus": getattr(elast, "bulk_modulus_vrh", np.nan) * EV_A3_TO_GPA
                    if elast is not None and getattr(elast, "bulk_modulus_vrh", None) is not None
                    else np.nan,
                    "shear_modulus": getattr(elast, "shear_modulus_vrh", np.nan) * EV_A3_TO_GPA
                    if elast is not None and getattr(elast, "shear_modulus_vrh", None) is not None
                    else np.nan,
                    "model": model_name,
                }
            )
        pred = pd.DataFrame(rows)

    # Ensure requested minimum columns are present and save one CSV per model.
    pred = pred[["mp_id", "bulk_modulus", "shear_modulus", "model"]].sort_values("mp_id")
    csv_path = outdir / f"{model_safe}_elasticity_predictions.csv"
    pred.to_csv(csv_path, index=False)
    print(f"[{model_name}] wrote {csv_path} ({len(pred)} rows)")
    return pred


def find_first_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_to_original = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    return None


def normalize_prediction_summary(raw: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Convert workflow property_summary.csv to mp_id/bulk_modulus/shear_modulus in GPa."""
    mp_col = find_first_column(
        raw.columns,
        ["source_identifier", "mp_id", "structure_name", "identifier"],
    )
    bulk_col = find_first_column(
        raw.columns,
        [
            "elasticity.bulk_modulus_vrh",
            "bulk_modulus_vrh",
            "bulk_modulus",
        ],
    )
    shear_col = find_first_column(
        raw.columns,
        [
            "elasticity.shear_modulus_vrh",
            "shear_modulus_vrh",
            "shear_modulus",
        ],
    )
    unit_col = find_first_column(raw.columns, ["elasticity.units", "units"])

    missing = [name for name, col in [("mp_id", mp_col), ("bulk", bulk_col), ("shear", shear_col)] if col is None]
    if missing:
        raise RuntimeError(
            f"Missing expected columns {missing} in workflow summary. Available columns: {list(raw.columns)}"
        )

    pred = pd.DataFrame(
        {
            "mp_id": raw[mp_col].astype(str).map(lambda x: Path(x).stem),
            "bulk_modulus": pd.to_numeric(raw[bulk_col], errors="coerce"),
            "shear_modulus": pd.to_numeric(raw[shear_col], errors="coerce"),
            "model": model_name,
        }
    )

    # MLFFElasticityResult stores moduli in eV/Å^3. Convert to GPa unless the
    # summary explicitly says values are already GPa.
    convert = True
    if unit_col is not None:
        units = set(raw[unit_col].dropna().astype(str).str.lower().unique())
        if units and all("gpa" in u for u in units):
            convert = False
    if convert:
        pred["bulk_modulus"] *= EV_A3_TO_GPA
        pred["shear_modulus"] *= EV_A3_TO_GPA

    return pred


def load_ground_truth(path: Path) -> pd.DataFrame:
    gt = pd.read_csv(path)
    bulk_col = find_first_column(gt.columns, ["bulk_modulus", "bulk_modulus_vrh"])
    shear_col = find_first_column(gt.columns, ["shear_modulus", "shear_modulus_vrh"])
    if "mp_id" not in gt.columns or bulk_col is None or shear_col is None:
        raise RuntimeError(
            f"Ground-truth CSV must contain mp_id and bulk/shear modulus columns. Got: {list(gt.columns)}"
        )
    return pd.DataFrame(
        {
            "mp_id": gt["mp_id"].astype(str).map(lambda x: Path(x).stem),
            "bulk_modulus_true": pd.to_numeric(gt[bulk_col], errors="coerce"),
            "shear_modulus_true": pd.to_numeric(gt[shear_col], errors="coerce"),
        }
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 0 else np.nan
    return {"n": int(len(y_true)), "mae": mae, "rmse": rmse, "r2": r2}


def make_plots_and_metrics(all_pred: pd.DataFrame, gt: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    merged = all_pred.merge(gt, on="mp_id", how="inner")
    merged_path = outdir / "all_models_elasticity_predictions_with_ground_truth.csv"
    merged.to_csv(merged_path, index=False)

    models = list(dict.fromkeys(merged["model"].tolist()))
    if not models:
        raise RuntimeError("No overlapping mp_id values between predictions and ground truth.")

    metrics_rows = []
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(models),
        figsize=(5.2 * len(models), 9.0),
        squeeze=False,
        constrained_layout=True,
    )

    plot_specs = [
        ("bulk_modulus_true", "bulk_modulus", "Bulk modulus"),
        ("shear_modulus_true", "shear_modulus", "Shear modulus"),
    ]

    for col_idx, model in enumerate(models):
        dfm = merged[merged["model"] == model]
        for row_idx, (true_col, pred_col, label) in enumerate(plot_specs):
            ax = axes[row_idx][col_idx]
            x = dfm[true_col].to_numpy(dtype=float)
            y = dfm[pred_col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            m = regression_metrics(x, y)
            metrics_rows.append({"model": model, "property": label, **m})
            ax.scatter(x[mask], y[mask], s=14, alpha=0.65, edgecolors="none")
            if mask.any():
                lo = float(min(np.nanmin(x[mask]), np.nanmin(y[mask])))
                hi = float(max(np.nanmax(x[mask]), np.nanmax(y[mask])))
                pad = 0.04 * (hi - lo) if hi > lo else 1.0
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"{model}: {label}\nMAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R²={m['r2']:.3f}")
            ax.set_xlabel("Ground truth (GPa)")
            ax.set_ylabel("MLIP prediction (GPa)")
            ax.grid(True, alpha=0.25)

    fig.suptitle("MLIP elasticity benchmark: predicted vs ground truth", fontsize=14)
    plot_path = outdir / "mlip_elasticity_benchmark.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    metrics = pd.DataFrame(metrics_rows)
    metrics_path = outdir / "mlip_elasticity_benchmark_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Wrote merged comparison CSV: {merged_path}")
    print(f"Wrote benchmark plot:       {plot_path}")
    print(f"Wrote benchmark metrics:    {metrics_path}")
    return metrics


async def async_main(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.force_elasticity_router:
        force_elasticity_router()

    cif_dir = maybe_make_limited_cif_dir(args.cif_dir.resolve(), args.limit, outdir)
    gt = load_ground_truth(args.ground_truth_csv.resolve())

    all_predictions = []
    for model in args.models:
        pred_csv = outdir / f"{safe_name(model)}_elasticity_predictions.csv"
        if args.resume and pred_csv.exists():
            print(f"[{model}] resume=True and {pred_csv} exists; reading existing predictions")
            pred = pd.read_csv(pred_csv)
        else:
            pred = await run_one_model(
                model,
                cif_dir,
                outdir,
                prefer_gpu=args.prefer_gpu,
                relax_cell=not args.keep_cell_fixed,
                relax_fmax=args.relax_fmax,
                relax_steps=args.relax_steps,
            )
        all_predictions.append(pred)

    all_pred = pd.concat(all_predictions, ignore_index=True)
    make_plots_and_metrics(all_pred, gt, outdir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MLFF elasticity predictions.")
    parser.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR)
    parser.add_argument("--ground-truth-csv", type=Path, default=DEFAULT_GT_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--prefer-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-cell-fixed", action="store_true", help="Disable cell relaxation before elasticity.")
    parser.add_argument("--relax-fmax", type=float, default=0.1)
    parser.add_argument("--relax-steps", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="Optional quick-test limit on number of CIFs.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing per-model prediction CSVs if present.")
    parser.add_argument(
        "--force-elasticity-router",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Monkeypatch MLFF router to run only MLFFElasticity (recommended for deterministic benchmark).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
