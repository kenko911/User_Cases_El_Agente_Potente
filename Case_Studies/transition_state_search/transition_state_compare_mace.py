#!/usr/bin/env python3
"""
Transition-state search for a pair of structures in a workspace directory,
with bottom two slab layers fixed, comparing MACE-OMAT and MACE-OC20.

This script:
  1. Discovers two endpoint structures in INPUT_DIR (reactant/product).
  2. Applies ASE FixAtoms constraints to atoms in the bottom two z-layers.
  3. Saves constrained endpoint files into GRAPHCHAT_WORKSPACE.
  4. Runs the public MLFF workflow API twice via MLFFWorkflowRequest and
     execute_mlff_workflow: once with MACE-OMAT and once with MACE-OC20.
  5. Extracts final pathway energies from transition_state_search.extxyz or
     returned workflow payloads.
  6. Generates a comparison plot of the optimized transition-state pathway.

Assumptions:
  - Endpoint files have identical atoms in identical order.
  - The slab normal is the Cartesian z direction.
  - The two lowest occupied z-layers are slab layers to be fixed.
  - Input files are ASE-readable (.xyz, .extxyz, .cif, .traj, .vasp/.poscar/.contcar).
"""

import asyncio
import json
import math
import os
import re
import shutil
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import logfire
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write

from grafico.deps import GraficoDeps
from domains.mlffs.graph.mlffs_graph import MLFFWorkflowRequest, execute_mlff_workflow


INPUT_DIR = Path("/scratch/kenko/akg4pyscf/.graphchat-workspaces/kenko_room/").resolve()
WORKSPACE = Path(os.getenv("GRAPHCHAT_WORKSPACE") or os.getcwd()).resolve()
OUT_DIR = WORKSPACE / "ts_mace_omat_vs_oc20"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Transition-state / NEB settings. Adjust here if a tighter/faster run is desired.
N_IMAGES = 7                  # Total images including endpoints
FMAX = 0.05                   # eV/Ang convergence threshold for path optimization
MAX_STEPS = 300               # optimizer steps for path search
REFINEMENT_METHOD = "sella"   # "sella", "dimer", or "none"
PATH_METHOD = "neb"           # "neb" or "ml-fsm"
RELAX_FMAX = 0.05
RELAX_STEPS = 500
RELAX_CELL = False            # keep slab cell fixed
Z_LAYER_TOL = 0.45            # Angstrom tolerance for grouping atoms into z layers


def build_deps() -> GraficoDeps:
    return GraficoDeps(
        ws_url=os.getenv("GRAPHCHAT_AGENT_WS_URL") or os.getenv("VITE_WS_URL", "ws://graphchat:3000"),
        room=os.getenv("GRAPHCHAT_ROOM", "room"),
        sparql_endpoint=os.getenv("SPARQL_ENDPOINT", "http://blazegraph:8080/blazegraph/namespace/kb/sparql"),
        workspace_path_override=os.getenv("GRAPHCHAT_WORKSPACE") or os.getcwd(),
    )


def ase_read_one(path: Path) -> Atoms:
    """Read one endpoint robustly; for multi-frame files use the last frame."""
    try:
        atoms = read(str(path), index=-1)
    except Exception:
        atoms = read(str(path))
    if not isinstance(atoms, Atoms):
        # ASE may return a list if format/index handling differs.
        atoms = atoms[-1]
    return atoms


def discover_structure_files(directory: Path) -> List[Path]:
    exts = {".xyz", ".extxyz", ".cif", ".traj", ".vasp", ".poscar", ".contcar"}
    candidates: List[Path] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        name_lower = path.name.lower()
        if suffix in exts or name_lower in {"poscar", "contcar"}:
            try:
                _ = ase_read_one(path)
                candidates.append(path)
            except Exception as exc:
                logfire.debug("Skipping unreadable candidate {path}: {exc}", path=str(path), exc=str(exc))
    return sorted(candidates, key=lambda p: p.name.lower())


def choose_endpoint_pair(files: Sequence[Path]) -> Tuple[Path, Path]:
    """Choose reactant/product endpoints using common filename conventions; otherwise require exactly two."""
    if len(files) < 2:
        raise RuntimeError(f"Need at least two ASE-readable endpoint structures in {INPUT_DIR}; found {len(files)}.")

    reactant_patterns = [r"react", r"reac", r"initial", r"init", r"start", r"is\b", r"minimum1", r"min1", r"r\b"]
    product_patterns = [r"prod", r"product", r"final", r"finish", r"end", r"fs\b", r"minimum2", r"min2", r"p\b"]

    def score(path: Path, patterns: Sequence[str]) -> int:
        stem = path.stem.lower()
        return sum(1 for pat in patterns if re.search(pat, stem))

    reactants = sorted([(score(p, reactant_patterns), p) for p in files], reverse=True)
    products = sorted([(score(p, product_patterns), p) for p in files], reverse=True)
    if reactants and products and reactants[0][0] > 0 and products[0][0] > 0 and reactants[0][1] != products[0][1]:
        return reactants[0][1], products[0][1]

    if len(files) == 2:
        return files[0], files[1]

    raise RuntimeError(
        "Found more than two readable structures, but could not unambiguously infer reactant/product filenames.\n"
        f"Files: {[p.name for p in files]}\n"
        "Rename two files with names containing reactant/initial/start and product/final/end, or leave only the endpoint pair."
    )


def validate_endpoint_compatibility(a: Atoms, b: Atoms, path_a: Path, path_b: Path) -> None:
    if len(a) != len(b):
        raise RuntimeError(f"Endpoint atom-count mismatch: {path_a.name} has {len(a)}, {path_b.name} has {len(b)}.")
    symbols_a = a.get_chemical_symbols()
    symbols_b = b.get_chemical_symbols()
    if symbols_a != symbols_b:
        mismatch = [(i, sa, sb) for i, (sa, sb) in enumerate(zip(symbols_a, symbols_b)) if sa != sb][:10]
        raise RuntimeError(
            "Endpoint atom ordering/elements differ. Transition-state search requires identical atoms in identical order. "
            f"First mismatches: {mismatch}"
        )


def cluster_z_layers(z_values: np.ndarray, tol: float = Z_LAYER_TOL) -> List[List[int]]:
    """Cluster atoms into z-layers by sorted z coordinate."""
    order = np.argsort(z_values)
    layers: List[List[int]] = []
    current: List[int] = []
    current_mean: Optional[float] = None
    for idx in order:
        z = float(z_values[idx])
        if current_mean is None or abs(z - current_mean) <= tol:
            current.append(int(idx))
            current_mean = float(np.mean(z_values[current]))
        else:
            layers.append(current)
            current = [int(idx)]
            current_mean = z
    if current:
        layers.append(current)
    return layers


def bottom_two_layer_indices(atoms: Atoms, tol: float = Z_LAYER_TOL) -> List[int]:
    z = atoms.get_positions()[:, 2]
    layers = cluster_z_layers(z, tol=tol)
    if len(layers) < 2:
        raise RuntimeError(
            f"Could not identify two bottom z-layers; found {len(layers)} layer(s). "
            "Check that the slab normal is z and adjust Z_LAYER_TOL."
        )
    fixed = sorted(layers[0] + layers[1])
    return fixed


def apply_bottom_two_layer_constraint(atoms: Atoms) -> Tuple[Atoms, List[int]]:
    constrained = atoms.copy()
    fixed = bottom_two_layer_indices(constrained)
    constrained.set_constraint(FixAtoms(indices=fixed))
    return constrained, fixed


def jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if is_dataclass(obj):
        return jsonable(asdict(obj))
    if hasattr(obj, "model_dump"):
        return jsonable(obj.model_dump())
    if hasattr(obj, "dict"):
        return jsonable(obj.dict())
    if hasattr(obj, "__dict__"):
        return jsonable(vars(obj))
    return repr(obj)


def recursive_find_paths(obj: Any, basename: str) -> List[Path]:
    found: List[Path] = []
    if isinstance(obj, dict):
        for v in obj.values():
            found.extend(recursive_find_paths(v, basename))
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            found.extend(recursive_find_paths(v, basename))
    elif isinstance(obj, str):
        if obj.endswith(basename) and Path(obj).exists():
            found.append(Path(obj))
    return found


def extract_energy_from_atoms(atoms: Atoms) -> Optional[float]:
    """Return an energy in eV from ASE Atoms info/arrays/calculator if available."""
    keys = ["energy", "Energy", "E", "potential_energy", "neb_energy", "mace_energy"]
    for key in keys:
        if key in atoms.info:
            try:
                return float(atoms.info[key])
            except Exception:
                pass
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        return None


def read_profile_from_extxyz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    frames = read(str(path), index=":")
    if isinstance(frames, Atoms):
        frames = [frames]
    energies: List[float] = []
    for frame in frames:
        energy = extract_energy_from_atoms(frame)
        if energy is None or not math.isfinite(energy):
            raise RuntimeError(f"No finite energy found for one image in {path}")
        energies.append(energy)
    energies_arr = np.array(energies, dtype=float)
    rel = energies_arr - energies_arr[0]
    rxn = np.linspace(0.0, 1.0, len(rel))
    return rxn, rel


def fallback_profile_from_json(payload: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Best-effort extraction if no extxyz path is available."""
    data = jsonable(payload)
    energies: List[float] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for key, val in x.items():
                kl = str(key).lower()
                if kl in {"path_energies", "energies", "neb_energies", "image_energies"} and isinstance(val, list):
                    tmp = []
                    for y in val:
                        if isinstance(y, (int, float)):
                            tmp.append(float(y))
                    if len(tmp) >= 2:
                        energies.clear()
                        energies.extend(tmp)
                        return
                walk(val)
        elif isinstance(x, list):
            for y in x:
                walk(y)

    walk(data)
    if len(energies) >= 2:
        arr = np.array(energies, dtype=float)
        return np.linspace(0.0, 1.0, len(arr)), arr - arr[0]
    return None


async def run_one_model(model_name: str, reactant_path: Path, product_path: Path, deps: GraficoDeps) -> Dict[str, Any]:
    run_id = f"ts_{model_name.lower().replace('-', '_')}"
    summary = (
        f"Run double-ended transition-state search / optimized NEB pathway for constrained slab endpoints. "
        f"Use {model_name}; compare transition-state pathway energy profile. "
        "The endpoint files contain FixAtoms constraints for the bottom two z-layers; keep the slab cell fixed."
    )

    request = MLFFWorkflowRequest(
        summarised_user_query=summary,
        identifier_type="from_path",
        identifier=str(reactant_path),
        transition_state_search={
            "endpoints": [
                {"identifier_type": "from_path", "identifier": str(reactant_path)},
                {"identifier_type": "from_path", "identifier": str(product_path)},
            ],
            "path_method": PATH_METHOD,
            "refinement_method": REFINEMENT_METHOD,
            "n_images": N_IMAGES,
            "climb": True,
            "fmax": FMAX,
            "max_steps": MAX_STEPS,
        },
        model_name=model_name,
        relax_cell=RELAX_CELL,
        relax_fmax=RELAX_FMAX,
        relax_steps=RELAX_STEPS,
        prefer_gpu=True,
    )

    # Public workflow attempt 1.
    try:
        logfire.info("Starting MLFF workflow for {model}", model=model_name)
        output = await execute_mlff_workflow(request=request, deps=deps, run_id=run_id)
        return {"model": model_name, "status": "success", "output": output}
    except Exception as exc1:
        tb1 = traceback.format_exc()
        logfire.info("First MLFF workflow attempt failed for {model}: {err}", model=model_name, err=str(exc1))
        # Single targeted retry: use no saddle refinement if refinement setup failed. This keeps the same NEB scientific target
        # and changes only post-path refinement orchestration.
        retry_request = request.model_copy(deep=True)
        retry_request.transition_state_search["refinement_method"] = "none"
        retry_request.transition_state_search["max_steps"] = max(MAX_STEPS, 500)
        try:
            logfire.info("Retrying MLFF workflow for {model} with refinement_method='none'", model=model_name)
            output = await execute_mlff_workflow(request=retry_request, deps=deps, run_id=f"{run_id}_retry_no_refine")
            return {
                "model": model_name,
                "status": "success_after_retry_no_refine",
                "first_error": tb1,
                "output": output,
            }
        except Exception as exc2:
            tb2 = traceback.format_exc()
            return {
                "model": model_name,
                "status": "failed",
                "first_error": tb1,
                "second_error": tb2,
                "output": None,
            }


def locate_profile_file(result_payload: Any, model_name: str) -> Optional[Path]:
    data = jsonable(result_payload)
    direct = recursive_find_paths(data, "transition_state_search.extxyz")
    if direct:
        return direct[0]

    # Search within likely artifact directories mentioned in the output.
    dirs: List[Path] = []
    def collect_dirs(x: Any) -> None:
        if isinstance(x, dict):
            for key, val in x.items():
                if str(key).endswith("artifact_dir") or str(key) in {"run_artifact_dir", "artifact_dir", "work_dir"}:
                    if isinstance(val, str) and Path(val).exists():
                        dirs.append(Path(val))
                collect_dirs(val)
        elif isinstance(x, list):
            for y in x:
                collect_dirs(y)
    collect_dirs(data)

    # Add workspace output directories as a final bounded search root.
    dirs.extend([OUT_DIR, WORKSPACE])
    seen = set()
    for root in dirs:
        root = root.resolve()
        if root in seen or not root.exists() or not root.is_dir():
            continue
        seen.add(root)
        matches = list(root.rglob("transition_state_search.extxyz"))
        model_token = model_name.lower().replace("-", "_")
        model_matches = [m for m in matches if model_token in str(m).lower() or model_name.lower() in str(m).lower()]
        if model_matches:
            return sorted(model_matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        if matches:
            return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def make_plot(profiles: Dict[str, Tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    plt.figure(figsize=(7.0, 4.8), dpi=180)
    markers = {"MACE-OMAT": "o", "MACE-OC20": "s"}
    for model, (x, e) in profiles.items():
        plt.plot(x, e, marker=markers.get(model, "o"), linewidth=2.0, label=model)
        max_i = int(np.argmax(e))
        plt.scatter([x[max_i]], [e[max_i]], s=70, zorder=5)
        plt.annotate(f"barrier {e[max_i]:.3f} eV", (x[max_i], e[max_i]), textcoords="offset points", xytext=(5, 7), fontsize=8)
    plt.xlabel("Normalized reaction coordinate")
    plt.ylabel("Relative energy / eV")
    plt.title("Optimized transition-state pathway energy profile")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


async def main() -> None:
    logfire.info("Starting TS comparison script; input_dir={input_dir}; workspace={workspace}", input_dir=str(INPUT_DIR), workspace=str(WORKSPACE))
    if not INPUT_DIR.exists():
        raise RuntimeError(f"Input directory does not exist: {INPUT_DIR}")

    files = discover_structure_files(INPUT_DIR)
    print("Discovered ASE-readable structure files:")
    for f in files:
        print(f"  - {f}")
    reactant_file, product_file = choose_endpoint_pair(files)
    print(f"Selected reactant endpoint: {reactant_file}")
    print(f"Selected product endpoint:  {product_file}")

    reactant = ase_read_one(reactant_file)
    product = ase_read_one(product_file)
    validate_endpoint_compatibility(reactant, product, reactant_file, product_file)

    # Apply bottom-two-layer constraints. Use indices determined from the reactant and verify the product is compatible.
    fixed_indices = bottom_two_layer_indices(reactant)
    reactant_constrained = reactant.copy()
    product_constrained = product.copy()
    reactant_constrained.set_constraint(FixAtoms(indices=fixed_indices))
    product_constrained.set_constraint(FixAtoms(indices=fixed_indices))

    # Preserve PBC/cell; if missing, warn but still write structures.
    if not any(reactant_constrained.get_pbc()):
        print("WARNING: Reactant endpoint is non-periodic according to ASE PBC flags; bottom-layer fixing still applied by z-coordinate.")
    if not np.allclose(reactant_constrained.get_cell().array, product_constrained.get_cell().array, atol=1e-4):
        print("WARNING: Endpoint cells differ; the workflow will keep cells fixed but NEB interpolation may be affected.")

    constrained_reactant_path = OUT_DIR / "reactant_bottom_two_layers_fixed.extxyz"
    constrained_product_path = OUT_DIR / "product_bottom_two_layers_fixed.extxyz"
    write(str(constrained_reactant_path), reactant_constrained)
    write(str(constrained_product_path), product_constrained)

    fixed_symbols = [reactant_constrained[i].symbol for i in fixed_indices]
    assumptions = {
        "input_dir": str(INPUT_DIR),
        "reactant_file": str(reactant_file),
        "product_file": str(product_file),
        "constrained_reactant_path": str(constrained_reactant_path),
        "constrained_product_path": str(constrained_product_path),
        "n_atoms": len(reactant_constrained),
        "formula": reactant_constrained.get_chemical_formula(),
        "fixed_bottom_two_layer_indices_zero_based": fixed_indices,
        "fixed_bottom_two_layer_symbols": fixed_symbols,
        "n_fixed_atoms": len(fixed_indices),
        "z_layer_tolerance_angstrom": Z_LAYER_TOL,
        "slab_normal_assumption": "Cartesian z direction",
        "relax_cell": RELAX_CELL,
        "n_images": N_IMAGES,
        "fmax_eV_per_ang": FMAX,
        "max_steps": MAX_STEPS,
        "refinement_method_initial_attempt": REFINEMENT_METHOD,
    }
    (OUT_DIR / "ts_search_assumptions.json").write_text(json.dumps(assumptions, indent=2))
    print(json.dumps(assumptions, indent=2))

    deps = build_deps()
    results = []
    # Run the two model comparisons concurrently.
    results = await asyncio.gather(
        run_one_model("MACE-OMAT", constrained_reactant_path, constrained_product_path, deps),
        run_one_model("MACE-OC20", constrained_reactant_path, constrained_product_path, deps),
    )

    serialised_results = jsonable(results)
    result_json_path = OUT_DIR / "workflow_results_summary.json"
    result_json_path.write_text(json.dumps(serialised_results, indent=2))

    profiles: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    profile_records: Dict[str, Any] = {}
    for result in results:
        model = result["model"]
        if not str(result.get("status", "")).startswith("success"):
            profile_records[model] = {"status": result.get("status"), "error": result.get("second_error") or result.get("first_error")}
            print(f"{model} failed; see {result_json_path}")
            continue
        profile_path = locate_profile_file(result.get("output"), model)
        if profile_path is not None:
            rxn, rel_e = read_profile_from_extxyz(profile_path)
            profiles[model] = (rxn, rel_e)
            profile_records[model] = {
                "status": result.get("status"),
                "profile_source": str(profile_path),
                "relative_energies_eV": rel_e.tolist(),
                "barrier_eV": float(np.max(rel_e)),
                "reaction_energy_eV": float(rel_e[-1]),
            }
        else:
            fallback = fallback_profile_from_json(result.get("output"))
            if fallback is not None:
                rxn, rel_e = fallback
                profiles[model] = (rxn, rel_e)
                profile_records[model] = {
                    "status": result.get("status"),
                    "profile_source": "workflow JSON payload",
                    "relative_energies_eV": rel_e.tolist(),
                    "barrier_eV": float(np.max(rel_e)),
                    "reaction_energy_eV": float(rel_e[-1]),
                }
            else:
                profile_records[model] = {
                    "status": result.get("status"),
                    "profile_source": None,
                    "warning": "Could not locate transition_state_search.extxyz or energy list in workflow payload.",
                }

    profiles_json_path = OUT_DIR / "energy_profiles.json"
    profiles_json_path.write_text(json.dumps(profile_records, indent=2))

    plot_path = OUT_DIR / "mace_omat_vs_oc20_ts_energy_profile.png"
    if profiles:
        make_plot(profiles, plot_path)
        print(f"Saved energy profile plot: {plot_path}")
    else:
        print("No successful energy profiles were available; plot was not generated.")

    print("\n=== Transition-state comparison summary ===")
    for model, rec in profile_records.items():
        print(f"{model}: {json.dumps(rec, indent=2)}")
    print(f"\nArtifacts directory: {OUT_DIR}")
    print(f"Assumptions: {OUT_DIR / 'ts_search_assumptions.json'}")
    print(f"Workflow summary JSON: {result_json_path}")
    print(f"Energy profiles JSON: {profiles_json_path}")
    if profiles:
        print(f"Energy profile plot: {plot_path}")


if __name__ == "__main__":
    asyncio.run(main())