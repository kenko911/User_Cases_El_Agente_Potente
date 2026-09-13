#!/usr/bin/env python3
"""
Find low-energy adsorption configurations of H2 on a sufficiently large Fe(100) slab
using the public MLFF workflow API.

Assumptions:
- Fe is modeled as bcc alpha-Fe with a = 2.866 Å.
- Slab: bcc Fe(100), 4 x 4 lateral repeat, 5 atomic layers, 15 Å vacuum.
- Cell is kept fixed during slab/candidate relaxation; bottom layers are fixed by
  the adsorption workflow and the top 2 slab layers + adsorbate are relaxed.
- Adsorbate: intact H2 molecule, singlet, both H atoms allowed as anchors.
- Backend: MACE-OC20, the default/recommended backend for surface adsorption-site search.
"""

import os
import json
import csv
import asyncio
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict, is_dataclass

import logfire
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()

from ase.build import bulk, surface
from ase.io import write
from ase.geometry import cell_to_cellpar

from grafico.deps import GraficoDeps
from domains.mlffs.graph.mlffs_graph import MLFFWorkflowRequest, execute_mlff_workflow


def to_plain(obj):
    """Best-effort conversion of pydantic/dataclass/tool-result objects to JSON-like objects."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return to_plain(obj.model_dump())
    if is_dataclass(obj):
        return to_plain(asdict(obj))
    if hasattr(obj, "__dict__"):
        return {k: to_plain(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def find_adsorption_result(output_obj):
    """Locate the adsorption result in an MLFFOutput-like object without recursive energy parsing."""
    # Prefer explicit attribute names used by the workflow result models.
    for name in (
        "adsorption_site_search",
        "adsorption_site_search_result",
        "adsorption_result",
        "surface_adsorption_site_search",
    ):
        if hasattr(output_obj, name):
            val = getattr(output_obj, name)
            if val is not None:
                return val
    data = to_plain(output_obj)
    if isinstance(data, dict):
        for name in (
            "adsorption_site_search",
            "adsorption_site_search_result",
            "adsorption_result",
            "surface_adsorption_site_search",
        ):
            val = data.get(name)
            if val:
                return val
    return None


def extract_candidate_records(result_plain):
    """Use canonical candidates/candidates.csv for per-configuration reporting."""
    candidates = result_plain.get("candidates") if isinstance(result_plain, dict) else None
    csv_path = None
    if isinstance(result_plain, dict):
        for key in ("candidates_csv_path", "candidates_csv", "candidate_summary_csv", "summary_csv_path"):
            if result_plain.get(key):
                p = Path(str(result_plain[key]))
                if p.exists() and p.is_file() and p.suffix.lower() == ".csv":
                    csv_path = p
                    break

    # If a candidate CSV is available, treat it as canonical and read it.
    if csv_path is not None:
        with csv_path.open(newline="") as f:
            return list(csv.DictReader(f)), csv_path

    if isinstance(candidates, list):
        return candidates, None
    return [], None


def num_from_record(record, keys):
    for key in keys:
        if key in record and record[key] not in (None, "", "None"):
            try:
                return float(record[key])
            except Exception:
                pass
    return None


def str_from_record(record, keys):
    for key in keys:
        if key in record and record[key] not in (None, "", "None"):
            return str(record[key])
    return ""


def rank_key(record):
    for key in ("rank", "candidate_rank", "index", "candidate_index", "site_index"):
        if key in record and record[key] not in (None, ""):
            try:
                return (0, int(float(record[key])))
            except Exception:
                return (0, str(record[key]))
    e = num_from_record(record, ["adsorption_energy", "adsorption_energy_ev", "E_ads", "e_ads"])
    return (1, float("inf") if e is None else e)


async def main():
    workspace = Path(os.getenv("GRAPHCHAT_WORKSPACE") or os.getcwd()).resolve()
    outdir = workspace / "h2_fe100_adsorption_outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Structure realization: generate and validate Fe(100) slab ---
    a_fe = 2.866  # Angstrom, alpha-Fe bcc room-temperature reference
    fe_bulk = bulk("Fe", "bcc", a=a_fe, cubic=True)
    slab = surface(fe_bulk, (1, 0, 0), layers=5, vacuum=15.0, periodic=True)
    slab = slab.repeat((4, 4, 1))
    slab.pbc = (True, True, False)
    slab.center(axis=2, vacuum=15.0)

    slab_path = outdir / "Fe100_bcc_4x4_5layer_vac15.cif"
    write(slab_path, slab)

    validation = {
        "formula": slab.get_chemical_formula(),
        "n_atoms": len(slab),
        "pbc": list(map(bool, slab.pbc)),
        "cell_parameters_A_deg": [float(x) for x in cell_to_cellpar(slab.cell)],
        "min_position_A": [float(x) for x in slab.positions.min(axis=0)],
        "max_position_A": [float(x) for x in slab.positions.max(axis=0)],
        "assumptions": {
            "bulk_phase": "bcc alpha-Fe",
            "lattice_parameter_A": a_fe,
            "surface_facet": "Fe(100)",
            "lateral_repeat": "4x4",
            "layers": 5,
            "vacuum_A": 15.0,
            "slab_cell_relaxed": False,
            "adsorbate": "intact H2",
            "adsorbate_charge": 0,
            "adsorbate_spin_multiplicity": 1,
            "model_name": "MACE-OC20",
        },
    }
    with (outdir / "input_slab_validation.json").open("w") as f:
        json.dump(validation, f, indent=2)
    print("Generated Fe(100) slab:")
    print(json.dumps(validation, indent=2))
    logfire.info("Generated Fe(100) slab", slab_path=str(slab_path), n_atoms=len(slab))

    deps = GraficoDeps(
        ws_url=os.getenv("GRAPHCHAT_AGENT_WS_URL") or os.getenv("VITE_WS_URL", "ws://graphchat:3000"),
        room=os.getenv("GRAPHCHAT_ROOM", "room"),
        sparql_endpoint=os.getenv("SPARQL_ENDPOINT", "http://blazegraph:8080/blazegraph/namespace/kb/sparql"),
        workspace_path_override=str(workspace),
    )

    # Construct the typed MLFF workflow request. Since adsorption_site_search is supplied,
    # this is the complete property set; AI property routing is intentionally bypassed.
    request = MLFFWorkflowRequest(
        summarised_user_query=(
            "Find the lowest-energy adsorption configurations of intact H2 on a sufficiently "
            "large bcc Fe(100) slab; report adsorption energies and relative energies; save "
            "all candidate geometries and relaxation trajectories."
        ),
        identifier_type="from_path",
        identifier=str(slab_path),
        model_name="MACE-OC20",
        relax_cell=False,
        relax_fmax=0.05,
        relax_steps=500,
        adsorption_site_search={
            "adsorbate_identifier": "[H][H]",
            "adsorbate_identifier_type": "smiles",
            "charge": 0,
            "spin_multiplicity": 1,
            "adsorbate_anchor_indices": [0, 1],
            "site_reduction_method": "ase",
            "polar_angles_degrees": [0.0, 45.0, 90.0],
            "azimuthal_rotations": 8,
            "n_relaxed_surface_layers": 2,
            # Large enough to relax all/most symmetry-distinct site/orientation candidates for this ideal slab.
            "n_geometry_relaxations": 64,
            "n_lowest_structures": 64,
            "relaxation_trajectory_interval": 1,
        },
        update_graph=False,
        prefer_gpu=True,
    )

    logfire.info("Starting MLFF adsorption workflow", model="MACE-OC20", slab=str(slab_path))
    outputs = await execute_mlff_workflow(request=request, deps=deps, run_id="h2_fe100_adsorption_search")
    plain_outputs = to_plain(outputs)
    with (outdir / "raw_mlff_outputs.json").open("w") as f:
        json.dump(plain_outputs, f, indent=2)

    # Workflow returns one output for this one slab.
    first_output = outputs[0] if isinstance(outputs, list) and outputs else outputs
    ads_result = find_adsorption_result(first_output)
    if ads_result is None:
        raise RuntimeError("No adsorption-site-search result found in MLFF workflow output.")
    ads_plain = to_plain(ads_result)
    with (outdir / "adsorption_result.json").open("w") as f:
        json.dump(ads_plain, f, indent=2)

    records, source_csv = extract_candidate_records(ads_plain)
    records = sorted(records, key=rank_key)

    # Write a compact report while preserving each candidate's own energy fields/paths.
    report_csv = outdir / "h2_fe100_candidate_energy_report.csv"
    fieldnames = [
        "rank",
        "adsorption_energy_eV",
        "relative_energy_eV",
        "structure_path",
        "relaxation_trajectory_path",
        "site_label",
        "orientation_label",
    ]
    rows = []
    for i, rec in enumerate(records, start=1):
        e_ads = num_from_record(rec, ["adsorption_energy", "adsorption_energy_eV", "adsorption_energy_ev", "E_ads", "e_ads"])
        e_rel = num_from_record(rec, ["relative_energy", "relative_energy_eV", "relative_energy_ev", "E_rel", "e_rel"])
        rows.append({
            "rank": str_from_record(rec, ["rank", "candidate_rank"]) or i,
            "adsorption_energy_eV": "" if e_ads is None else f"{e_ads:.10f}",
            "relative_energy_eV": "" if e_rel is None else f"{e_rel:.10f}",
            "structure_path": str_from_record(rec, ["structure_path", "relaxed_structure_path", "atoms_path", "candidate_structure_path", "geometry_path"]),
            "relaxation_trajectory_path": str_from_record(rec, ["relaxation_trajectory_path", "trajectory_path", "traj_path", "relax_traj_path"]),
            "site_label": str_from_record(rec, ["site_label", "site_type", "site", "site_index"]),
            "orientation_label": str_from_record(rec, ["orientation_label", "orientation", "orientation_index"]),
        })

    with report_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_summary = ads_plain.get("best_adsorption_energy") if isinstance(ads_plain, dict) else None
    candidate_ads = [float(r["adsorption_energy_eV"]) for r in rows if r["adsorption_energy_eV"]]
    min_candidate_ads = min(candidate_ads) if candidate_ads else None
    best_check = {
        "best_adsorption_energy_from_result": best_summary,
        "minimum_candidate_adsorption_energy_eV": min_candidate_ads,
        "matches_min_candidate": None,
    }
    if best_summary is not None and min_candidate_ads is not None:
        try:
            best_check["matches_min_candidate"] = abs(float(best_summary) - min_candidate_ads) < 1e-6
        except Exception:
            best_check["matches_min_candidate"] = False
    with (outdir / "best_energy_consistency_check.json").open("w") as f:
        json.dump(best_check, f, indent=2)

    print("\nAdsorption-site-search artifacts:")
    print(f"  Run-level/per-structure raw output: {outdir / 'raw_mlff_outputs.json'}")
    print(f"  Adsorption result JSON:           {outdir / 'adsorption_result.json'}")
    if source_csv:
        print(f"  Canonical candidates CSV:         {source_csv}")
    print(f"  Compact energy report CSV:        {report_csv}")
    print(f"  Input slab CIF:                   {slab_path}")
    print("\nBest-energy consistency check:")
    print(json.dumps(best_check, indent=2))

    print("\nLowest-energy H2/Fe(100) candidate configurations:")
    if not rows:
        print("  No candidate records were found; inspect adsorption_result.json and workflow artifacts.")
    else:
        for row in rows:
            print(
                f"  rank {row['rank']}: "
                f"E_ads={row['adsorption_energy_eV']} eV, "
                f"E_rel={row['relative_energy_eV']} eV, "
                f"structure={row['structure_path']}, "
                f"trajectory={row['relaxation_trajectory_path']}"
            )


if __name__ == "__main__":
    asyncio.run(main())