#!/usr/bin/env python3
"""
Run a 2D enhanced-sampling calculation for alanine dipeptide with Orb-OMOL.

Input structure:
  /scratch/kenko/repos/adaptive_sampling/examples/JCTC_2025_opeseabf/alanine_dipeptide/test_potente/alanine_c7ax.xyz

Workflow:
  - Read and validate the supplied XYZ.
  - Infer alanine-dipeptide backbone torsions phi and psi from bonding.
  - Build an MLFFWorkflowRequest with explicit enhanced_sampling settings.
  - Execute Potente's public MLFF workflow API using Orb-OMOL.
"""

import asyncio
import json
import os
import shutil
from pathlib import Path
from pprint import pprint

import logfire

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()

from ase.io import read, write
from ase.neighborlist import natural_cutoffs, NeighborList

from grafico.deps import GraficoDeps
from domains.mlffs.graph.mlffs_graph import MLFFWorkflowRequest, execute_mlff_workflow

SOURCE_XYZ = Path(
    "/scratch/kenko/repos/adaptive_sampling/examples/JCTC_2025_opeseabf/alanine_dipeptide/test_potente/alanine_c7ax.xyz"
)
WORKSPACE = Path(os.getenv("GRAPHCHAT_WORKSPACE") or os.getcwd()).resolve()
LOCAL_XYZ = WORKSPACE / "alanine_c7ax_input.xyz"
SUMMARY_JSON = WORKSPACE / "alanine_dipeptide_orb_omol_enhanced_sampling_summary.json"


def build_deps() -> GraficoDeps:
    return GraficoDeps(
        ws_url=os.getenv("GRAPHCHAT_AGENT_WS_URL") or os.getenv("VITE_WS_URL", "ws://graphchat:3000"),
        room=os.getenv("GRAPHCHAT_ROOM", "room"),
        sparql_endpoint=os.getenv("SPARQL_ENDPOINT", "http://blazegraph:8080/blazegraph/namespace/kb/sparql"),
        workspace_path_override=str(WORKSPACE),
    )


def connectivity(atoms):
    """Return a simple covalent-neighbor dictionary from ASE natural cutoffs."""
    cutoffs = natural_cutoffs(atoms, mult=1.20)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    neigh = {}
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        # XYZ molecule is non-periodic; ignore offset duplicates defensively.
        neigh[i] = sorted(set(int(j) for j in indices))
    return neigh


def infer_alanine_dipeptide_phi_psi(atoms):
    """
    Infer the standard alanine-dipeptide Ramachandran torsions:
      phi = C(acetyl carbonyl)-N(alanine)-Cα-C(alanine carbonyl)
      psi = N(alanine)-Cα-C(alanine carbonyl)-N(methylamide)

    The algorithm uses only element labels and local connectivity, so it is
    robust to one-based/zero-based formatting and to atom-order differences.
    Returned indices are zero-based, as required by the MLFF workflow.
    """
    syms = atoms.get_chemical_symbols()
    neigh = connectivity(atoms)

    def is_elem(i, e):
        return syms[i] == e

    carbonyl_cs = []
    for i, e in enumerate(syms):
        if e != "C":
            continue
        has_o_neighbor = any(is_elem(j, "O") for j in neigh[i])
        has_heavy_neighbor = sum(syms[j] != "H" for j in neigh[i]) >= 2
        if has_o_neighbor and has_heavy_neighbor:
            carbonyl_cs.append(i)

    nitrogens = [i for i, e in enumerate(syms) if e == "N"]
    alpha_candidates = []
    for i, e in enumerate(syms):
        if e != "C":
            continue
        if i in carbonyl_cs:
            continue
        n_neighbors = [j for j in neigh[i] if is_elem(j, "N")]
        carbonyl_neighbors = [j for j in neigh[i] if j in carbonyl_cs]
        carbon_neighbors = [j for j in neigh[i] if is_elem(j, "C") and j not in carbonyl_cs]
        h_neighbors = [j for j in neigh[i] if is_elem(j, "H")]
        # Alanine Cα is bonded to N, the alanine carbonyl C, methyl C, and H.
        if n_neighbors and carbonyl_neighbors and carbon_neighbors and h_neighbors:
            alpha_candidates.append((i, n_neighbors, carbonyl_neighbors, carbon_neighbors))

    if len(alpha_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one alanine C-alpha candidate, found {len(alpha_candidates)}: {alpha_candidates}. "
            f"Carbonyl C candidates={carbonyl_cs}, N candidates={nitrogens}"
        )

    ca, n_neighbors, carbonyl_neighbors, _ = alpha_candidates[0]
    n_ala = n_neighbors[0]
    c_ala = carbonyl_neighbors[0]

    # The acetyl carbonyl C is bonded to N_ala but is not the alanine carbonyl C.
    c_prev = [c for c in carbonyl_cs if c != c_ala and n_ala in neigh[c]]
    # The methylamide N is bonded to the alanine carbonyl C but is not N_ala.
    n_next = [n for n in nitrogens if n != n_ala and n in neigh[c_ala]]

    if len(c_prev) != 1 or len(n_next) != 1:
        raise RuntimeError(
            "Could not uniquely identify flanking atoms for phi/psi. "
            f"Cprev candidates={c_prev}, Nnext candidates={n_next}, "
            f"carbonyl C candidates={carbonyl_cs}, N candidates={nitrogens}, Cα={ca}"
        )

    phi = [c_prev[0], n_ala, ca, c_ala]
    psi = [n_ala, ca, c_ala, n_next[0]]
    return phi, psi, neigh


def torsion_degrees(atoms, idxs):
    # ASE get_dihedral returns degrees for the four atom indices.
    return float(atoms.get_dihedral(*idxs))


def result_to_summary(result):
    """Collect a compact, JSON-serializable summary from workflow outputs."""
    outputs = result if isinstance(result, list) else [result]
    compact = []
    for out in outputs:
        if hasattr(out, "model_dump"):
            data = out.model_dump(mode="json")
        elif hasattr(out, "dict"):
            data = out.dict()
        else:
            data = out
        item = {
            "chemical_formula": data.get("chemical_formula"),
            "structure_iri": data.get("structure_iri"),
            "run_artifact_dir": data.get("run_artifact_dir"),
            "run_summary_path": data.get("run_summary_path"),
            "output_json_path": data.get("output_json_path"),
            "summary_csv_path": data.get("summary_csv_path"),
        }
        # Preserve enhanced-sampling-related result roots without recursively dumping everything.
        for key, value in data.items():
            if "enhanced" in key.lower() or "sampling" in key.lower():
                item[key] = value
        compact.append(item)
    return compact


async def main():
    logfire.info("Starting Orb-OMOL enhanced-sampling workflow for alanine dipeptide")
    print(f"Workspace: {WORKSPACE}")
    print(f"Source XYZ: {SOURCE_XYZ}")

    if not SOURCE_XYZ.exists():
        raise FileNotFoundError(f"Required input file not found: {SOURCE_XYZ}")

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_XYZ, LOCAL_XYZ)

    atoms = read(str(LOCAL_XYZ))
    # Ensure molecular/non-periodic treatment for alanine dipeptide.
    atoms.pbc = False
    write(str(LOCAL_XYZ), atoms)

    formula = atoms.get_chemical_formula()
    phi, psi, neigh = infer_alanine_dipeptide_phi_psi(atoms)
    phi0 = torsion_degrees(atoms, phi)
    psi0 = torsion_degrees(atoms, psi)

    print("Input validation")
    print("----------------")
    print(f"Copied input XYZ to: {LOCAL_XYZ}")
    print(f"Formula: {formula}")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Periodic boundary conditions: {atoms.pbc.tolist()}")
    print(f"Initial phi atom indices, zero-based: {phi}; value = {phi0:.3f} degrees")
    print(f"Initial psi atom indices, zero-based: {psi}; value = {psi0:.3f} degrees")

    enhanced_sampling = {
        "collective_variables": [
            {"type": "torsion", "name": "phi", "atom_indices": phi},
            {"type": "torsion", "name": "psi", "atom_indices": psi},
        ],
        "methods": [
            {
                "method": "metadynamics",
                "collective_variables": ["phi", "psi"],
                "minimum": [-180.0, -180.0],
                "maximum": [180.0, 180.0],
                "bin_width": [5.0, 5.0],
                "periodicity": [[-180.0, 180.0], [-180.0, 180.0]],
                "hill_height": 1.0,
                "hill_std": [10.0, 10.0],
                "hill_drop_frequency": 100,
                "bias_factor": 10.0,
                "confinement_force": 0.0,
                "output_frequency": 100,
                "verbose": True,
                "kinetics": False,
            }
        ],
    }

    request = MLFFWorkflowRequest(
        summarised_user_query=(
            "Run a two-dimensional enhanced-sampling calculation for alanine dipeptide "
            "using Orb-OMOL. Use the supplied alanine_c7ax.xyz structure. Sample the "
            "Ramachandran phi and psi torsions with well-tempered metadynamics in NVT at "
            "300 K, timestep 0.5 fs, 1,000 equilibration steps, 100,000 production steps, "
            "and record frames every 100 steps."
        ),
        identifier_type="from_path",
        identifier=str(LOCAL_XYZ),
        model_name="Orb-OMOL",
        charge=0,
        spin_multiplicity=1,
        relax_cell=False,
        md_ensemble="nvt_bussi",
        md_temperature=300.0,
        md_timestep=0.5,
        equilibration_steps=1000,
        md_steps=100000,
        md_loginterval=100,
        md_traj_filename="enhanced_sampling_md.traj",
        enhanced_sampling=enhanced_sampling,
    )

    # Save the exact scientific/request settings before execution.
    request_record = {
        "source_xyz": str(SOURCE_XYZ),
        "local_xyz": str(LOCAL_XYZ),
        "formula": formula,
        "n_atoms": len(atoms),
        "pbc": atoms.pbc.tolist(),
        "initial_phi_degrees": phi0,
        "initial_psi_degrees": psi0,
        "enhanced_sampling": enhanced_sampling,
        "mlff_settings": {
            "model_name": "Orb-OMOL",
            "charge": 0,
            "spin_multiplicity": 1,
            "relax_cell": False,
            "md_ensemble": "nvt_bussi",
            "md_temperature_K": 300.0,
            "md_timestep_fs": 0.5,
            "equilibration_steps": 1000,
            "production_steps": 100000,
            "record_interval_steps": 100,
        },
    }
    (WORKSPACE / "alanine_dipeptide_orb_omol_request_settings.json").write_text(
        json.dumps(request_record, indent=2)
    )

    deps = build_deps()
    result = await execute_mlff_workflow(request=request, deps=deps, run_id="alanine_dipeptide_orb_omol_2d_es")
    compact = result_to_summary(result)

    final_summary = {
        "request": request_record,
        "outputs": compact,
    }
    SUMMARY_JSON.write_text(json.dumps(final_summary, indent=2))

    print("\nWorkflow completed")
    print("------------------")
    print(f"Saved compact summary JSON: {SUMMARY_JSON}")
    pprint(compact, width=120)
    return final_summary


if __name__ == "__main__":
    asyncio.run(main())