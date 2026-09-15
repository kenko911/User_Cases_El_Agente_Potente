import struct
import numpy as np
from ase.io import read

EV_TO_KJ_MOL = 96.4853321233


def read_binary_log(log_path):
    records = []

    with open(log_path, "rb") as f:
        while True:
            header_bytes = f.read(struct.calcsize("iiddi"))
            if len(header_bytes) < struct.calcsize("iiddi"):
                break

            step, uptake, interaction_energy, total_energy, n_atoms = struct.unpack(
                "iiddi", header_bytes
            )

            records.append(
                {
                    "step": int(step),
                    "uptake": int(uptake),
                    "interaction_energy": float(interaction_energy),
                    "total_energy": float(total_energy),
                    "n_atoms": int(n_atoms),
                }
            )

    return records


def compute_gcmc_adsorption_summary(
    log_path,
    framework_cif,
    equilibration_steps=5000,
):
    """
    Compute production-averaged uptake in mmol/g and heat of adsorption
    in kJ/mol per adsorbate from log_1.0000bar.bin.

    Important:
    The binary log stores accepted moves only, so equilibration is removed
    using the MC step number, not by dropping the first N log records.
    """

    records = read_binary_log(log_path)

    if len(records) == 0:
        raise ValueError(f"No records found in {log_path}")

    production_records = [r for r in records if r["step"] > equilibration_steps]

    if len(production_records) == 0:
        raise ValueError(
            f"No production records found after step {equilibration_steps}. "
            "This can happen if no accepted moves were logged during production."
        )

    uptake = np.array([r["uptake"] for r in production_records], dtype=float)
    interaction_energy = np.array(
        [r["interaction_energy"] for r in production_records], dtype=float
    )

    avg_n_ads = float(np.mean(uptake))
    std_n_ads = float(np.std(uptake))

    avg_interaction_energy_eV = float(np.mean(interaction_energy))
    std_interaction_energy_eV = float(np.std(interaction_energy))

    atoms_frame = read(framework_cif)
    framework_mass_g_mol = float(np.sum(atoms_frame.get_masses()))

    uptake_mmol_g = avg_n_ads * 1000.0 / framework_mass_g_mol
    uptake_std_mmol_g = std_n_ads * 1000.0 / framework_mass_g_mol

    if avg_n_ads > 0:
        heat_adsorption_kj_mol_per_adsorbate = (
            avg_interaction_energy_eV / avg_n_ads * EV_TO_KJ_MOL
        )
    else:
        heat_adsorption_kj_mol_per_adsorbate = None

    return {
        "n_total_log_records": len(records),
        "n_production_log_records": len(production_records),
        "equilibration_steps_removed": equilibration_steps,

        "avg_uptake_molecules_per_cell": avg_n_ads,
        "std_uptake_molecules_per_cell": std_n_ads,

        "framework_mass_g_mol": framework_mass_g_mol,
        "uptake_mmol_g": uptake_mmol_g,
        "uptake_std_mmol_g": uptake_std_mmol_g,

        "avg_total_interaction_energy_eV": avg_interaction_energy_eV,
        "std_total_interaction_energy_eV": std_interaction_energy_eV,

        "heat_adsorption_kj_mol_per_adsorbate": heat_adsorption_kj_mol_per_adsorbate,
    }


summary = compute_gcmc_adsorption_summary(
    log_path="log_1.0000bar.bin",
    framework_cif="traj_1.0000bar.xyz",
    equilibration_steps=5000,
)

for key, value in summary.items():
    print(f"{key}: {value}")
