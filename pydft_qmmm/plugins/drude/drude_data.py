"""Drude oscillator metadata extracted from OpenMM systems."""
from __future__ import annotations

__all__ = ["DrudeData", "extract_drude_data"]

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import openmm
import openmm.unit

ONE_4PI_EPS0 = 138.93545764438198


@dataclass(frozen=True)
class DrudeData:
    """Metadata needed to relax Drude oscillator positions.

    Attributes:
        drude_indices: Particle indices for Drude particles.
        parent_indices: Particle indices for each Drude parent.
        charges: Drude particle charges in elementary charge.
        polarizabilities: Drude polarizabilities in nm^3.
        force_constants: Harmonic spring constants in kJ/mol/nm^2.
    """
    drude_indices: NDArray[np.int64]
    parent_indices: NDArray[np.int64]
    charges: NDArray[np.float64]
    polarizabilities: NDArray[np.float64]
    force_constants: NDArray[np.float64]

    def __len__(self) -> int:
        """Get the number of Drude oscillators."""
        return len(self.drude_indices)


def extract_drude_data(omm_system: openmm.System) -> DrudeData:
    """Extract Drude oscillator metadata from an OpenMM system.

    Args:
        omm_system: The OpenMM system containing one DrudeForce.

    Returns:
        Drude oscillator metadata for fixed-point relaxation.
    """
    drude_forces = [
        force for force in omm_system.getForces()
        if isinstance(force, openmm.DrudeForce)
    ]
    if not drude_forces:
        raise ValueError("The OpenMM system does not contain a DrudeForce.")
    if len(drude_forces) > 1:
        raise ValueError("Expected one DrudeForce in the OpenMM system.")
    drude_force = drude_forces[0]
    drude_indices = []
    parent_indices = []
    charges = []
    polarizabilities = []
    force_constants = []
    for i in range(drude_force.getNumParticles()):
        particle, parent, *_rest, charge, polarizability, _a12, _a34 = (
            drude_force.getParticleParameters(i)
        )
        charge_e = charge / openmm.unit.elementary_charge
        alpha_nm3 = polarizability / openmm.unit.nanometer**3
        drude_indices.append(int(particle))
        parent_indices.append(int(parent))
        charges.append(float(charge_e))
        polarizabilities.append(float(alpha_nm3))
        force_constants.append(float(ONE_4PI_EPS0 * charge_e**2 / alpha_nm3))
    return DrudeData(
        drude_indices=np.array(drude_indices, dtype=np.int64),
        parent_indices=np.array(parent_indices, dtype=np.int64),
        charges=np.array(charges, dtype=float),
        polarizabilities=np.array(polarizabilities, dtype=float),
        force_constants=np.array(force_constants, dtype=float),
    )
