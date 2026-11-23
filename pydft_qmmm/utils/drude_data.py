"""Drude oscillator metadata extracted from OpenMM systems."""
from __future__ import annotations

__all__ = ["DrudeData", "extract_drude_data"]

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import openmm
from .virtual_sites import _extract_drude_sites


@dataclass(frozen=True)
class DrudeData:
    """Metadata needed to relax Drude oscillator positions.

    Attributes:
        drude_indices: Particle indices for Drude particles.
        parent_indices: Particle indices for each Drude parent.
        charges: Drude particle charges in elementary charge.
        polarizabilities: Drude polarizabilities in angstrom^3.
        force_constants: Isotropic harmonic spring constants in
            kJ/mol/angstrom^2.
    """
    drude_indices: NDArray[np.int64]
    parent_indices: NDArray[np.int64]
    charges: NDArray[np.float64]
    polarizabilities: NDArray[np.float64]
    force_constants: NDArray[np.float64]

    def __len__(self) -> int:
        """Get the number of Drude oscillators."""
        return len(self.drude_indices)


def extract_drude_data(omm_system: openmm.System) -> DrudeData | None:
    """Extract Drude oscillator metadata from an OpenMM system.

    Args:
        omm_system: The OpenMM system containing one DrudeForce.

    Returns:
        Drude oscillator metadata, or None if no DrudeForce is present.
    """
    sites = _extract_drude_sites(omm_system)
    if not sites:
        return None
    charges, polarizabilities, force_constants = np.asarray(
        [site.parameters for site in sites],
    ).T
    return DrudeData(
        np.array([site.index for site in sites], dtype=np.int64),
        np.array([site.parents[0] for site in sites], dtype=np.int64),
        charges, polarizabilities, force_constants,
    )
