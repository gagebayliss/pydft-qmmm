"""OpenMM force oracle for native Drude-SCF relaxation."""
from __future__ import annotations

__all__ = ["OpenMMDrudeForceOracle"]

from typing import TYPE_CHECKING

import openmm.unit

from pydft_qmmm.interfaces.openmm import openmm_utils

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from pydft_qmmm.interfaces.openmm.openmm_interface import OpenMMPotential
    from .drude_data import DrudeData


class OpenMMDrudeForceOracle:
    """Read Drude particle forces from an OpenMM-backed potential."""

    def __init__(
            self,
            potential: OpenMMPotential,
            data: DrudeData,
    ) -> None:
        self.potential = potential
        self.data = data

    def __call__(
            self,
            positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return forces on Drude particles in kJ/mol/nm."""
        self.potential.update_positions(positions)
        self.potential.base_context.computeVirtualSites()
        state = openmm_utils._generate_state(self.potential.base_context)
        forces = state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilojoule_per_mole/openmm.unit.nanometer,
        )
        return forces[self.data.drude_indices, :]
