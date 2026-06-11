"""OpenMM force oracle for native Drude-SCF relaxation."""
from __future__ import annotations

__all__ = ["OpenMMDrudeForceOracle"]

from collections.abc import Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import openmm
import openmm.unit
from numpy.typing import NDArray

from pydft_qmmm.interfaces.openmm import openmm_utils

if TYPE_CHECKING:
    from pydft_qmmm.interfaces.openmm.openmm_interface import OpenMMPotential
    from .drude_data import DrudeData


class OpenMMDrudeForceOracle:
    """Read Drude particle forces from an OpenMM-backed potential."""

    def __init__(
            self,
            potential: OpenMMPotential,
            data: DrudeData,
            *,
            zero_charge_atoms: Iterable[int] | None = None,
            masked_drude_indices: Iterable[int] | None = None,
    ) -> None:
        self.potential = potential
        self.data = data
        self.zero_charge_atoms = frozenset(zero_charge_atoms or ())
        self.masked_drude_indices = frozenset(masked_drude_indices or ())

    def _forces(self) -> NDArray[np.float64]:
        """Return all OpenMM forces in kJ/mol/nm."""
        self.potential.base_context.computeVirtualSites()
        state = openmm_utils._generate_state(self.potential.base_context)
        return state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilojoule_per_mole/openmm.unit.nanometer,
        )

    @contextmanager
    def _zeroed_charges(self):
        """Temporarily zero selected NonbondedForce particle charges."""
        if not self.zero_charge_atoms:
            yield
            return
        base_system = self.potential.base_context.getSystem()
        nonbonded_forces = [
            force for force in base_system.getForces()
            if isinstance(force, openmm.NonbondedForce)
        ]
        originals = []
        try:
            for force in nonbonded_forces:
                particle_params = []
                for atom in range(force.getNumParticles()):
                    charge, sigma, epsilon = force.getParticleParameters(atom)
                    particle_params.append((atom, charge, sigma, epsilon))
                    if atom in self.zero_charge_atoms:
                        force.setParticleParameters(atom, 0.0, sigma, epsilon)
                exception_params = []
                for exception in range(force.getNumExceptions()):
                    p1, p2, chargeprod, sigma, epsilon = (
                        force.getExceptionParameters(exception)
                    )
                    exception_params.append(
                        (exception, p1, p2, chargeprod, sigma, epsilon),
                    )
                    if (
                            p1 in self.zero_charge_atoms
                            or p2 in self.zero_charge_atoms
                    ):
                        force.setExceptionParameters(
                            exception,
                            p1,
                            p2,
                            0.0,
                            sigma,
                            epsilon,
                        )
                force.updateParametersInContext(self.potential.base_context)
                originals.append((force, particle_params, exception_params))
            yield
        finally:
            for force, particle_params, exception_params in originals:
                for atom, charge, sigma, epsilon in particle_params:
                    force.setParticleParameters(atom, charge, sigma, epsilon)
                for params in exception_params:
                    force.setExceptionParameters(*params)
                force.updateParametersInContext(self.potential.base_context)

    def __call__(
            self,
            positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return forces on Drude particles in kJ/mol/nm."""
        self.potential.update_positions(positions)
        forces = self._forces()
        if self.masked_drude_indices and self.zero_charge_atoms:
            with self._zeroed_charges():
                masked_forces = self._forces()
            for atom_index in self.data.drude_indices:
                if int(atom_index) in self.masked_drude_indices:
                    forces[atom_index, :] = masked_forces[atom_index, :]
        return forces[self.data.drude_indices, :]
