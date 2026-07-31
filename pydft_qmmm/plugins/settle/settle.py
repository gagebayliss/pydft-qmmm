"""Plugin for applying SETTLE to select residues after integration.
"""
from __future__ import annotations

__all__ = ["SETTLE"]

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .settle_utils import settle_positions
from .settle_utils import settle_velocities
from pydft_qmmm.integrators import IntegratorPlugin

if TYPE_CHECKING:
    from pydft_qmmm.integrators import Integrator
    from pydft_qmmm.integrators import Returns
    from pydft_qmmm import System


class SETTLE(IntegratorPlugin):
    r"""Apply the SETTLE algorithm to water residues after integration.

    This plugin is based off of the implementation of OpenMM in
    :openmm:`SimTKReference/ReferenceSETTLEAlgorithm.cpp`.

    Args:
        query: The VMD-like selection query which should correspond to
            water residues.
        oh_distance: The distance between the oxygen and hydrogens
            (:math:`\mathrm{\mathring{A}}`).
        hh_distance: The distance between the hydrogens
            (:math:`\mathrm{\mathring{A}}`).
    """

    def __init__(
            self,
            query: str = "resname HOH",
            oh_distance: float | int = 1.,
            hh_distance: float | int = 1.632981,
    ) -> None:
        self.query = "(" + query + ") and not subsystem I"
        self.oh_distance = oh_distance
        self.hh_distance = hh_distance
        self._residue_cache: tuple[int, list[list[int]]] | None = None

    def modify(self, integrator: Integrator) -> None:
        """Register SETTLE and defer Drude post-step operations."""
        super().modify(integrator)
        defer_post_step = getattr(integrator, "defer_post_step", None)
        if defer_post_step is not None:
            defer_post_step()

    def constrain_velocities(self, system: System) -> NDArray[np.float64]:
        """Apply the SETTLE algorithm to system velocities.

        Args:
            system: The system whose velocities will be SETTLEd.

        Returns:
            New velocities which result from the application of the
            SETTLE algorithm to system velocities.
        """
        residues = self._get_hoh_residues(system)
        velocities = settle_velocities(
            residues,
            system.positions,
            system.velocities,
            system.masses,
        )
        return velocities

    def _get_hoh_residues(
            self,
            system: System,
    ) -> list[list[int]]:
        """Get the water residues from the system.

        Args:
            system: The system containing selected water residues.

        Returns:
            A list of list of atom indices, representing the all water
            residues in the system.
        """
        if self._residue_cache is not None:
            system_id, residues = self._residue_cache
            if system_id == id(system):
                return residues
        residue_indices = np.unique(
            np.asarray(system.residues)[sorted(system.select(self.query))],
        )
        residue_map = system.residue_map
        drudes = frozenset(getattr(self.integrator, "drude_indices", ()))
        virtual_sites = frozenset(
            getattr(self.integrator, "virtual_site_indices", ()),
        )
        excluded = drudes | virtual_sites
        hoh_residues = []
        for residue_index in residue_indices:
            atoms = [
                atom for atom in residue_map[residue_index]
                if atom not in excluded
            ]
            if len(atoms) != 3:
                raise ValueError(
                    "SETTLE residues must contain exactly three non-Drude, "
                    "non-virtual-site atoms.",
                )
            # settle_positions expects the central, heavy oxygen first.
            oxygen = max(atoms, key=lambda atom: system.masses[atom])
            hoh_residues.append([oxygen] + sorted(set(atoms) - {oxygen}))
        self._residue_cache = (id(system), hoh_residues)
        return hoh_residues

    def _modify_integrate(
            self,
            integrate: Callable[[System], Returns],
    ) -> Callable[[System], Returns]:
        """Modify the integrate routine to perform SETTLE afterward.

        Args:
            integrate: The integration routine to modify.

        Returns:
            The modified integration routine which implements the SETTLE
            algorithm after integration.
        """
        def inner(system: System) -> Returns:
            positions, velocities = integrate(system)
            residues = self._get_hoh_residues(system)
            if residues:
                positions = settle_positions(
                    residues,
                    system.positions,
                    positions,
                    system.masses,
                    self.oh_distance,
                    self.hh_distance,
                )
                velocities[residues, :] = (
                    (
                        positions[residues, :]
                        - system.positions[residues, :]
                    ) / self.integrator.timestep
                )
            finalize_positions = getattr(
                self.integrator, "finalize_positions", None,
            )
            if finalize_positions is not None:
                positions, velocities = finalize_positions(
                    positions,
                    velocities,
                    np.asarray(system.masses),
                    np.asarray(system.box),
                )
            return positions, velocities
        return inner

    def _modify_compute_kinetic_energy(
            self,
            compute_kinetic_energy: Callable[[System], float],
    ) -> Callable[[System], float]:
        """Modify the kinetic energy computation to use SETTLE.

        Args:
            compute_kinetic_energy: The kinetic energy routine to
                modify.

        Returns:
            The modified kinetic energy routine which applies the SETTLE
            algorithm to velocities.
        """
        def inner(system: System) -> float:
            masses = system.masses.reshape(-1, 1)
            forces = np.asarray(system.forces)
            physical_forces = getattr(self.integrator, "physical_forces", None)
            if physical_forces is not None:
                forces = physical_forces(
                    np.asarray(system.positions),
                    forces,
                    np.asarray(system.box),
                )
            accelerations = np.zeros_like(forces)
            np.divide(
                forces*(10**-4),
                masses,
                out=accelerations,
                where=masses != 0,
            )
            velocities = (
                system.velocities
                + 0.5*self.integrator.timestep*accelerations
            )
            excluded = getattr(
                self.integrator, "kinetic_exclusion_indices", (),
            )
            if len(excluded):
                velocities[excluded] = 0.0
            residues = self._get_hoh_residues(system)
            if residues:
                velocities = settle_velocities(
                    residues,
                    system.positions,
                    velocities,
                    system.masses,
                )
            kinetic_energy = (
                np.sum(0.5*masses*(velocities)**2)
                * (10**4)
            )
            return kinetic_energy
        return inner
