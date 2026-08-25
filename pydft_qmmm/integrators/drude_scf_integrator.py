"""Nuclear equations of motion for self-consistent Drude dynamics."""
from __future__ import annotations

__all__ = ["DrudeSCFIntegrator"]

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import openmm

from pydft_qmmm.plugins.drude.drude_data import extract_drude_data
from pydft_qmmm.system import VirtualSiteData
from pydft_qmmm.system import extract_virtual_sites
from pydft_qmmm.utils import pluggable_method

from .integrator import Integrator

if TYPE_CHECKING:
    from typing import Any
    from pydft_qmmm import System
    from pydft_qmmm.calculators import Calculator
    from .integrator import Returns


class _PropagationSystem:
    """Array-isolated system view used by a wrapped base integrator."""

    def __init__(
            self,
            system: System,
            excluded: NDArray[np.int64],
            forces: NDArray[np.float64] | None = None,
    ) -> None:
        self._system = system
        self.positions = np.array(system.positions, copy=True)
        self.velocities = np.array(system.velocities, copy=True)
        self.forces = np.array(
            system.forces if forces is None else forces,
            copy=True,
        )
        self.masses = np.array(system.masses, copy=True)
        self.velocities[excluded] = 0.0
        self.forces[excluded] = 0.0
        # Generic pydft-qmmm integrators divide by every mass.  Give massless
        # virtual sites an inert placeholder mass in this isolated view.
        self.masses[self.masses == 0] = 1.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._system, name)


class DrudeSCFIntegrator(Integrator):
    """Propagate nuclei while leaving Drude relaxation to the calculator.

    This wrapper deliberately separates the two operations in SCF dynamics:

    1. ``integrate()`` advances only physical nuclei with ``base_integrator``.
    2. The subsequent calculator call relaxes Drudes, either with the native
       :class:`~pydft_qmmm.plugins.drude.DrudeSCF` plugin for MM or with a
       QM/MM plugin that interleaves Drude updates with the electronic SCF.

    Drude particles and virtual sites are excluded from EOM propagation and
    kinetic energy.  Virtual-site coordinates are reconstructed after
    constraints; Drude coordinates are not modified by this class.
    """

    def __init__(self, base_integrator: Integrator) -> None:
        object.__setattr__(self, "timestep", base_integrator.timestep)
        object.__setattr__(self, "_plugins", [])
        object.__setattr__(self, "base_integrator", base_integrator)
        object.__setattr__(self, "_drude_indices", None)
        object.__setattr__(self, "_virtual_sites", None)
        object.__setattr__(self, "_mm_potential", None)
        object.__setattr__(self, "_defer_post_step", False)

    @property
    def drude_indices(self) -> NDArray[np.int64]:
        """Indices of SCF-relaxed Drude particles."""
        self._require_bound()
        return self._drude_indices

    @property
    def virtual_site_indices(self) -> NDArray[np.int64]:
        """Indices of derived massless particles."""
        self._require_bound()
        return self._virtual_sites.indices

    @property
    def kinetic_exclusion_indices(self) -> NDArray[np.int64]:
        """Particles excluded from physical nuclear kinetic energy."""
        return np.union1d(self.drude_indices, self.virtual_site_indices)

    def _find_openmm_system(
            self,
            calculator: Calculator,
    ) -> tuple[openmm.System, openmm.Integrator, Any]:
        """Locate an OpenMM potential in a simple or composite calculator."""
        calculators = getattr(calculator, "calculators", (calculator,))
        for component in calculators:
            potential = getattr(component, "potential", None)
            context = getattr(potential, "base_context", None)
            if context is not None:
                omm_system = context.getSystem()
                if any(
                    isinstance(force, openmm.DrudeForce)
                    for force in omm_system.getForces()
                ):
                    return omm_system, context.getIntegrator(), potential
        raise TypeError(
            "DrudeSCFIntegrator requires an OpenMM Drude potential in the "
            "calculator.",
        )

    def bind(self, calculator: Calculator) -> None:
        """Bind Drude and virtual-site topology from the MM calculator."""
        omm_system, context_integrator, potential = self._find_openmm_system(
            calculator,
        )
        if isinstance(context_integrator, openmm.DrudeSCFIntegrator):
            raise RuntimeError(
                "DrudeSCFIntegrator requires MMHamiltonian(..., "
                "drude_engine='native') so relaxation is owned by the "
                "pydft-qmmm/QM-SCF plugin.",
            )
        data = extract_drude_data(omm_system)
        object.__setattr__(self, "_drude_indices", data.drude_indices)
        object.__setattr__(self, "_virtual_sites", extract_virtual_sites(omm_system))
        object.__setattr__(self, "_mm_potential", potential)

    def _require_bound(self) -> None:
        if self._drude_indices is None or self._virtual_sites is None:
            raise RuntimeError(
                "DrudeSCFIntegrator has not been bound to a calculator.",
            )

    def defer_post_step(self) -> None:
        """Let a constraint plugin perform virtual-site reconstruction."""
        object.__setattr__(self, "_defer_post_step", True)

    def update_virtual_sites(
            self,
            positions: NDArray[np.float64],
            box: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Reconstruct virtual-site coordinates without relaxing Drudes."""
        self._require_bound()
        return self._virtual_sites.compute_positions(positions, box.T)

    def finalize_positions(
            self,
            positions: NDArray[np.float64],
            velocities: NDArray[np.float64],
            masses: NDArray[np.float64],
            box: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Perform post-constraint virtual-site reconstruction only."""
        del masses
        return self.update_virtual_sites(positions, box), velocities

    def physical_forces(
            self,
            positions: NDArray[np.float64],
            forces: NDArray[np.float64],
            box: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return forces on the independent propagated coordinates."""
        self._require_bound()
        indices = self.virtual_site_indices
        base_forces = getattr(
            self._mm_potential,
            "_last_base_forces",
            forces,
        )
        residual = np.zeros_like(forces)
        residual[indices] = forces[indices] - base_forces[indices]
        redistributed = self._virtual_sites.redistribute_forces(
            positions,
            residual,
            box.T,
        )
        result = np.array(forces, copy=True)
        result[indices] = 0.0
        result += redistributed
        return result

    @pluggable_method
    def integrate(self, system: System) -> Returns:
        """Advance physical nuclei by one step and freeze auxiliary sites."""
        excluded = self.kinetic_exclusion_indices
        forces = self.physical_forces(
            np.asarray(system.positions),
            np.asarray(system.forces),
            np.asarray(system.box),
        )
        propagation_system = _PropagationSystem(system, excluded, forces)
        positions, velocities = self.base_integrator.integrate(
            propagation_system,
        )
        positions = np.asarray(positions)
        velocities = np.asarray(velocities)
        positions[excluded] = np.asarray(system.positions)[excluded]
        velocities[excluded] = 0.0
        if not self._defer_post_step:
            positions, velocities = self.finalize_positions(
                positions,
                velocities,
                np.asarray(system.masses),
                np.asarray(system.box),
            )
        return positions, velocities

    @pluggable_method
    def compute_kinetic_energy(self, system: System) -> float:
        """Compute physical nuclear kinetic energy with the base integrator."""
        propagation_system = _PropagationSystem(
            system,
            self.kinetic_exclusion_indices,
            self.physical_forces(
                np.asarray(system.positions),
                np.asarray(system.forces),
                np.asarray(system.box),
            ),
        )
        return self.base_integrator.compute_kinetic_energy(propagation_system)
