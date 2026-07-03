"""Dual-thermostat extended-Lagrangian Drude dynamics."""
from __future__ import annotations

__all__ = ["DrudeLangevinIntegrator"]

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import openmm

from pydft_qmmm.utils import KB
from pydft_qmmm.utils import pluggable_method
from pydft_qmmm.plugins.drude.drude_data import extract_drude_data
from pydft_qmmm.plugins.drude.virtual_sites import VirtualSiteData
from pydft_qmmm.plugins.drude.virtual_sites import extract_virtual_sites

from .integrator import Integrator

if TYPE_CHECKING:
    from pydft_qmmm import System
    from pydft_qmmm.calculators import Calculator
    from .integrator import Returns


@dataclass(frozen=True)
class DrudeLangevinIntegrator(Integrator):
    """Integrate Drude pairs with separate center-of-mass and relative baths.

    The update follows OpenMM's ``DrudeLangevinIntegrator`` reference
    algorithm.  Temperature is in kelvin, friction coefficients are in
    inverse femtoseconds, and the random seed controls NumPy's generator.

    Metadata is bound from the OpenMM calculator by ``Simulation``.  The
    integrator can therefore be constructed before the Hamiltonian builds its
    calculator, like the other pydft-qmmm integrators.
    """

    temperature: float | int
    friction: float | int
    drude_temperature: float | int = 1.0
    drude_friction: float | int = 0.02
    max_drude_distance: float | int = 0.2
    random_seed: int | None = None
    _drude_indices: NDArray[np.int64] | None = field(default=None, init=False)
    _parent_indices: NDArray[np.int64] | None = field(default=None, init=False)
    _normal_indices: NDArray[np.int64] | None = field(default=None, init=False)
    _virtual_sites: VirtualSiteData | None = field(default=None, init=False)
    _cm_motion_frequency: int | None = field(default=None, init=False)
    _step_count: int = field(default=0, init=False)
    _defer_post_step: bool = field(default=False, init=False)
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.temperature < 0 or self.drude_temperature < 0:
            raise ValueError("Temperatures must be nonnegative.")
        object.__setattr__(self, "_rng", np.random.default_rng(self.random_seed))

    @property
    def drude_indices(self) -> NDArray[np.int64]:
        """Indices of Drude particles after calculator binding."""
        self._require_bound()
        return self._drude_indices  # type: ignore[return-value]

    @property
    def parent_indices(self) -> NDArray[np.int64]:
        """Indices of Drude parent particles after calculator binding."""
        self._require_bound()
        return self._parent_indices  # type: ignore[return-value]

    @property
    def virtual_site_indices(self) -> NDArray[np.int64]:
        """Indices of massless virtual sites after calculator binding."""
        self._require_bound()
        return self._virtual_sites.indices  # type: ignore[union-attr]

    def bind(self, calculator: Calculator) -> None:
        """Extract Drude and virtual-site metadata from an OpenMM calculator."""
        try:
            omm_system = calculator.potential.base_context.getSystem()
            context_integrator = calculator.potential.base_context.getIntegrator()
        except AttributeError as error:
            raise TypeError(
                "DrudeLangevinIntegrator requires an OpenMM-backed calculator.",
            ) from error
        if isinstance(context_integrator, openmm.DrudeSCFIntegrator):
            raise RuntimeError(
                "DrudeLangevinIntegrator requires MMHamiltonian(..., "
                "drude_engine='native') so OpenMM does not independently "
                "relax the Drude particles.",
            )
        data = extract_drude_data(omm_system)
        cm_removers = [
            force for force in omm_system.getForces()
            if isinstance(force, openmm.CMMotionRemover)
        ]
        if len(cm_removers) > 1:
            raise ValueError("Expected at most one OpenMM CMMotionRemover.")
        object.__setattr__(self, "_drude_indices", data.drude_indices)
        object.__setattr__(self, "_parent_indices", data.parent_indices)
        paired = set(data.drude_indices.tolist()) | set(data.parent_indices.tolist())
        normal_indices = np.asarray([
            index for index in range(omm_system.getNumParticles())
            if index not in paired
            and omm_system.getParticleMass(index)/openmm.unit.dalton > 0
        ], dtype=np.int64)
        object.__setattr__(self, "_normal_indices", normal_indices)
        object.__setattr__(self, "_virtual_sites", extract_virtual_sites(omm_system))
        object.__setattr__(
            self,
            "_cm_motion_frequency",
            cm_removers[0].getFrequency() if cm_removers else None,
        )

    def _require_bound(self) -> None:
        if self._drude_indices is None or self._virtual_sites is None:
            raise RuntimeError(
                "DrudeLangevinIntegrator has not been bound to a calculator.",
            )

    @staticmethod
    def _scales(
            timestep: float,
            friction: float,
            temperature: float,
    ) -> tuple[float, float, float]:
        """Return velocity, force, and thermal-noise scales."""
        if friction < 0:
            raise ValueError("Friction coefficients must be nonnegative.")
        if friction == 0:
            return 1.0, timestep, 0.0
        velocity_scale = np.exp(-timestep*friction)
        force_scale = (1.0 - velocity_scale)/friction
        # KB is J/mol/K.  1e-2 converts sqrt(kJ/mol/Da) from nm/ps
        # to Angstrom/fs, and 1e-3 converts J to kJ.
        noise_scale = np.sqrt(
            KB*1e-3*temperature*(1.0 - velocity_scale**2),
        ) * 1e-2
        return velocity_scale, force_scale, noise_scale

    def update_virtual_sites(
            self,
            positions: NDArray[np.float64],
            box: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return positions with all virtual sites recomputed."""
        self._require_bound()
        # System.box stores lattice vectors as columns; VirtualSiteData uses
        # OpenMM's row-vector convention.
        return self._virtual_sites.compute_positions(positions, box.T)  # type: ignore[union-attr]

    def defer_post_step(self) -> None:
        """Defer hard-wall and virtual-site updates until after constraints."""
        object.__setattr__(self, "_defer_post_step", True)

    def finalize_positions(
            self,
            positions: NDArray[np.float64],
            velocities: NDArray[np.float64],
            masses: NDArray[np.float64],
            box: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply post-constraint hard-wall and virtual-site operations."""
        self._apply_hard_wall(positions, velocities, masses)
        positions = self.update_virtual_sites(positions, box)
        return positions, velocities

    def _apply_hard_wall(
            self,
            positions: NDArray[np.float64],
            velocities: NDArray[np.float64],
            masses: NDArray[np.float64],
    ) -> None:
        """Apply OpenMM's Drude hard-wall bounce in Angstrom/fs units."""
        maximum = float(self.max_drude_distance)
        if maximum <= 0:
            return
        thermal_speed = np.sqrt(
            KB*1e-3*float(self.drude_temperature),
        ) * 1e-2
        dt = float(self.timestep)
        for drude, parent in zip(self.drude_indices, self.parent_indices):
            delta = positions[drude] - positions[parent]
            distance = np.linalg.norm(delta)
            if distance <= maximum:
                continue
            if distance > 2*maximum:
                raise RuntimeError(
                    "Drude particle moved too far beyond hard wall constraint.",
                )
            direction = delta/distance
            v1 = velocities[drude].copy()
            v2 = velocities[parent].copy()
            m1, m2 = masses[drude], masses[parent]
            excess = distance - maximum
            radial1 = float(v1 @ direction)
            transverse1 = v1 - radial1*direction
            if m2 == 0:
                crossing_time = min(
                    dt if radial1 == 0 else excess/abs(radial1), dt,
                )
                if radial1 != 0:
                    radial1 = -np.copysign(thermal_speed/np.sqrt(m1), radial1)
                positions[drude] += direction*(-excess + crossing_time*radial1)
                velocities[drude] = transverse1 + direction*radial1
                continue
            inv_total = 1.0/(m1 + m2)
            radial2 = float(v2 @ direction)
            transverse2 = v2 - radial2*direction
            center_velocity = (m1*radial1 + m2*radial2)*inv_total
            relative1 = radial1 - center_velocity
            relative2 = radial2 - center_velocity
            crossing_time = min(
                dt if relative1 == relative2
                else excess/abs(relative1 - relative2),
                dt,
            )
            bond_speed = thermal_speed/np.sqrt(m1)
            if relative1 != 0:
                relative1 = -np.copysign(
                    bond_speed*m2*inv_total, relative1,
                )
            if relative2 != 0:
                relative2 = -np.copysign(
                    bond_speed*m1*inv_total, relative2,
                )
            positions[drude] += direction*(
                -excess*m2*inv_total + crossing_time*relative1
            )
            positions[parent] += direction*(
                excess*m1*inv_total + crossing_time*relative2
            )
            velocities[drude] = transverse1 + direction*(relative1 + center_velocity)
            velocities[parent] = transverse2 + direction*(relative2 + center_velocity)

    @pluggable_method
    def integrate(self, system: System) -> Returns:
        """Advance one dual-Langevin Drude dynamics step."""
        self._require_bound()
        masses = np.asarray(system.masses).reshape((-1, 1))
        positions = np.array(system.positions, copy=True)
        velocities = np.array(system.velocities, copy=True)
        forces = np.asarray(system.forces)
        inverse_masses = np.zeros_like(masses)
        np.divide(1.0, masses, out=inverse_masses, where=masses != 0)

        if (
                self._cm_motion_frequency is not None
                and self._step_count % self._cm_motion_frequency == 0
        ):
            massive = masses[:, 0] > 0
            center_velocity = np.sum(
                masses[massive]*velocities[massive], axis=0,
            )/np.sum(masses[massive])
            velocities[massive] -= center_velocity

        normal = self._normal_indices
        vscale, fscale, noise = self._scales(
            float(self.timestep), float(self.friction), float(self.temperature),
        )
        if normal is not None and normal.size:
            velocities[normal] = (
                vscale*velocities[normal]
                + fscale*inverse_masses[normal]*forces[normal]*1e-4
                + noise*np.sqrt(inverse_masses[normal])
                * self._rng.standard_normal((len(normal), 3))
            )

        dvscale, dfscale, dnoise = self._scales(
            float(self.timestep),
            float(self.drude_friction),
            float(self.drude_temperature),
        )
        for drude, parent in zip(self.drude_indices, self.parent_indices):
            m1 = masses[drude, 0]
            m2 = masses[parent, 0]
            if m1 <= 0 or m2 <= 0:
                raise ValueError("Drude particles and parents must have positive mass.")
            inv_total = 1.0/(m1 + m2)
            inv_reduced = (m1 + m2)/(m1*m2)
            mass1_fraction = m1*inv_total
            mass2_fraction = m2*inv_total
            center_velocity = (
                mass1_fraction*velocities[drude]
                + mass2_fraction*velocities[parent]
            )
            relative_velocity = velocities[parent] - velocities[drude]
            center_force = forces[drude] + forces[parent]
            relative_force = (
                mass1_fraction*forces[parent]
                - mass2_fraction*forces[drude]
            )
            center_velocity = (
                vscale*center_velocity
                + fscale*inv_total*center_force*1e-4
                + noise*np.sqrt(inv_total)*self._rng.standard_normal(3)
            )
            relative_velocity = (
                dvscale*relative_velocity
                + dfscale*inv_reduced*relative_force*1e-4
                + dnoise*np.sqrt(inv_reduced)*self._rng.standard_normal(3)
            )
            velocities[drude] = center_velocity - mass2_fraction*relative_velocity
            velocities[parent] = center_velocity + mass1_fraction*relative_velocity

        massive = masses[:, 0] > 0
        positions[massive] += float(self.timestep)*velocities[massive]
        if not self._defer_post_step:
            positions, velocities = self.finalize_positions(
                positions,
                velocities,
                masses[:, 0],
                np.asarray(system.box),
            )
        object.__setattr__(self, "_step_count", self._step_count + 1)
        return positions, velocities

    @pluggable_method
    def compute_kinetic_energy(self, system: System) -> float:
        """Compute leapfrog kinetic energy while excluding virtual sites."""
        masses = np.asarray(system.masses).reshape((-1, 1))
        massive = masses[:, 0] > 0
        velocities = np.array(system.velocities, copy=True)
        velocities[massive] += (
            0.5*float(self.timestep)*np.asarray(system.forces)[massive]
            * 1e-4/masses[massive]
        )
        return float(np.sum(0.5*masses[massive]*velocities[massive]**2)*1e4)
