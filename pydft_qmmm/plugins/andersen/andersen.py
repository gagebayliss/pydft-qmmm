"""Stochastic velocity collisions with a thermal bath."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.integrators import IntegratorPlugin
from pydft_qmmm.utils import KB

if TYPE_CHECKING:
    from pydft_qmmm import System
    from pydft_qmmm.integrators import Returns

__all__ = ["AndersenThermostat"]


class AndersenThermostat(IntegratorPlugin):
    """Resample selected massive particles before integration.

    Args:
        temperature: Bath temperature in K (nonnegative).
        collision_frequency: Collision rate in inverse fs (nonnegative).
        query: Atom selection to thermostat.
        seed: Optional seed for an independent NumPy random generator.

    Collisions update the input velocities temporarily, allowing position
    and velocity constraints to act on the resulting integration step.
    Massless particles are never resampled.
    """

    def __init__(self, temperature: float, collision_frequency: float,
                 query: str = "all", seed: int | None = None) -> None:
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not np.isfinite(collision_frequency) or collision_frequency < 0:
            raise ValueError(
                "collision_frequency must be finite and nonnegative",
            )
        self.temperature = temperature
        self.collision_frequency = collision_frequency
        self.query = query
        self._rng = np.random.default_rng(seed)

    def _modify_integrate(
        self, integrate: Callable[[System], Returns],
    ) -> Callable[[System], Returns]:
        """Wrap integration with independent particle collisions."""
        def inner(system: System) -> Returns:
            timestep = self.integrator.timestep
            if not np.isfinite(timestep) or timestep <= 0:
                raise ValueError("timestep must be finite and positive")
            atoms = np.array(sorted(system.select(self.query)), dtype=int)
            atoms = atoms[np.asarray(system.masses)[atoms] > 0]
            probability = -np.expm1(-self.collision_frequency * timestep)
            atoms = atoms[self._rng.random(len(atoms)) < probability]
            original = np.array(system.velocities, copy=True)
            velocities = original.copy()
            # KB is J/(mol K); velocities are angstrom/fs and masses AMU.
            sigma = np.sqrt(KB * self.temperature * 1.e-7
                            / np.asarray(system.masses)[atoms])
            velocities[atoms] = (
                self._rng.normal(size=(len(atoms), 3)) * sigma[:, None]
            )
            try:
                system.velocities[:] = velocities
                positions, final_velocities = integrate(system)
                return (
                    np.array(positions, copy=True),
                    np.array(final_velocities, copy=True),
                )
            finally:
                system.velocities[:] = original
        return inner
