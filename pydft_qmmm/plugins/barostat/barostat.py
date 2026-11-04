"""Monte Carlo volume moves using calculator energies."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.calculators import Calculator
from pydft_qmmm.integrators import IntegratorPlugin
from pydft_qmmm.utils import KB

if TYPE_CHECKING:
    from pydft_qmmm import System
    from pydft_qmmm.integrators import Returns

__all__ = ["MonteCarloBarostat"]

# bar * angstrom**3 -> kJ/mol
_PV_CONVERSION = 6.02214076e-5


class MonteCarloBarostat(IntegratorPlugin):
    """Attempt isotropic volume moves before integration.

    Args:
        calculator: Calculator bound to the integrated system, including
            all energy terms needed for acceptance.
        pressure: External pressure in bar.
        temperature: Positive bath temperature in K. A separate thermostat
            must maintain this temperature.
        interval: Number of integration calls between attempts.
        volume_step: Maximum absolute volume change in angstrom cubed.
        seed: Optional seed for an independent NumPy random generator.

    Each residue is treated as one molecule and translated by its centroid;
    internal coordinates and velocities are preserved. Residues must contain
    whole, unwrapped molecules, including their virtual sites. Fixed atoms
    and molecules spanning multiple residues are not supported. Register
    this plugin before other integrator plugins. Calculators must support
    box updates and energy-only evaluations without irreversible side
    effects.
    """

    def __init__(self, calculator: Calculator, pressure: float,
                 temperature: float, interval: int = 25,
                 volume_step: float = 10., seed: int | None = None) -> None:
        if not np.isfinite(pressure):
            raise ValueError("pressure must be finite")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if (isinstance(interval, bool) or not isinstance(interval, int)
                or interval <= 0):
            raise ValueError("interval must be a positive integer")
        if not np.isfinite(volume_step) or volume_step <= 0:
            raise ValueError("volume_step must be finite and positive")
        self.calculator = calculator
        self.pressure = pressure
        self.temperature = temperature
        self.interval = interval
        self.volume_step = volume_step
        self._rng = np.random.default_rng(seed)
        self._step = 0

    def _attempt(self, system: System) -> None:
        """Evaluate a symmetric volume proposal, restoring failed trials."""
        box = np.array(system.box, copy=True)
        volume = float(np.linalg.det(box))
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError(
                "barostat requires a finite, positive periodic volume",
            )
        new_volume = volume + self._rng.uniform(
            -self.volume_step, self.volume_step,
        )
        if new_volume <= 0:
            return
        positions = np.array(system.positions, copy=True)
        velocities = np.array(system.velocities, copy=True)
        forces = np.array(system.forces, copy=True)
        accepted = False
        try:
            energy = float(self.calculator.calculate(False, False).energy)
            # Calculator plugins may wrap/center coordinates during evaluation.
            trial = np.array(system.positions, copy=True)
            scale = (new_volume / volume)**(1. / 3.)
            for residue in system.residue_map.values():
                atoms = sorted(residue)
                trial[atoms] += (scale - 1.) * trial[atoms].mean(axis=0)
            system.box[:] = box * scale
            system.positions[:] = trial
            trial_energy = float(self.calculator.calculate(False, False).energy)
            thermal_energy = KB * self.temperature / 1000.
            work = (trial_energy - energy
                    + self.pressure * (new_volume - volume) * _PV_CONVERSION)
            log_probability = (
                -work / thermal_energy
                + len(system.residue_map) * np.log(new_volume / volume)
            )
            if not np.isfinite(energy) or not np.isfinite(trial_energy):
                raise ValueError("calculator returned a nonfinite barostat energy")
            log_draw = np.log(max(self._rng.random(), np.finfo(float).tiny))
            if log_draw < log_probability:
                system.forces[:] = self.calculator.calculate(
                    True, False,
                ).forces
                accepted = True
        finally:
            if not accepted:
                system.box[:] = box
                system.positions[:] = positions
                system.velocities[:] = velocities
                system.forces[:] = forces

    def _modify_integrate(
        self, integrate: Callable[[System], Returns],
    ) -> Callable[[System], Returns]:
        """Attempt pressure coupling at the requested interval."""
        def inner(system: System) -> Returns:
            if system is not self.calculator.system:
                raise ValueError(
                    "barostat calculator must use the integrated system",
                )
            self._step += 1
            if self._step % self.interval == 0:
                self._attempt(system)
            return integrate(system)
        return inner
