"""An integrator implementing the Langevin algorithm.
"""
from __future__ import annotations

__all__ = ["LangevinIntegrator"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    import cupy as xp
    test = xp.asarray([0,1,2])
    del test
    USE_CUPY = True
except:
    import numpy as xp
    USE_CUPY = False

from pydft_qmmm.utils import KB
from pydft_qmmm.utils import pluggable_method

from .integrator import Integrator

if TYPE_CHECKING:
    from pydft_qmmm import System
    from .integrator import Returns


@dataclass(frozen=True)
class LangevinIntegrator(Integrator):
    r"""An integrator implementing the Leapfrog Verlet algorithm.

    Args:
        timestep: The timestep (:math:`\mathrm{fs}`) used to perform
            integrations.
        temperature: The temperature (:math:`\mathrm{K}`) of the bath
            used to couple to the system.
        friction: The friction (:math:`\mathrm{fs^{-1}}`) experienced
            by particles in the system.
    """
    temperature: int | float
    friction: int | float

    @pluggable_method
    def integrate(self, system: System) -> Returns:
        r"""Integrate forces into new positions and velocities.

        Args:
            system: The system whose forces
                (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) and
                existing positions (:math:`\mathrm{\mathring{A}}`) and
                velocities (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) will
                be used to determine new positions and velocities.

        This method is based off of the implementation of OpenMM in
        :openmm:`ReferenceKernels.cpp`.

        Returns:
            New positions (:math:`\mathrm{\mathring{A}}`) and velocities
            (:math:`\mathrm{\mathring{A}\;fs^{-1}}`) integrated from the
            forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            and existing positions and velocities of the system.
        """
        masses = xp.asarray(system.masses.reshape((-1, 1)))
        velocities = xp.asarray(system.velocities.base)
        vel_scale = xp.exp(-self.timestep*self.friction)
        frc_scale = (
            self.timestep
            if self.friction == 0
            else (1 - vel_scale)/self.friction
        )
        noi_scale = (KB*self.temperature*(1 - vel_scale**2)*1000)**0.5
        z = xp.random.standard_normal((len(masses), 3))
        forces = xp.asarray(system.forces.base)
        momenta = velocities*masses
        momenta = (
            vel_scale*momenta
            + frc_scale*forces*(10**-4)
            + noi_scale*(10**-5)*z*masses**0.5
        )
        mask = masses[:,0] != 0
        final_velocities = xp.zeros_like(momenta)
        final_velocities[mask,:] = momenta[mask,:] / masses[mask,:]
        if USE_CUPY:
            final_velocities = xp.asnumpy(final_velocities)
        final_positions = (
            system.positions
            + self.timestep*final_velocities
        )
        return final_positions, final_velocities
