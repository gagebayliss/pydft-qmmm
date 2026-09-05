"""Native Drude-SCF fixed-point solver."""
from __future__ import annotations

__all__ = [
    "DrudeSCFInfo",
    "DrudeStepInfo",
    "DrudeSolver",
    "drude_relaxation_step",
]

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .drude_data import DrudeData


@dataclass(frozen=True)
class DrudeStepInfo:
    """Diagnostics from one Drude relaxation step."""
    max_force: float
    max_displacement: float
    rms_force: float = 0.0


@dataclass(frozen=True)
class DrudeSCFInfo:
    """Convergence information from a Drude relaxation."""
    iterations: int
    final_max_force: float
    final_max_displacement: float
    converged: bool
    final_rms_force: float = 0.0


def drude_relaxation_step(
        data: DrudeData,
        positions: NDArray[np.float64],
        forces: NDArray[np.float64],
        *,
        damping: float = 1.0,
) -> tuple[NDArray[np.float64], DrudeStepInfo]:
    """Apply one diagonal-Newton Drude relaxation step.

    Args:
        data: Drude oscillator metadata.
        positions: Full system positions in Angstrom.
        forces: Forces on Drude particles in kJ/mol/A.
        damping: Scalar multiplier applied to the diagonal-Newton step.

    Returns:
        Updated full-system positions in Angstrom and step diagnostics.
    """
    relaxed = np.array(positions, dtype=float, copy=True)

    displacement_ang = (
        damping
        * forces
        / data.force_constants.reshape((-1, 1))
    )
    relaxed[data.drude_indices, :] += displacement_ang

    force_norms = np.linalg.norm(forces, axis=1)
    displacement_norms = np.linalg.norm(displacement_ang, axis=1)
    return relaxed, DrudeStepInfo(
        max_force=float(np.max(force_norms, initial=0.0)),
        max_displacement=float(np.max(displacement_norms, initial=0.0)),
        rms_force=float(np.sqrt(np.mean(forces**2))) if forces.size else 0.0,
    )


class DrudeSolver:
    """Relax Drude oscillator positions at fixed real-atom positions.

    Args:
        data: Drude oscillator metadata.
        calculator: Callable returning results containing forces on system in
            kJ/mol/A.
        force_tolerance: RMS Cartesian Drude-force component required for
            convergence, in kJ/mol/A.
        displacement_tolerance: Optional maximum Drude-particle displacement
            in one iteration required for convergence, in Angstrom.  This is
            disabled by default to match OpenMM.
        max_iterations: Maximum fixed-point iterations.
        damping: Scalar multiplier applied to each diagonal-Newton step.
        stagnation_ratio: If provided, accept the current step when the sum
            of squared Drude forces exceeds this fraction of its value in the
            preceding iteration.  OpenMM's Drude-SCF minimizer uses 0.9.
        algorithm: ``"diagonal"`` for the original fixed-point update or
    """

    def __init__(
            self,
            data: DrudeData,
            calculator: Callable[...,NDArray[np.float64]],
            *,
            force_tolerance: float = 10.0, #angstrom
            displacement_tolerance: float | None = None,
            max_iterations: int = 50,
            damping: float = 1.0,
            stagnation_ratio: float | None = 0.9,
            algorithm: str = "diagonal",
    ) -> None:
        self.data = data
        self.calculator = calculator
        self.force_tolerance = force_tolerance
        self.displacement_tolerance = displacement_tolerance
        self.max_iterations = max_iterations
        self.damping = damping
        self.stagnation_ratio = stagnation_ratio
        if algorithm not in {"diagonal"}:
            raise ValueError(f"Unknown Drude relaxation algorithm: {algorithm}")
        self.algorithm = algorithm

    def step(
            self,
            positions: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], DrudeStepInfo]:
        """Apply one Drude relaxation step.

        Args:
            positions: Full system positions in Angstrom.

        Returns:
            Updated full-system positions in Angstrom and step
            diagnostics.
        """
        self.calculator.system.positions[:] = positions
        results = self.calculator.calculate()
        forces = results.forces[self.data.drude_indices]
        return drude_relaxation_step(
            self.data,
            positions,
            forces,
            damping=self.damping,
        )

    def relax(
            self,
            positions: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], DrudeSCFInfo]:
        """Relax Drude positions.

        Args:
            positions: Full system positions in Angstrom.

        Returns:
            Relaxed full-system positions in Angstrom and convergence
            diagnostics.
        """
        relaxed = np.array(positions, dtype=float, copy=True)
        final_max_force = np.inf
        final_max_displacement = np.inf
        final_rms_force = np.inf
        previous_force_squared = np.inf
        for iteration in range(1, self.max_iterations + 1):
            self.calculator.positions[:] = relaxed
            results = self.calculator.calculate()
            forces = results.forces[self.data.drude_indices]
            stepped, step_info = drude_relaxation_step(
                self.data,
                relaxed,
                forces,
                damping=self.damping,
            )

            final_max_force = step_info.max_force
            final_max_displacement = step_info.max_displacement
            final_rms_force = step_info.rms_force

            if (
                final_rms_force <= self.force_tolerance
                or (
                    self.displacement_tolerance is not None
                    and final_max_displacement
                    <= self.displacement_tolerance
                ) or (
                self.stagnation_ratio is not None
                and iteration > 1
                and force_squared
                > self.stagnation_ratio*previous_force_squared
                )
            ):
                return stepped, DrudeSCFInfo(
                    iterations=iteration,
                    final_max_force=final_max_force,
                    final_max_displacement=final_max_displacement,
                    converged=True,
                    final_rms_force=final_rms_force,
                )

            force_squared = float(np.sum(forces*forces))
            previous_force_squared = force_squared
            relaxed = stepped

        return stepped, DrudeSCFInfo(
            iterations=self.max_iterations,
            final_max_force=final_max_force,
            final_max_displacement=final_max_displacement,
            converged=False,
            final_rms_force=final_rms_force,
        )
