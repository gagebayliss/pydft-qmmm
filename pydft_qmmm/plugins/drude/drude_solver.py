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


@dataclass(frozen=True)
class DrudeSCFInfo:
    """Convergence information from a Drude relaxation."""
    iterations: int
    final_max_force: float
    final_max_displacement: float
    converged: bool


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
        forces: Forces on Drude particles in kJ/mol/nm.
        damping: Scalar multiplier applied to the diagonal-Newton step.

    Returns:
        Updated full-system positions in Angstrom and step diagnostics.
    """
    relaxed = np.array(positions, dtype=float, copy=True)
    displacement_nm = (
        damping
        * forces
        / data.force_constants.reshape((-1, 1))
    )
    displacement_ang = 10.0*displacement_nm
    relaxed[data.drude_indices, :] += displacement_ang
    force_norms = np.linalg.norm(forces, axis=1)
    displacement_norms = np.linalg.norm(displacement_ang, axis=1)
    return relaxed, DrudeStepInfo(
        max_force=float(np.max(force_norms, initial=0.0)),
        max_displacement=float(np.max(displacement_norms, initial=0.0)),
    )


class DrudeSolver:
    """Relax Drude oscillator positions at fixed real-atom positions.

    Args:
        data: Drude oscillator metadata.
        force_oracle: Callable returning forces on Drude particles in
            kJ/mol/nm for a full positions array in Angstrom.
        force_tolerance: Maximum Drude-particle force norm required for
            convergence, in kJ/mol/nm.
        displacement_tolerance: Maximum Drude-particle displacement in
            one iteration required for convergence, in Angstrom.
        max_iterations: Maximum fixed-point iterations.
        damping: Scalar multiplier applied to each diagonal-Newton step.
        stagnation_ratio: If provided, accept the current step when the sum
            of squared Drude forces exceeds this fraction of its value in the
            preceding iteration.  OpenMM's Drude-SCF minimizer uses 0.9.
    """

    def __init__(
            self,
            data: DrudeData,
            force_oracle: Callable[[NDArray[np.float64]], NDArray[np.float64]],
            *,
            force_tolerance: float = 5e-2,
            displacement_tolerance: float = 1e-6,
            max_iterations: int = 100,
            damping: float = 1.0,
            stagnation_ratio: float | None = None,
    ) -> None:
        self.data = data
        self.force_oracle = force_oracle
        self.force_tolerance = force_tolerance
        self.displacement_tolerance = displacement_tolerance
        self.max_iterations = max_iterations
        self.damping = damping
        self.stagnation_ratio = stagnation_ratio

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
        forces = self.force_oracle(positions)
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
        previous_force_squared = np.inf
        for iteration in range(1, self.max_iterations + 1):
            forces = self.force_oracle(relaxed)
            stepped, step_info = drude_relaxation_step(
                self.data,
                relaxed,
                forces,
                damping=self.damping,
            )
            final_max_force = step_info.max_force
            final_max_displacement = step_info.max_displacement
            if (
                    final_max_force < self.force_tolerance
                    or final_max_displacement < self.displacement_tolerance
            ):
                return relaxed, DrudeSCFInfo(
                    iterations=iteration,
                    final_max_force=final_max_force,
                    final_max_displacement=final_max_displacement,
                    converged=True,
                )
            force_squared = float(np.sum(forces*forces))
            if (
                    self.stagnation_ratio is not None
                    and iteration > 1
                    and force_squared
                    > self.stagnation_ratio*previous_force_squared
            ):
                return stepped, DrudeSCFInfo(
                    iterations=iteration,
                    final_max_force=final_max_force,
                    final_max_displacement=final_max_displacement,
                    converged=True,
                )
            previous_force_squared = force_squared
            relaxed = stepped
        raise RuntimeError(
            "Drude SCF did not converge after "
            f"{self.max_iterations} iterations; "
            f"max force = {final_max_force:.6g} kJ/mol/nm, "
            f"max displacement = {final_max_displacement:.6g} A",
        )
