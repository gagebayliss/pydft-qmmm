"""Native Drude-SCF fixed-point solver."""
from __future__ import annotations

__all__ = [
    "DrudeSCFInfo",
    "DrudeStepInfo",
    "DrudeCGState",
    "DrudeSolver",
    "drude_conjugate_gradient_step",
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


@dataclass(frozen=True)
class DrudeCGState:
    """History required for a preconditioned conjugate-gradient step."""
    forces: NDArray[np.float64]
    preconditioned_forces: NDArray[np.float64]
    direction_nm: NDArray[np.float64]


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
        # OpenMM defines the residual as the RMS over Cartesian force
        # components, rather than the RMS of the per-particle force norms.
        rms_force=float(np.sqrt(np.mean(forces**2))) if forces.size else 0.0,
    )


def drude_conjugate_gradient_step(
        data: DrudeData,
        positions: NDArray[np.float64],
        force_oracle: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        *,
        state: DrudeCGState | None = None,
        damping: float = 1.0,
        max_backtracks: int = 8,
) -> tuple[NDArray[np.float64], DrudeStepInfo, DrudeCGState]:
    """Apply one diagonally preconditioned conjugate-gradient update.

    The Drude spring constants form the preconditioner.  A force probe along
    the conjugate direction supplies the directional curvature and hence the
    CG step length.  For nonlinear or changing QM/MM fields, the direction is
    restarted when it ceases to be downhill and the step is backtracked when
    the residual force grows.
    """
    current = np.asarray(positions, dtype=float)
    forces = np.asarray(force_oracle(current), dtype=float)
    preconditioned = forces / data.force_constants.reshape((-1, 1))
    direction = np.array(preconditioned, copy=True)

    if state is not None and state.forces.shape == forces.shape:
        denominator = float(np.sum(
            state.forces * state.preconditioned_forces,
        ))
        numerator = float(np.sum(
            forces * (preconditioned - state.preconditioned_forces),
        ))
        beta = numerator / denominator if denominator > 0.0 else 0.0
        if np.isfinite(beta) and beta > 0.0:
            direction += beta * state.direction_nm
        if float(np.sum(forces * direction)) <= 0.0:
            direction = np.array(preconditioned, copy=True)

    probed_direction = np.array(direction, copy=True)
    probe = np.array(current, copy=True)
    probe[data.drude_indices, :] += 10.0 * direction
    probe_forces = np.asarray(force_oracle(probe), dtype=float)
    curvature_direction = -(probe_forces - forces)
    curvature = float(np.sum(direction * curvature_direction))
    residual_product = float(np.sum(forces * preconditioned))
    if curvature > 0.0 and np.isfinite(curvature):
        step_length = damping * residual_product / curvature
    else:
        direction = np.array(preconditioned, copy=True)
        step_length = damping
    if not np.isfinite(step_length) or step_length <= 0.0:
        direction = np.array(preconditioned, copy=True)
        step_length = damping

    initial_force_squared = float(np.sum(forces * forces))
    accepted = None
    accepted_forces = None
    for _ in range(max_backtracks + 1):
        trial = np.array(current, copy=True)
        trial[data.drude_indices, :] += 10.0 * step_length * direction
        if (
                np.isclose(step_length, 1.0)
                and np.array_equal(direction, probed_direction)
        ):
            trial_forces = probe_forces
        else:
            trial_forces = np.asarray(force_oracle(trial), dtype=float)
        if float(np.sum(trial_forces * trial_forces)) <= initial_force_squared:
            accepted = trial
            accepted_forces = trial_forces
            break
        step_length *= 0.5
    if accepted is None or accepted_forces is None:
        direction = np.array(preconditioned, copy=True)
        step_length = damping * 0.5**max_backtracks
        accepted = np.array(current, copy=True)
        accepted[data.drude_indices, :] += 10.0 * step_length * direction
        accepted_forces = np.asarray(force_oracle(accepted), dtype=float)

    displacement_ang = 10.0 * step_length * direction
    accepted_force_norms = np.linalg.norm(accepted_forces, axis=1)
    info = DrudeStepInfo(
        max_force=float(np.max(accepted_force_norms, initial=0.0)),
        max_displacement=float(np.max(
            np.linalg.norm(displacement_ang, axis=1),
            initial=0.0,
        )),
        rms_force=float(np.sqrt(np.mean(accepted_forces**2)))
        if accepted_forces.size else 0.0,
    )
    next_state = DrudeCGState(
        forces=np.array(forces, copy=True),
        preconditioned_forces=np.array(preconditioned, copy=True),
        direction_nm=np.array(direction, copy=True),
    )
    return accepted, info, next_state


class DrudeSolver:
    """Relax Drude oscillator positions at fixed real-atom positions.

    Args:
        data: Drude oscillator metadata.
        force_oracle: Callable returning forces on Drude particles in
            kJ/mol/nm for a full positions array in Angstrom.
        force_tolerance: RMS Cartesian Drude-force component required for
            convergence, in kJ/mol/nm.
        displacement_tolerance: Optional maximum Drude-particle displacement
            in one iteration required for convergence, in Angstrom.  This is
            disabled by default to match OpenMM.
        max_iterations: Maximum fixed-point iterations.
        damping: Scalar multiplier applied to each diagonal-Newton step.
        stagnation_ratio: If provided, accept the current step when the sum
            of squared Drude forces exceeds this fraction of its value in the
            preceding iteration.  OpenMM's Drude-SCF minimizer uses 0.9.
        algorithm: ``"diagonal"`` for the original fixed-point update or
            ``"cg"`` for preconditioned conjugate gradient.
    """

    def __init__(
            self,
            data: DrudeData,
            force_oracle: Callable[[NDArray[np.float64]], NDArray[np.float64]],
            *,
            force_tolerance: float = 1.0,
            displacement_tolerance: float | None = None,
            max_iterations: int = 50,
            damping: float = 1.0,
            stagnation_ratio: float | None = 0.9,
            algorithm: str = "diagonal",
            iteration_callback: Callable[
                [int, NDArray[np.float64], NDArray[np.float64], DrudeStepInfo],
                None,
            ] | None = None,
    ) -> None:
        self.data = data
        self.force_oracle = force_oracle
        self.force_tolerance = force_tolerance
        self.displacement_tolerance = displacement_tolerance
        self.max_iterations = max_iterations
        self.damping = damping
        self.stagnation_ratio = stagnation_ratio
        self.iteration_callback = iteration_callback
        if algorithm not in {"diagonal", "cg"}:
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
        if self.algorithm == "cg":
            stepped, info, _ = drude_conjugate_gradient_step(
                self.data,
                positions,
                self.force_oracle,
                damping=self.damping,
            )
            return stepped, info
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
        final_rms_force = np.inf
        previous_force_squared = np.inf
        cg_state = None
        for iteration in range(1, self.max_iterations + 1):
            if self.algorithm == "cg":
                stepped, step_info, cg_state = (
                    drude_conjugate_gradient_step(
                        self.data,
                        relaxed,
                        self.force_oracle,
                        state=cg_state,
                        damping=self.damping,
                    )
                )
                forces = self.force_oracle(stepped)
            else:
                forces = self.force_oracle(relaxed)
                stepped, step_info = drude_relaxation_step(
                    self.data,
                    relaxed,
                    forces,
                    damping=self.damping,
                )
            final_max_force = step_info.max_force
            final_max_displacement = step_info.max_displacement
            final_rms_force = step_info.rms_force
            if self.iteration_callback is not None:
                self.iteration_callback(
                    iteration,
                    relaxed,
                    stepped,
                    step_info,
                )
            # Match OpenMM's default acceptance policy: test the RMS over all
            # Cartesian Drude-force components.  Displacement convergence is
            # retained only as an explicitly requested extension.
            if (
                    final_rms_force <= self.force_tolerance
                    or (
                        self.displacement_tolerance is not None
                        and final_max_displacement
                        <= self.displacement_tolerance
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
                    final_rms_force=final_rms_force,
                )
            previous_force_squared = force_squared
            relaxed = stepped
        # OpenMM does not raise when its iteration limit is reached.  Return
        # the latest positions while preserving the unconverged diagnostic.
        return stepped, DrudeSCFInfo(
            iterations=self.max_iterations,
            final_max_force=final_max_force,
            final_max_displacement=final_max_displacement,
            converged=False,
            final_rms_force=final_rms_force,
        )
