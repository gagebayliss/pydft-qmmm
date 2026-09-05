"""Drude SCF solver."""
from __future__ import annotations

__all__ = [
    "DrudeSCFIterator",
    "SCF"
]

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .drude_data import DrudeData


class DrudeSCFIterator:
    def __init__(self,calculator,data,force_tolerance,stopping_ratio,damping):
        self.calculator = calculator
        self.data = data
        self.force_tolerance = force_tolerance
        self.stopping_ratio = stopping_ratio
        self.damping = damping
        self._forces = np.array([np.inf])
        self._last_forces = np.array([np.inf])
        self._history = {
            "drude_positions": [],
            "drude_forces": []
        }

    def step(self) -> None:
        self._last_forces = self._forces
        forces = self.calculate()
        self._forces = forces
        
        displacement_ang = (
            self.damping
            * forces
            / self.data.force_constants.reshape((-1, 1))
        )
        self.state = self.state + displacement_ang
        
        # tighten this up.
        # do we need to save first step separately?
        self._history["drude_positions"].append(self.state)
        self._history["drude_forces"].append(forces)
        
    
    def calculate(self) -> NDArray[np.float64]:
        results = self.calculator.calculate()
        forces = results.forces[self.data.drude_indices]
        return forces
    
    def is_converged(self):
        if self.objective_function <= self.force_tolerance:
            return True
        elif self.stopping_ratio is not None:
            force_squared = float(np.sum(self._forces**2))
            last_force_squared = float(np.sum(self._last_forces**2))
            if (force_squared > self.stopping_ratio*last_force_squared):
                return True
        else:
            return False
    
    @property
    def objective_function(self) -> NDArray[np.float64]:
        return np.sqrt(np.mean(self._last_forces**2))
    
    @property
    def state(self) -> NDArray[np.float64]:
        return self.calculator.system.positions[self.data.drude_indices]
    
    @state.setter
    def state(self,drude_coordinates) -> None:
        self.calculator.system.positions[self.data.drude_indices] = drude_coordinates
    
    @property
    def history(self) -> dict[str, list[NDArray[np.float64]]]:
        return self._history
    
    def reset_history(self) -> None:
        self._history = {
            "drude_positions": [],
            "drude_forces": []
        }
    

class CompositeSCF:
    def __init__(self,scf_iterator,max_iterations):
        self.scf_iterator = scf_iterator
        self.max_iterations = max_iterations
        self._is_converged = False
    
    def solve(self) -> None:
        self._is_converged = False
        self.scf_iterator.reset_history()
        for iteration in range(1, self.max_iterations + 1):
            self.scf_iterator.step()
            if self.scf_iterator.is_converged():
                self._is_converged = True
                return
        self._is_converged = False
        
    def is_converged(self) -> bool:
        return self._is_converged
    

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

            # if (
            #     final_rms_force <= self.force_tolerance
            #     or (
            #         self.displacement_tolerance is not None
            #         and final_max_displacement
            #         <= self.displacement_tolerance
            #     ) or (
            #     self.stagnation_ratio is not None
            #     and iteration > 1
            #     and force_squared
            #     > self.stagnation_ratio*previous_force_squared
            #     )
            # ):
            #     return stepped, DrudeSCFInfo(
            #         iterations=iteration,
            #         final_max_force=final_max_force,
            #         final_max_displacement=final_max_displacement,
            #         converged=True,
            #         final_rms_force=final_rms_force,
            #     )

        #     force_squared = float(np.sum(forces*forces))
        #     previous_force_squared = force_squared
        #     relaxed = stepped

        # return stepped