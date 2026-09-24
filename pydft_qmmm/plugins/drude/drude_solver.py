"""Relax Drude coordinates using calculator energies and forces."""
from __future__ import annotations

__all__ = ["DrudeCalculator", "DrudeSCF", "SCFAdapter", "CompositeSCF"]

from typing import Protocol
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult
    from pydft_qmmm.calculators import Calculator
    from .drude_data import DrudeData


class DrudeCalculator:
    """Expose Drude coordinates as an energy minimization problem.

    The wrapped calculator supplies the base energy and forces. An
    optional external calculator supplies an additional contribution
    through the same interface. Coordinates are in angstrom and energies
    in kJ/mol. Neither calculator is required to use a particular backend.

    Args:
        calculator: Calculator for the base energy and forces.
        drude_data: Metadata identifying the coordinates to optimize.
        external_potential: Optional calculator for external energy and
            forces, evaluated at each trial geometry. A cached-force
            calculator can supply a fixed QM field without QM evaluations.
    """

    def __init__(
            self,
            calculator: Calculator,
            drude_data: DrudeData,
            external_potential: Calculator | None = None,
    ) -> None:
        self.calculator = calculator
        self.indices = drude_data.drude_indices.copy()
        self.external_potential = external_potential

    def calculate(
            self,
            positions: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        """Return combined energy and its gradient over Drude coordinates."""
        self.state = positions
        results = self.calculator.calculate(return_components=False)
        energy = results.energy
        gradient = -results.forces[self.indices].reshape(-1)
        if self.external_potential is not None:
            external = self.external_potential.calculate(
                return_components=False,
            )
            energy += external.energy
            gradient -= external.forces[self.indices].reshape(-1)
        return energy, gradient

    @property
    def state(self) -> NDArray[np.float64]:
        """Return a copy of the flattened Drude coordinates."""
        return np.asarray(
            self.calculator.system.positions[self.indices],
        ).reshape(-1).copy()

    @state.setter
    def state(self, coordinates: NDArray[np.float64]) -> None:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.size != 3 * len(self.indices):
            raise ValueError("Expected three coordinates per Drude particle.")
        # Write through the observed array to update the calculator caches.
        self.calculator.system.positions[self.indices] = coordinates.reshape(
            -1, 3,
        )


class DrudeSCF:
    """Minimize energy with respect to Drude positions at fixed nuclei.

    Args:
        calculator: Energy and gradient adapter for the Drude coordinates.
        max_iterations: Maximum number of optimizer iterations.
        algorithm: Either BFGS or L-BFGS-B.
        force_tolerance: Maximum absolute force component in kJ/mol/angstrom.

    Attributes:
        result: SciPy result from the latest minimization, for diagnostics.
    """

    def __init__(
            self,
            calculator: DrudeCalculator,
            max_iterations: int = 50,
            algorithm: str = "L-BFGS-B",
            force_tolerance: float = 0.10,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if not np.isfinite(force_tolerance) or force_tolerance <= 0:
            raise ValueError("force_tolerance must be finite and positive.")
        if algorithm not in {"BFGS", "L-BFGS-B"}:
            raise ValueError("algorithm must be 'BFGS' or 'L-BFGS-B'.")
        self.calculator = calculator
        self.max_iterations = max_iterations
        self.algorithm = algorithm
        self.force_tolerance = force_tolerance
        self.result: OptimizeResult | None = None

    def solve(self) -> tuple[float, bool]:
        """Leave Drudes at the accepted minimum and report force convergence.

        Optimizer success alone is insufficient: energy stagnation may
        terminate minimization before the force tolerance is satisfied.
        """
        from scipy.optimize import minimize

        if not len(self.calculator.indices):
            energy, _ = self.calculator.calculate(self.calculator.state)
            return energy, True

        options = {
            "gtol": self.force_tolerance,
            "maxiter": self.max_iterations,
        }
        if self.algorithm == "L-BFGS-B":
            # Use forces, rather than relative energy changes, to stop.
            options["ftol"] = 0.0
        self.result = minimize(
            self.calculator.calculate,
            self.calculator.state,
            jac=True,
            method=self.algorithm,
            options=options,
        )
        # A line search can finish at a rejected trial point. Restore and
        # evaluate the accepted coordinates so all calculator state agrees.
        energy, gradient = self.calculator.calculate(self.result.x)
        converged = bool(
            np.isfinite(energy)
            and np.all(np.isfinite(gradient))
            and np.max(np.abs(gradient)) <= self.force_tolerance
        )
        return energy, converged


class SCF(Protocol):
    """An iterative solver returning its objective and convergence status."""

    def solve(self) -> tuple[float, bool]:
        """Update the state and return its objective and convergence."""
        ...


class SCFAdapter:
    """Adapt a calculator that raises an exception if its SCF fails."""

    def __init__(self, calculator: Calculator) -> None:
        self.calculator = calculator

    def solve(self) -> tuple[float, bool]:
        """Evaluate the calculator and report a finite energy as success."""
        result = self.calculator.calculate()
        return result.energy, bool(np.isfinite(result.energy))


class CompositeSCF:
    """Cycle through coupled solvers until their objective stabilizes.

    All solvers must also report convergence in the final cycle. This
    driver does not freeze densities or otherwise change solver behavior.

    Args:
        scf_objects: Solvers in evaluation order.
        objective_index: Solver whose objective is compared between cycles.
        tolerance: Absolute tolerance for changes in that objective.
        maxiter: Maximum number of full cycles.
    """

    def __init__(
            self,
            scf_objects: list[SCF],
            objective_index: int,
            tolerance: float = 1e-6,
            maxiter: int = 10,
    ) -> None:
        if not 0 <= objective_index < len(scf_objects):
            raise ValueError("objective_index must identify a solver.")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive.")
        if maxiter < 1:
            raise ValueError("maxiter must be positive.")
        self.scf_objects = scf_objects
        self.objective_index = objective_index
        self.tolerance = tolerance
        self.maxiter = maxiter

    def solve(self) -> tuple[float, bool]:
        """Return the last objective and the coupled convergence status."""
        previous_value = np.inf
        for _ in range(self.maxiter):
            all_converged = True
            for index, scf in enumerate(self.scf_objects):
                energy, converged = scf.solve()
                all_converged = all_converged and converged
                if index == self.objective_index:
                    value = energy
            if all_converged and abs(previous_value - value) <= self.tolerance:
                return value, True
            previous_value = value
        return value, False
