"""Relax Drude coordinates using calculator energies and forces."""
from __future__ import annotations

__all__ = ["DrudeCalculator", "DrudeSCF", "SCFAdapter", "CompositeSCF"]

from typing import Protocol
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pydft_qmmm.interfaces.psi4.psi4_interface import Psi4Interface
    from scipy.optimize import OptimizeResult
    from pydft_qmmm.calculators import Calculator


class DrudeCalculator:
    """Expose parent-relative Drude displacements for energy minimization.

    Args:
        calculator: Calculator for the base energy and forces.
        external_potential: Optional additional energy and forces.
        qm_interface: Psi4 interface for frozen-density embedding.
    """

    def __init__(
            self,
            calculator: Calculator,
            external_potential: Calculator | None = None,
            qm_interface: Psi4Interface | None = None,
    ) -> None:
        self.calculator = calculator
        system = calculator.system
        mask = system.virtual_types.base == "drude"
        self.indices = system.virtual_site_indices.base[mask]
        self.parent_indices = np.array(
            [parent for (parent,) in system.virtual_parents.base[mask]],
            dtype=np.int64,
        )
        self.external_potential = external_potential
        self.qm_interface = qm_interface
        self.density_matrix = None

    def freeze_density(self) -> None:
        """Keep the QM density for this minimization."""
        qm = self.qm_interface
        if qm is not None:
            self.density_matrix = qm.get_density_matrix()

    def calculate(
            self,
            positions: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        """Return combined energy and its gradient over Drude displacements."""
        self.state = positions
        results = self.calculator.calculate(return_components=False)
        energy = results.energy
        forces = results.forces.copy()
        if self.external_potential is not None:
            external = self.external_potential.calculate(
                return_components=False,
            )
            energy += external.energy
            forces += external.forces
        if self.qm_interface is not None:
            if self.density_matrix is None:
                raise RuntimeError("Set the QM density before evaluating Drude forces.")
            system = self.calculator.system
            energy += self.qm_interface.compute_embedding_energy(
                self.density_matrix, arbitrary_potential=False,
            )
            embedding = system.select("subsystem II")
            indices = [i for i in self.indices if i in embedding]
            field = self.qm_interface.compute_electric_field(
                system.positions[indices], self.density_matrix,
            )
            for i, electric_field in zip(indices, field):
                forces[i] += system.charges[i] * electric_field
        return energy, -forces[self.indices].reshape(-1)

    @property
    def state(self) -> NDArray[np.float64]:
        """Return the flattened displacements from the parent atoms."""
        return np.asarray(
            self.calculator.system.positions[self.indices]
            - self.calculator.system.positions[self.parent_indices],
        ).reshape(-1).copy()

    @state.setter
    def state(self, coordinates: NDArray[np.float64]) -> None:
        coordinates = np.asarray(coordinates, dtype=float)
        self.calculator.system.positions[self.indices] = (
            self.calculator.system.positions[self.parent_indices]
            + coordinates.reshape(-1, 3)
        )

class DrudeSCF:
    """Minimize energy with respect to Drude positions at fixed nuclei.

    Args:
        calculator: Energy and gradient adapter for the Drude coordinates.
        max_iterations: Maximum number of optimizer iterations.
        algorithm: Any scipy.optimize algorithm accepting a gradient and
            not requiring a Hessian.
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
        self.calculator = calculator
        self.max_iterations = max_iterations
        self.algorithm = algorithm
        self.force_tolerance = force_tolerance
        self.result: OptimizeResult | None = None
        self._last_displacements: NDArray[np.float64] | None = None

    def restore_guess(self) -> None:
        """Place the previous solution relative to the current parents."""
        if self._last_displacements is not None:
            self.calculator.state = self._last_displacements

    def solve(self) -> tuple[float, bool]:
        from scipy.optimize import minimize
        self.calculator.freeze_density()
        if not len(self.calculator.indices):
            energy, _ = self.calculator.calculate(self.calculator.state)
            return energy, True

        options = {
            "gtol": self.force_tolerance,
            "maxiter": self.max_iterations,
        }
        if self.algorithm == "L-BFGS-B":
            options["ftol"] = 0.0
        initial = self._last_displacements
        if initial is None:
            initial = self.calculator.state
        self.result = minimize(
            self.calculator.calculate,
            initial.copy(),
            jac=True,
            method=self.algorithm,
            options=options,
        )
        # A failed L-BFGS-B line search can leave a good accepted point
        # near the minimum. Retry once with fresh curvature information.
        if (
            self.algorithm == "L-BFGS-B" and not self.result.success
            and self.result.status == 2
            and 0 < self.result.nit < self.max_iterations
        ):
            options["maxiter"] -= self.result.nit
            self.result = minimize(
                self.calculator.calculate, self.result.x, jac=True,
                method=self.algorithm, options=options,
            )
        energy, gradient = self.calculator.calculate(self.result.x)
        converged = bool(
            np.isfinite(energy)
            and np.all(np.isfinite(gradient))
            and np.max(np.abs(gradient)) <= self.force_tolerance
        )
        if converged:
            self._last_displacements = self.result.x.copy()
        return energy, converged


class SCF(Protocol):
    """An iterative solver returning its objective and convergence status."""

    def solve(self) -> tuple[float, bool]:
        """Update the state and return its objective and convergence."""
        ...


class SCFAdapter:
    def __init__(self, calculator: Calculator) -> None:
        self.calculator = calculator
        self.result = None

    def solve(self) -> tuple[float, bool]:
        result = self.calculator.calculate(return_forces=False, return_components=False)
        return result.energy, bool(np.isfinite(result.energy))


class CompositeSCF:
    """Cycle through coupled solvers until their objective stabilizes.

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
