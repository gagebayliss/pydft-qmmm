"""Calculator plugin for native Drude-SCF relaxation."""
from __future__ import annotations

__all__ = ["DrudeSCF"]

from collections.abc import Callable
from typing import TYPE_CHECKING

import openmm

from pydft_qmmm.calculators import CalculatorPlugin
from pydft_qmmm.integrators import IntegratorPlugin
from pydft_qmmm.integrators import Returns

from .drude_data import extract_drude_data
from .drude_solver import DrudeSCFIterator
from .drude_solver import CompositeSCF

if TYPE_CHECKING:
    from pydft_qmmm.calculators import Results
    from pydft_qmmm.system import System
    from .drude_solver import DrudeSCFInfo


class DrudeSCF(IntegratorPlugin):
    def __init__(
        self,
        calculator,
        drude_data,
        force_tolerance: float = 1.0,
    ) -> None:
        self.calculator = calculator
        self.drude_data = drude_data
        self.force_tolerance = force_tolerance
        
    def _modify_integrate(
            self,
            integrate: Callable[[bool, bool], Results],
    ) -> Callable[[bool, bool], Results]:
        """Modify the calculate routine to relax Drudes beforehand."""
        def inner(system: System) -> Returns:
            
            updated_positions, updated_velocities = integrate(system)
            original_positions = system.positions.copy() # deep copy?
            
            self.calculator.system.positions[:] = updated_positions
            scf_iterator = DrudeSCFIterator(
                self.calculator,
                self.drude_data,
                self.force_tolerance,
            )
            scf = CompositeSCF(
                scf_iterator,
                max_iterations=50,
            )
            scf.solve()
            updated_positions = self.calculator.system.positions.copy()
            # this is brutal. 
            # could redesign SCF to avoid this... but seems inevitable.
            self.calculator.system.positions[:] = original_positions
            updated_velocities = self.calculator.system.velocities.copy()
            updated_velocities[self.drude_data.drude_indices] = 0.0

            return updated_positions, updated_velocities
        return inner
    

class DrudeSCF(CalculatorPlugin):
    """Relax Drude oscillators before calculator evaluations.

    Args:
        force_tolerance: RMS Cartesian Drude-force component required for
            convergence, in kJ/mol/nm.
        displacement_tolerance: Optional maximum Drude-particle displacement
            in one iteration required for convergence, in Angstrom.
        max_iterations: Maximum fixed-point iterations.
        damping: Scalar multiplier applied to each diagonal-Newton step.
        stagnation_ratio: Force-stagnation threshold.  The default 0.9
            matches OpenMM trajectory behavior; use ``None`` for strict
            force convergence.
        algorithm: Relaxation algorithm, ``"diagonal"`` or ``"cg"``.
    """

    def __init__(
            self,
            force_tolerance: float = 1.0,
            displacement_tolerance: float | None = None,
            max_iterations: int = 50,
            damping: float = 1.0,
            stagnation_ratio: float | None = 0.9,
            algorithm: str = "diagonal",
    ) -> None:
        self.force_tolerance = force_tolerance
        self.displacement_tolerance = displacement_tolerance
        self.max_iterations = max_iterations
        self.damping = damping
        self.stagnation_ratio = stagnation_ratio
        self.algorithm = algorithm
        self._solver: DrudeSolver | None = None
        self.last_info: DrudeSCFInfo | None = None

    def _get_solver(self) -> DrudeSolver:
        """Build or retrieve the Drude solver for this calculator."""
        if self._solver is not None:
            return self._solver
        potential = self.calculator.potential # needs to be MM potential?
        data = extract_drude_data(potential.base_context.getSystem())
        self._solver = DrudeSolver(
            data,
            potential,
            force_tolerance=self.force_tolerance,
            displacement_tolerance=self.displacement_tolerance,
            max_iterations=self.max_iterations,
            damping=self.damping,
            stagnation_ratio=self.stagnation_ratio,
            algorithm=self.algorithm,
        )
        return self._solver

    def relax(self) -> None:
        """Relax Drude positions and update the calculator system."""

    def _modify_calculate(
            self,
            calculate: Callable[[bool, bool], Results],
    ) -> Callable[[bool, bool], Results]:
        """Modify the calculate routine to relax Drudes beforehand."""
        def inner(
                return_forces: bool = True,
                return_components: bool = True,
        ) -> Results:
            solver = self._get_solver()

            positions, info = solver.relax(self.calculator.system.positions)
            self.calculator.system.positions[:] = positions

            self.last_info = info

            return calculate(return_forces, return_components)
        return inner
