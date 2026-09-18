"""Calculator plugin for native Drude-SCF relaxation."""
from __future__ import annotations

__all__ = ["DrudeSCF"]

from collections.abc import Callable
from typing import TYPE_CHECKING

import openmm

from pydft_qmmm.calculators import CalculatorPlugin
from pydft_qmmm.integrators import IntegratorPlugin
from pydft_qmmm.integrators import Returns
from pydft_qmmm.utils import load_omm_system

from .drude_data import extract_drude_data
from .drude_solver import CompositeSCF
from .drude_solver import DrudeCalculator

if TYPE_CHECKING:
    from pydft_qmmm.calculators import Results
    from pydft_qmmm.system import System
    from .drude_solver import DrudeSCFInfo


class DrudeSCF(IntegratorPlugin):
    def __init__(
        self,
        calculator, 
        forcefield: list[str] | str,
        force_tolerance: float = 10.0, #kjmol/angstrom
        algorithm: str = "BFGS",
        max_iterations: int = 50,
        debug_log: str | None = None,
    ) -> None:
        self.calculator = calculator
        self.force_tolerance = force_tolerance
        omm_system = load_omm_system(calculator.system,forcefield)
        self.drude_data = extract_drude_data(omm_system)
        self.algorithm = algorithm
        self.max_iterations = max_iterations
        self.debug_log = debug_log
        del omm_system
        
    def _modify_integrate(
            self,
            integrate: Callable[[bool, bool], Results],
    ) -> Callable[[bool, bool], Results]:
        """Modify the calculate routine to relax Drudes beforehand."""
        def inner(system: System) -> Returns:
            updated_positions, updated_velocities = integrate(system)

            original_positions = system.positions.base.copy()
            self.calculator.system.positions[:] = updated_positions
            
            scf_calculator = DrudeCalculator(
                self.calculator,
                self.drude_data,
            )

            scf = CompositeSCF(
                scf_calculator,
                max_iterations=self.max_iterations,
                algorithm=self.algorithm,
                force_tolerance=self.force_tolerance
            )

            result = scf.solve()
            if self.debug_log:
                with open(self.debug_log,"w") as f:
                    f.write(str(result))

            updated_positions = self.calculator.system.positions.base.copy()
            
            drude_indices = self.drude_data.drude_indices

            self.calculator.system.positions[:] = original_positions
            assert self.calculator.system is system
            updated_velocities[self.drude_data.drude_indices] = 0.0
            

            return updated_positions, updated_velocities
        return inner
    
