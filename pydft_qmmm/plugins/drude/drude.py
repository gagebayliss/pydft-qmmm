"""Integrator plugin for Drude relaxation with fixed QM forces."""
from __future__ import annotations

__all__ = ["IntegratorDrudeSCF", "CalculatorDrudeSCF","QMForceCache", "CachedForceCalculator"]

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pydft_qmmm.calculators import Calculator
from pydft_qmmm.calculators import CalculatorPlugin
from pydft_qmmm.calculators import Results
from pydft_qmmm.calculators import CompositeCalculator
from pydft_qmmm.integrators import IntegratorPlugin

from .drude_solver import DrudeCalculator
from .drude_solver import DrudeSCF
from .drude_solver import CompositeSCF
from .drude_solver import SCFAdapter

if TYPE_CHECKING:
    from pydft_qmmm.integrators import Returns
    from pydft_qmmm.system import System


class QMForceCache(CalculatorPlugin):
    """Keep a copy of the most recent QM forces for Drude relaxation.

    Energy-only evaluations leave the last force snapshot unchanged.
    A CachedForceCalculator reads the snapshot without invoking QM.
    """

    def __init__(self) -> None:
        self.forces: NDArray[np.float64] | None = None
        self.positions: NDArray[np.float64] | None = None

    def _modify_calculate(
            self,
            calculate: Callable[[bool, bool], Results],
    ) -> Callable[[bool, bool], Results]:
        """Capture forces returned by the normal QM calculation."""
        def inner(
                return_forces: bool = True,
                return_components: bool = True,
        ) -> Results:
            results = calculate(return_forces, return_components)
            if return_forces:
                self.forces = np.asarray(results.forces).copy()
                self.positions = self.calculator.system.positions.base.copy()
            return results
        return inner


@dataclass(frozen=True)
class CachedForceCalculator(Calculator):
    """Expose observed forces as a linear external potential.

    Forces are returned by reference from the observer. The energy is
    their negative work relative to the observed geometry, with an
    arbitrary zero there. No underlying QM calculation is performed.

    Args:
        system: The system whose trial coordinates define the work.
        observer: Calculator plugin holding the latest force snapshot.
    """
    observer: QMForceCache
    calculator_group: str = field(default="External", init=False)

    @property
    def name(self) -> str:
        """Name of the external contribution."""
        return "CachedForces"

    def calculate(
            self,
            return_forces: bool = True,
            return_components: bool = True,
    ) -> Results:
        """Return linear energy and a reference to the observed forces."""
        if self.observer.forces is None or self.observer.positions is None:
            raise RuntimeError(
                "QM forces have not been cached. Calculate total forces "
                "after installing the observer and before relaxing Drudes.",
            )
        displacement = self.system.positions - self.observer.positions
        energy = -float(np.sum(self.observer.forces * displacement))
        results = Results(energy)
        if return_forces:
            results.forces = self.observer.forces
        if return_components:
            results.components = {self.name: energy}
        return results

class IntegratorDrudeSCF(IntegratorPlugin):
    def __init__(
            self,
            forcefield: list[str] | str,
            force_tolerance: float = 0.10,
            algorithm: str = "L-BFGS-B",
            max_iterations: int = 50,
            strict: bool = False,
    ) -> None:
        raise NotImplementedError()


class CalculatorDrudeSCF(CalculatorPlugin):
    """
    Args:
        forcefield: Optional legacy argument. The MM calculator supplies
            the system's Drude metadata.
        force_tolerance: Maximum residual force component in kJ/mol/angstrom.
        algorithm: any scipy.optimize algorithm not requiring a Hessian.
        max_iterations: Maximum Drude optimization iterations per step.
    """

    def __init__(
            self,
            forcefield: list[str] | str | None = None,
            force_tolerance: float = 0.10,
            algorithm: str = "L-BFGS-B",
            max_iterations: int = 50,
            strict: bool = False,
    ) -> None:
        self.force_tolerance = force_tolerance
        self.algorithm = algorithm
        self.max_iterations = max_iterations
        self.strict = strict 
        self._bypass_plugin = False

    def modify(self, calculator: Calculator):
        self.calculator = calculator
        self.mm_calculator = calculator
        self.qm_force_cache: QMForceCache | None = None
        qm_calculator = None

        if isinstance(calculator, CompositeCalculator):
            mm_calculators = [
                calc for calc in calculator.calculators
                if calc.calculator_group == "MM"
            ]
            qm_calculators = [
                calc for calc in calculator.calculators
                if calc.calculator_group == "QM"
            ]
            if len(mm_calculators) != 1 or len(qm_calculators) != 1:
                raise ValueError("...")
            if any(
                calc.system is not calculator.system
                for calc in calculator.calculators
            ):
                raise ValueError("All calculators must share the same system.")
            self.mm_calculator = mm_calculators[0]

            qm_calculator = CompositeCalculator( # for PME potentials
                calculator.system,
                [calc for calc in calculator.calculators
                    if calc.calculator_group != "MM"]
            )

        elif calculator.calculator_group != "MM":
            raise ValueError("Drude relaxation requires an MM calculator.")

        external_potential = None

        if qm_calculator is not None:
            self.qm_force_cache = QMForceCache()
            external_potential = CachedForceCalculator(
                calculator.system, self.qm_force_cache,
            )
        adapter = DrudeCalculator(self.mm_calculator, external_potential=external_potential)
        if not len(adapter.indices):
            raise ValueError("The system must contain Drude particles.")
        self.drude_scf = DrudeSCF(
            adapter,
            max_iterations=self.max_iterations,
            algorithm=self.algorithm,
            force_tolerance=self.force_tolerance,
        )
        if qm_calculator is None:
            self.scf = self.drude_scf
        else:
            self.scf = CompositeSCF(
                [SCFAdapter(qm_calculator), self.drude_scf],
                objective_index=0,  # Compare QM energies across macroiterations.
                tolerance=1e-6,
                maxiter=10,
            )
            qm_calculator.register_plugin(self.qm_force_cache)

        # OpenMM assigns part of each parent's mass to its oscillator.
        # With SCF relaxation only the parent is integrated.
        masses = calculator.system.masses.base.copy()
        indices = adapter.indices
        np.add.at(masses, adapter.parent_indices, masses[indices])
        masses[indices] = 0
        calculator.system.masses[:] = masses
        calculator.system.velocities[indices] = 0

    def _modify_calculate(
            self,
            calculate: Callable[[bool, bool], Results],
        ) -> Callable[[bool, bool], Results]:
        """Relax Drude coordinates before evaluating the requested result."""
        def inner(
                return_forces: bool = True,
                return_components: bool = True,
            ) -> Results:
            
            if not self._bypass_plugin: # prevent recursion when using MM calculator
                self._bypass_plugin = True
                try:
                    if self.qm_force_cache is not None:
                        # The first QM field must see the translated oscillators.
                        self.drude_scf.restore_guess()
                    _, converged = self.scf.solve()
                    if not converged and self.strict:
                        raise RuntimeError(
                            "Drude SCF did not reach the force tolerance."
                        )
                finally:
                    self._bypass_plugin = False

            # TODO
            # can avoid this extra calculation by
            # using systemcache on calculate
            results = calculate(return_forces,return_components)
            return results
        return inner
