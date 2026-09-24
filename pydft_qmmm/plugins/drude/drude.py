"""Integrator plugin for Drude relaxation with fixed QM forces."""
from __future__ import annotations

__all__ = ["IntegratorDrudeSCF", "QMForceCache", "CachedForceCalculator"]

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
from pydft_qmmm.utils import load_omm_system

from .drude_data import extract_drude_data
from .drude_solver import DrudeCalculator
from .drude_solver import DrudeSCF

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
    """Relax Drudes with the previous step's QM forces held fixed.

    A calculator plugin snapshots QM forces during the normal force
    evaluation. The integrator holds that snapshot fixed while relaxing
    Drudes, without evaluating QM. Only MM forces change during relaxation.
    This is a lagged-force approximation, not a converged coupled
    electronic/Drude SCF.

    Register at index 0, outside position constraints and virtual-site
    updates, so relaxation sees the final nuclear geometry. The normal
    simulation force calculation updates QM forces after the step.

    Args:
        calculator: MM calculator or a composite containing one MM and
            one QM calculator. Both must share the integrated system.
        forcefield: XML forcefield files used to identify Drude particles.
        force_tolerance: Maximum residual force component in kJ/mol/angstrom.
        algorithm: BFGS or L-BFGS-B.
        max_iterations: Maximum Drude optimization iterations per step.
        debug_log: Optional file to append optimization diagnostics to.
    """

    def __init__(
            self,
            calculator: Calculator,
            forcefield: list[str] | str,
            force_tolerance: float = 0.10,
            algorithm: str = "BFGS",
            max_iterations: int = 50,
            debug_log: str | None = None,
    ) -> None:
        self.calculator = calculator
        self.debug_log = debug_log
        self.mm_calculator = calculator
        self.qm_force_cache: QMForceCache | None = None
        qm_calculator = None
        if isinstance(calculator, CompositeCalculator):
            # Use the existing group labels rather than inspecting backends.
            mm_calculators = [
                calc for calc in calculator.calculators
                if calc.calculator_group == "MM"
            ]
            qm_calculators = [
                calc for calc in calculator.calculators
                if calc.calculator_group == "QM"
            ]
            if (len(mm_calculators) != 1 or len(qm_calculators) != 1
                    or len(calculator.calculators) != 2):
                raise ValueError(
                    "Drude relaxation requires one MM and one QM calculator; "
                    "additional energy terms need an explicit force split.",
                )
            if any(
                calc.system is not calculator.system
                for calc in calculator.calculators
            ):
                raise ValueError("All calculators must share the same system.")
            self.mm_calculator = mm_calculators[0]
            qm_calculator = qm_calculators[0]
        elif calculator.calculator_group != "MM":
            raise ValueError("Drude relaxation requires an MM calculator.")
        omm_system = load_omm_system(calculator.system, forcefield)
        self.drude_data = extract_drude_data(omm_system)
        if self.drude_data is None or not len(self.drude_data):
            raise ValueError("The forcefield must contain Drude particles.")
        external_potential = None
        if qm_calculator is not None:
            self.qm_force_cache = QMForceCache()
            external_potential = CachedForceCalculator(
                calculator.system, self.qm_force_cache,
            )
        self.scf = DrudeSCF(
            DrudeCalculator(
                self.mm_calculator, self.drude_data, external_potential,
            ),
            max_iterations=max_iterations,
            algorithm=algorithm,
            force_tolerance=force_tolerance,
        )
        if qm_calculator is not None:
            qm_calculator.register_plugin(self.qm_force_cache)

    def _modify_integrate(
            self,
            integrate: Callable[[System], Returns],
    ) -> Callable[[System], Returns]:
        """Return relaxed positions without mutating the input system."""
        def inner(system: System) -> Returns:
            if system is not self.calculator.system:
                raise ValueError(
                    "Integrator and Drude calculator must share a system.",
                )
            positions, velocities = integrate(system)
            original_positions = system.positions.base.copy()
            try:
                system.positions[:] = positions
                energy, converged = self.scf.solve()
                if self.debug_log is not None:
                    with open(self.debug_log, "a") as log:
                        log.write(
                            f"Energy: {energy}; converged: {converged}\n",
                        )
                        log.write(f"{self.scf.result}\n")
                if not converged:
                    raise RuntimeError(
                        "Drude SCF did not reach the force tolerance. "
                        "Inspect the optimizer result in plugin.scf.result.",
                    )
                positions = system.positions.base.copy()
            finally:
                # Notify interfaces when restoring, including on failure.
                system.positions[:] = original_positions
            velocities[self.drude_data.drude_indices] = 0.0
            return positions, velocities
        return inner
