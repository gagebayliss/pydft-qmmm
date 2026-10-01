"""A plugin interface to the Plumed enhanced sampling suite.
"""
from __future__ import annotations

__all__ = ["PLUMED"]

from copy import deepcopy
from typing import TYPE_CHECKING

import time

import numpy as np

from pydft_qmmm.utils import DependencyImportError
from pydft_qmmm.calculators.calculator import CalculatorPlugin

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable
    from pydft_qmmm.calculators import Calculator
    from pydft_qmmm.calculators import Results
    from .collective_variable import CollectiveVariable


class PLUMED(CalculatorPlugin):
    """Apply enhanced sampling biases to energy and force calculations.

    Args:
        input_commands: A multi-line string containing all pertinent
            instructions for Plumed.
        log_file: A directory for recording output from Plumed.
        arbitrary_cvs: A dictionary containing named CV objects.
            Names in dictionary should agree with name in Plumed
            input, e.g. EXTRACV NAME={name}
    """

    def __init__(
            self,
            input_commands: str,
            log_file: str,
            arbitrary_cvs: dict[str,CollectiveVariable] | None = None,
    ) -> None:
        try:
            import plumed
        except ImportError:
            raise DependencyImportError(
                "plumed",
                "performing enhanced sampling",
                "https://github.com/plumed/plumed2",
            )
        self.input_commands = input_commands
        self.log_file = log_file
        self.plumed = plumed.Plumed()
        self.plumed.cmd("setMDEngine", "python")
        self.arbitrary_cvs = arbitrary_cvs
        self.frame = 0
        self._bypass_plumed = False
        self._initialized = False

    def __deepcopy__(self, memo: dict[int, Any]) -> PLUMED:
        """Copy plugin state with an independent, lazily created engine.

        Native PLUMED bias history is not cloned. Bypassed copies used
        for CV evaluation never initialize an engine or open its files.
        """
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        for name, value in vars(self).items():
            if name not in {"plumed", "_initialized"}:
                setattr(copied, name, deepcopy(value, memo))
        copied.plumed = None
        copied._initialized = False
        return copied

    def modify(
            self,
            calculator: Calculator,
    ) -> None:
        """Modify the functionality of a calculator and set up Plumed.

        Args:
            calculator: The calculator whose functionality will be
                modified by the plugin.
        """
        self.calculator = calculator
        self._initialize_plumed()

    def _initialize_plumed(self) -> None:
        """Initialize the engine against this plugin's own calculator."""
        if self._initialized:
            return
        if self.plumed is None:
            import plumed
            self.plumed = plumed.Plumed()
            self.plumed.cmd("setMDEngine", "python")
        self.plumed.cmd("setNatoms", len(self.calculator.system))
        self.plumed.cmd("setMDLengthUnits", 1/10)
        self.plumed.cmd("setMDTimeUnits", 1/1000)
        self.plumed.cmd("setMDMassUnits", 1.)
        self.plumed.cmd("setTimestep", 1.) # but we use other timesteps sometimes. This might be a problem
        self.plumed.cmd("setKbT", 1.)
        self.plumed.cmd("setLogFile", self.log_file) 
        self.plumed.cmd("init")
        for line in self.input_commands.split("\n"):
            self.plumed.cmd("readInputLine", line)
        self._initialized = True

    def _modify_calculate(
            self,
            calculate: Callable[[bool, bool], Results],
    ) -> Callable[[bool, bool], Results]:
        """Modify the calculate routine to perform biasing afterward.

        Args:
            calculate: The calculation routine to modify.

        Returns:
            The modified calculation routine which implements Plumed
            enhanced sampling after performing the unbiased calculation
            routine.
        """
        def inner(
                return_forces: bool = True,
                return_components: bool = True,
        ) -> Results:
            time_0 = time.perf_counter()
            results = calculate(return_forces, return_components)
            time_1 = time.perf_counter()
            
            if self._bypass_plumed:
                # arbitrary CVs call calculator,
                # infinite recursion if we don't do this
                return results

            self._initialize_plumed()
            self.plumed.cmd("setStep", self.frame)
            self.frame += 1
            self.plumed.cmd("setBox", self.calculator.system.box.T)
            self.plumed.cmd("setPositions", self.calculator.system.positions)
            self.plumed.cmd("setEnergy", results.energy)
            self.plumed.cmd("setMasses", self.calculator.system.masses)
            biased_forces = np.zeros(self.calculator.system.positions.shape)
            self.plumed.cmd("setForces", biased_forces)
            virial = np.zeros((3, 3))
            self.plumed.cmd("setVirial", virial)

            time_2 = time.perf_counter()
            if self.arbitrary_cvs is not None:
                cv_values = np.zeros((len(self.arbitrary_cvs),))
                cv_forces = np.zeros((len(self.arbitrary_cvs),))
                for i, (cv_name, cv) in enumerate(self.arbitrary_cvs.items()):
                    self._bypass_plumed = True
                    try:
                        cv_values[i] = cv.get_scalar_cv(self.calculator,results)
                    finally:
                        self._bypass_plumed = False
                    self.plumed.cmd(f"setExtraCV {cv_name}", cv_values[i:i+1])
                    self.plumed.cmd(f"setExtraCVForce {cv_name}", cv_forces[i:i+1])
            time_3 = time.perf_counter()

            self.plumed.cmd("prepareCalc")
            self.plumed.cmd("performCalc")
            biased_energy = np.zeros((1,))
            self.plumed.cmd("getBias", biased_energy)

            time_4 = time.perf_counter()

            if self.arbitrary_cvs is not None:
                for i, (cv_name, cv) in enumerate(self.arbitrary_cvs.items()):
                    biased_forces += cv_forces[i] * cv.chain_rule()

            time_5 = time.perf_counter()

            results.energy += biased_energy[0]
            results.forces += biased_forces
            results.components.update(
                {"Plumed Bias Energy": biased_energy[0]},
            )
            time_6 = time.perf_counter()
#             print(f"plumed")
#             print(f"calculate: {time_1 - time_0:.5f}")
#             print(f"arbitrary cv: {time_3 - time_2:.5f}")
#             print(f"total: {time_6 - time_0:.5f}")
            return results
        return inner
