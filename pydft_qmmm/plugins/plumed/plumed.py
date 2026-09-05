"""A plugin interface to the Plumed enhanced sampling suite.
"""
from __future__ import annotations

__all__ = ["PLUMED"]

from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.utils import DependencyImportError
from pydft_qmmm.calculators.calculator import CalculatorPlugin

if TYPE_CHECKING:
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
            arbitrary_cvs: dict[str,CollectiveVariable],
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
            results = calculate(return_forces, return_components)
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

            cv_values = np.zeros((len(self.arbitrary_cvs),))
            cv_forces = np.zeros((len(self.arbitrary_cvs),))
            for i, (cv_name, cv) in enumerate(self.arbitrary_cvs.items()):
                cv_values[i] = cv.get_scalar_cv(self.calculator,results)
                self.plumed.cmd(f"setExtraCV {cv_name}", cv_values[i])
                self.plumed.cmd(f"setExtraCVForce {cv_name}", cv_forces[i])

            self.plumed.cmd("prepareCalc")
            self.plumed.cmd("performCalc")
            biased_energy = np.zeros((1,))
            self.plumed.cmd("getBias", biased_energy)

            for i, (cv_name, cv) in enumerate(self.arbitrary_cvs.items()):
                biased_forces += cv_forces[i] * cv.chain_rule()

            results.energy += biased_energy[0]
            results.forces += biased_forces
            results.components.update(
                {"Plumed Bias Energy": biased_energy[0]},
            )
            return results
        return inner
