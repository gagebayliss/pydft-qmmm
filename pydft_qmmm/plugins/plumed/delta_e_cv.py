from __future__ import annotations

__all__ = ["DeltaECV"]

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np

from .collective_variable import CollectiveVariable
from pydft_qmmm.calculators import CompositeCalculator

if TYPE_CHECKING:
    from pydft_qmmm.calculators import Calculator
    from pydft_qmmm.calculators import Results
    from numpy.typing import NDArray


class DeltaECV(CollectiveVariable):
    """The generalized solvation collective variable class.

    Args:
        calculator: The calculator whose system state and energies/forces
            will be used to calculate collective variables.
        scheme: a string of the form "{state 1} - {state 2}", e.g.
            "anion - neutral"; used as functional form of the CV.
        states: If MM only, a dict mapping the above states to dicts containing
            arrays of "indices" and "charges". If QM/MM, a dict mapping the above
            states to tuples for charge and multiplicity.
    """
    def __init__(
            self,
            scheme: str,
            states: dict[str,dict[str,NDArray[np.float64] | NDArray[np.int64]]] |\
                    dict[str,tuple],
            ):
        self.states = states
        self._cv_gradient = None
        self._state_names = (key.strip() for key in scheme.split("-"))
        if len(states) != 2:
            raise ValueError("...")
        if not (self._state_names[0] in states.keys() and\
                self._state_names[1] in states.keys()):
            raise ValueError("...")

    def get_scalar_cv(
            self,
            calculator: Calculator,
            results: Results,
    ) -> Results:
        r"""Calculate a scalar collective variable from system state
            and calculator results.

        Args:
            results: The energies and forces from a calculation.

        Returns:
            The scalar value of the collective variable.
        """
        system = calculator.system
        original_charges = system.charges.copy()

        skip_state = ""
        for state_name, state in self.states.items():
            if np.array_equal(original_charges,state):
                skip_state = state_name

        energies = list()
        forces = list() 

        for state_name in self._state_names:
            if skip_state == state_name:
                result = results
            elif isinstance(calculator,CompositeCalculator): #e.g., QM/MM calculation
                raise NotImplementedError("Cannot use QM/MM with DeltaECV")
            else:
                solvent_indices = self.states[state_name]["indices"]
                system.charges[solvent_indices] = self.states[state_name]["charges"]
                result = calculator.calculate()
            energies.append(result.energy)
            forces.append(result.forces)

        print(self._state_names)
        delta_E = energies[self._state_names[0]] - energies[self._state_names[1]]

        # used in chain_rule()
        self._cv_gradient = -(forces[self._state_names[0]] - forces[self._state_names[1]])

        system.charges[:] = original_charges
        return delta_E

    def chain_rule(
            self,
    ) -> NDArray[np.float64]:
        r"""Calculate change in atomic position per unit change in collective variable.

        Returns:
            An array of forces.
        """
        return self._cv_gradient.copy()