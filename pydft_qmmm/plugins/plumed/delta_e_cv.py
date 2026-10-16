from __future__ import annotations

__all__ = ["DeltaECV"]

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.hamiltonians import MMHamiltonian

from .collective_variable import CollectiveVariable
from .plumed import PLUMED
from pydft_qmmm.calculators import CompositeCalculator

if TYPE_CHECKING:
    from typing import Any
    from pydft_qmmm.calculators import Calculator
    from pydft_qmmm.calculators import Results
    from pydft_qmmm.system import System
    from numpy.typing import NDArray


class DeltaECV(CollectiveVariable):
    """The generalized solvation collective variable class.

    Charge-state calculators are initialized once. The driving system
    must keep its charges and box fixed; only positions are synchronized.

    Args:
        query: search string for the residue used with the DeltaECV.
            (use "subsystem I" for QM/MM)
        scheme: A string of the form "{state 1} - {state 2}", e.g.
            "anion - neutral"; used as functional form of the CV.
        states_forcefield: If using MM, a dict mapping the above 
            states to filenames for the forcefields for the 
            different states.
        states_charge_multiplicity: If QM/MM, a dict mapping the
            above states to tuples for charge and multiplicity.
    """
    def __init__(
            self,
            query: str,
            scheme: str,
            states_forcefield: dict[str,str] = None,
            states_charge_multiplicity: dict[str,tuple] = None,
        ):
        self.query = query
        self._state_names = [key.strip() for key in scheme.split("-")]
        self.forcefields = states_forcefield
        self.charge_multiplicity_tuples = states_charge_multiplicity 
        self._cv_gradient = None
        self._aux_calculators = {}
        self._state_calculators = {}
        self._indices = None

    def get_scalar_cv(
            self,
            calculator: Calculator,
            results: Results,
    ) -> float:
        r"""Calculate a scalar collective variable from system state
            and calculator results.

        Args:
            results: The energies and forces from a calculation.

        Returns:
            The scalar value of the collective variable.
        """
        system = calculator.system
        positions = system.positions.base.copy()
        energies = {}
        forces = {}

        if self._indices is None:
            self._indices = sorted(system.select(self.query))

        for state_name in self._state_names:
            if state_name not in self._state_calculators:
                aux_calculator = self._build_aux_calculator(
                    system, self.forcefields[state_name],
                )
                charges = aux_calculator.system.charges.base.copy()
                state_calculator = None
                if not np.allclose(system.charges[self._indices], charges):
                    state_calculator = calculator.copy()
                    self._bypass_bias(state_calculator)
                    # Assign charges once, only on the independent system.
                    state_calculator.system.charges[self._indices] = charges
                self._aux_calculators[state_name] = aux_calculator
                self._state_calculators[state_name] = state_calculator

            state_calculator = self._state_calculators[state_name]
            if state_calculator is None:
                result = results
            else:
                state_calculator.system.positions[:] = positions
                result = state_calculator.calculate()

            aux_calculator = self._aux_calculators[state_name]
            aux_calculator.system.positions[:] = positions[self._indices]
            intra_result = aux_calculator.calculate()
            energies[state_name] = result.energy - intra_result.energy
            forces[state_name] = result.forces.copy()
            forces[state_name][self._indices] -= intra_result.forces

        delta_E = energies[self._state_names[0]] - energies[self._state_names[1]]
        self._cv_gradient = -(
            forces[self._state_names[0]] - forces[self._state_names[1]]
        )
        return delta_E

    def __deepcopy__(self, memo: dict[int, Any]) -> DeltaECV:
        """Copy configuration; state calculators belong to their owning CV."""
        copied = type(self)(
            self.query,
            " - ".join(self._state_names),
            deepcopy(self.forcefields, memo),
            deepcopy(self.charge_multiplicity_tuples, memo),
        )
        memo[id(self)] = copied
        return copied

    @staticmethod
    def _bypass_bias(calculator: Calculator) -> None:
        """Keep copied plugins, but avoid recursive PLUMED evaluation."""
        for plugin in calculator.active_plugins():
            if isinstance(plugin, PLUMED):
                plugin.bypass_plumed = True
        if isinstance(calculator, CompositeCalculator):
            for child in calculator.calculators:
                DeltaECV._bypass_bias(child)

    def chain_rule(
            self,
    ) -> NDArray[np.float64]:
        r"""Calculate change in atomic position per unit change in collective variable.

        Returns:
            An array of forces.
        """
        return self._cv_gradient
    
    def _build_aux_calculator(
            self,
            system: System,
            forcefield: str,
        ) -> Calculator:
        """
        Args:
            system: System object from the relevant Calculator.
            forcefield: .xml filename for forcefield for relevant
            charge state.

        Returns:
            Calculator with only the selected residue and no PBCs.
        """
        aux_system = system.copy_selection(self.query,keep_box=False)
        aux_hamiltonian = MMHamiltonian(
            forcefield=forcefield,
            nonbonded_method="NoCutoff",
        )
        aux_calculator = aux_hamiltonian.build_calculator(aux_system)
        return aux_calculator
