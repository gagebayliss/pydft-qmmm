from __future__ import annotations

__all__ = ["DeltaECV"]

from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.hamiltonians import MMHamiltonian

from .collective_variable import CollectiveVariable

if TYPE_CHECKING:
    from pydft_qmmm.calculators import Calculator
    from pydft_qmmm.calculators import Results
    from pydft_qmmm.system import System
    from numpy.typing import NDArray


class DeltaECV(CollectiveVariable):
    """The generalized solvation collective variable class.

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
        self._charge_arrays = {}

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
        
        original_charges = system.charges.base.copy()
        original_positions = system.positions.base.copy()

        energies = {}
        forces = {}

        indices = [i for i in sorted(system.select(self.query))]

        for state_name in self._state_names:
            if self._aux_calculators.get(state_name,None) is None:
                self._aux_calculators[state_name] = self._build_aux_calculator(
                        system,
                        self.forcefields[state_name],
                )
            if self._charge_arrays.get(state_name,None) is None:
                self._charge_arrays[state_name] = self._aux_calculators[state_name]\
                    .system.charges.base.copy()

        
            system.charges[indices] = self._charge_arrays[state_name]
            if np.allclose(system.charges.base, original_charges, atol=1e-8):
                result = results
            else:
                result = calculator.calculate()

            self._aux_calculators[state_name].system.positions[:] =\
                                        calculator.system.positions[indices]
            intra_result = self._aux_calculators[state_name].calculate()
            
#             print("///////////////////////////////////////////////")
#             print(f"STATE: {state_name}")
#             print(f"FULL ENERGY: {result.energy:.4f}")
#             print(f"INTRAMOLECULAR ENERGY: {intra_result.energy:.4f}")
#             print("///////////////////////////////////////////////")

            energies[state_name] = result.energy - intra_result.energy
            forces[state_name] = result.forces 
            forces[state_name][indices] = forces[state_name][indices]\
                                          - intra_result.forces

            system.charges[:] = original_charges
            system.positions[:] = original_positions

        delta_E = energies[self._state_names[0]] - energies[self._state_names[1]]

        # used in chain_rule()
        self._cv_gradient = -(forces[self._state_names[0]]\
                                - forces[self._state_names[1]])

        return delta_E

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
