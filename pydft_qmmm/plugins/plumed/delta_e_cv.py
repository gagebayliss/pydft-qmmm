from __future__ import annotations

__all__ = ["DeltaECV"]

from typing import TYPE_CHECKING

import time

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
        time_0 = time.perf_counter()
        system = calculator.system
        
        original_charges = system.charges.base.copy()
        original_positions = system.positions.base.copy()

        energies = {}
        forces = {}

        if self._indices is None:
            self._indices = [i for i in sorted(system.select(self.query))]
        time_1 = time.perf_counter()
#         print(f" copying system arrays and sorting indices: {time_1 - time_0:.5f}")

        for state_name in self._state_names:
            time_00 = time.perf_counter()
            if self._aux_calculators.get(state_name,None) is None:
                self._aux_calculators[state_name] = self._build_aux_calculator(
                        system,
                        self.forcefields[state_name],
                )
            if self._charge_arrays.get(state_name,None) is None:
                self._charge_arrays[state_name] = self._aux_calculators[state_name]\
                    .system.charges.base.copy()
            time_01 = time.perf_counter()
#             print(f"building calculators: {time_01 - time_00:.5f}")

#             print(f"####################")
            time_000 = time.perf_counter()
            test_charges = system.charges.base.copy()
            test_charges[self._indices] = self._charge_arrays[state_name]
            time_001 = time.perf_counter()
#             print(f"make test charges: {time_001 - time_000:.5f}")
            if np.allclose(test_charges, original_charges):
                time_002 = time.perf_counter()
#                 print(f"evaluate conditional : {time_002 - time_001:.5f}")
#                 print("charges recognized, not... redoing calculation")
                result = results
#                 intra_result = self._aux_calculators[state_name].calculate() 
            else:
                time_002 = time.perf_counter()
#                 print(f"conditional : {time_002 - time_001:.5f}")
#                 system.charges[self._indices] = self._charge_arrays[state_name]
                time_003 = time.perf_counter()
#                 print(f"charge update: {time_003 - time_002:.5f}")

                system.charges[:] = test_charges
                result = calculator.calculate()
#                 intra_result = self._aux_calculators[state_name].calculate() 

#                 intra_result = self._aux_calculators[state_name].calculate() 
                time_004 = time.perf_counter()
#                 print(f"calculation: {time_004 - time_003:.5f}")
                

#             print(f"####################")
            time_02 = time.perf_counter()
#             print(f"calculation and charge update: {time_02 - time_01:.5f}")
                
            intra_result = self._aux_calculators[state_name].calculate()
            energies[state_name] = result.energy - intra_result.energy
            forces[state_name] = result.forces.copy() 
            forces[state_name][self._indices] = forces[state_name][self._indices]\
                                          - intra_result.forces
#             energies[state_name] = result.energy
#             forces[state_name] = result.forces.copy()
            time_03 = time.perf_counter()
#             print(f"intra calculation and subtraction: {time_03 - time_02:.5f}")
            
        time_6 = time.perf_counter()

        system.charges[:] = original_charges # this takes basically all the time...

        delta_E = energies[self._state_names[0]] - energies[self._state_names[1]]

        # used in chain_rule()
        self._cv_gradient = -(forces[self._state_names[0]]\
                                - forces[self._state_names[1]])

        time_final = time.perf_counter()
#         print(f"all the other stuff: {time_final - time_6:.5f}")
#         print(f"total time for delta E cv scalar cv: {time_final - time_0:.4f}")
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
