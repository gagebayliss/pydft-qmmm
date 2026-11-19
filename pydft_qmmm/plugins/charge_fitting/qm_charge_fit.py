"""Plugin for fitting MM charges in subsystem I to QM electron density."""
from __future__ import annotations

__all__ = ["QMChargeFit"]

from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.calculators import CalculatorPlugin
from pydft_qmmm.calculators import PotentialCalculator
from pydft_qmmm.calculators import CompositeCalculator

if TYPE_CHECKING:
    from collections.abc import Callable
    from pydft_qmmm.calculators import Calculator
    from pydft_qmmm.calculators import Results

class QMChargeFit(CalculatorPlugin):
    """Fit subsystem I charges after the QM calculation."""
    def __init__(self,charge_fitting_scheme: str = "mbis"):
        super().__init__()
        self.charge_fitting_scheme = charge_fitting_scheme
                                            
    ### register
    def modify(self, calculator: Calculator):
        self.calculator = calculator 
        if isinstance(calculator, CompositeCalculator):
            qm_calculators = [
                calc for calc in calculator.calculators
                if calc.calculator_group == "QM"
            ]
            for qm_calculator in qm_calculators:
                qm_calculator.register_plugin(
                    QMChargeFit(self.charge_fitting_scheme)
                )

    def _modify_calculate(
            self,
            calculate: Callable[[bool, bool], Results],
        ) -> Callable[[bool, bool], Results]:
        """Modify the calculate routine to fit subsystem I charges to
        QM electron density.

        Args:
            calculate: The calculation routine to modify.

        Returns:
            The modified calculation routine which fits mm charges to
            QM density.
        """
        def inner(
                return_forces: bool = True,
                return_components: bool = True,
            ) -> Results:
            results = calculate(return_forces, return_components)
            # The instance registered to CompositeCalculator is a no-op here
            if isinstance(self.calculator, PotentialCalculator):
                # has to run before modifying charges because
                # generate_wavefunction uses system_cache
                qm_charges = self.calculator.potential.fit_mm_charges(
                    self.charge_fitting_scheme,
                )
                # we do this because of virtual sites / Drudes
                ep_indices = np.array(sorted(
                    self.calculator.system.select(
                        "subsystem I and element Ep Lp"
                    )
                ), dtype=int)
                self.calculator.system.charges[ep_indices] = 0.0

                qm_nuclei_indices = np.array(sorted(
                    self.calculator.system.select(
                        "subsystem I and not element Ep Lp"
                    )
                ), dtype=int)
                self.calculator.system.charges[qm_nuclei_indices] = qm_charges
            return results
        return inner
