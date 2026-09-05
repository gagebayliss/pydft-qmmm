"""Classes for using arbitrary collective variables in Plumed.
"""
from __future__ import annotations

__all__ = ["CollectiveVariable"]

from typing import TYPE_CHECKING
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

if TYPE_CHECKING:
    from pydft_qmmm.calculators import Calculator
    from pydft_qmmm.calculators import Results
    from numpy.typing import NDArray

class CollectiveVariable(ABC):
    """The abstract collective variable base class.
    """
    @abstractmethod
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

    @abstractmethod
    def chain_rule(
            self,
    ) -> NDArray[np.float64]:
        r"""Calculate change in atomic position per unit change in collective variable.

        Returns:
            An array of forces.
        """