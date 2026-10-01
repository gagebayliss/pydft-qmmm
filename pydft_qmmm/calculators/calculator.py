"""Base classes for calculators and related classes.

This module contains the base classes for calculators and calculator
plugins, as well as the results class and the components type.
"""
from __future__ import annotations

__all__ = [
    "Calculator",
    "CalculatorPlugin",
    "Components",
    "Results",
]

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeAlias

import numpy as np

from pydft_qmmm.utils import pluggable_method

if TYPE_CHECKING:
    from pydft_qmmm import System
    from numpy.typing import NDArray

Components: TypeAlias = dict[str, Any]


@dataclass
class Results:
    r"""Store the results of a calculation.

    Args:
        energy: The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of a system
            determined by a calculator.
        forces: The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            on a system determined by a calculator.
        components: The energy components
            (:math:`\mathrm{kJ\;mol^{-1}}`) of a system determined by a
            calculator.
    """
    energy: float = 0
    forces: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0),
    )
    components: Components = field(
        default_factory=dict,
    )


@dataclass(frozen=True)
class Calculator(ABC):
    """The abstract calculator base class.

    Args:
        system: The system whose atom positions, atom identities, and
            geometry will be used to calculate energies and forces.
        calculator_group: label, one of QM, MM, or Composite.

    Attributes:
        _plugins: The list of plugins that have been registered by the
            calculator.
    """
    system: System
    calculator_group: str = field(kw_only=True)
    _plugins: list[CalculatorPlugin] = field(default_factory=list, init=False)

    def copy(self) -> Calculator:
        """Copy the calculator, its system, potentials, and all plugins.

        Shared references within a composite calculator remain shared
        within the copy. Plugin state is independent and references to
        calculators are rebound to their copies. Backend caches and
        method wrappers are rebuilt instead of retaining old bindings.
        Native plugin engines may be recreated; PLUMED bias history is
        not cloned.
        """
        return deepcopy(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> Calculator:
        """Copy state without copying closures bound to this calculator."""
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        object.__setattr__(copied, "system", deepcopy(self.system, memo))
        for name, value in vars(self).items():
            if name != "system" and not name.startswith("__wrapped_"):
                object.__setattr__(copied, name, deepcopy(value, memo))
        return copied

    @pluggable_method
    @abstractmethod
    def calculate(
            self,
            return_forces: bool = True,
            return_components: bool = True,
    ) -> Results:
        r"""Calculate energies and forces.

        Args:
            return_forces: Whether or not to return forces.
            return_components: Whether or not to return the components
                of the energy.

        Returns:
            A wrapper containing the energy
            (:math:`\mathrm{kJ\;mol^{-1}}`), forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`), and
            energy components (:math:`\mathrm{kJ\;mol^{-1}}`) of the
            calculation.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the calculator, for logging purposes."""

    def register_plugin(
            self,
            plugin: CalculatorPlugin,
            index: int | None = None,
    ) -> None:
        """Record plugin and apply it to the calculator.

        Args:
            plugin: A plugin that will modify the behavior of the
                calculator's pluggable methods.
            index: The index at which to insert the plugin in the
                plugin load order, i.e., 0 will be called first,
                1 will be called second, etc.  The default behavior
                is to append the plugin at the end of the plugin
                load order.
        """
        if index is not None:
            self._plugins.insert(index, plugin)
        else:
            self._plugins.append(plugin)
        plugin.modify(self)

    def active_plugins(self) -> list[CalculatorPlugin]:
        """Get the current list of active plugins.

        Returns:
            A list of the active plugins registered by the calculator.
        """
        return self._plugins


class CalculatorPlugin(ABC):
    """The abstract base class for modifying calculator routines.
    """

    def modify(self, calculator: Calculator) -> None:
        """Modify the functionality of a calculator.

        Args:
            calculator: The calculator whose functionality will be
                modified by the plugin.
        """
        self.calculator = calculator
