"""Drude SCF solver."""
from __future__ import annotations

__all__ = [
    "DrudeCalculator",
    "CompositeSCF"
]

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .drude_data import DrudeData

class DrudeCalculator:
    def __init__(self,calculator,drude_data):
        self.calculator = calculator
        self.indices = drude_data.drude_indices
        self._gradient = None
    
    def calculate(
            self,
            positions: NDArray[np.float64],
        ) -> float:
        """Compute the energy for a given configuration of Drudes.

        Args: 
            positions: A shape(3 * n_atoms) array representing
                the positions of the drude oscillators.

        Returns:
            The scalar value of the energy.
        """
        self.state = positions
        results = self.calculator.calculate() 
        
        return results.energy, -results.forces[self.indices].reshape(-1)
    
    @property
    def state(self) -> NDArray[np.float64]:
        return self.calculator.system.positions[self.indices].reshape(-1)
    
    @state.setter
    def state(self,drude_coordinates) -> None:
        self.calculator.system.positions[self.indices] = drude_coordinates.reshape(-1,3)

class CompositeSCF:
    def __init__(
            self,
            calculator,
            max_iterations: int = 50,
            algorithm: str = "L-BFGS-B",
            force_tolerance: float = 10.0,
        ) -> None:
        self.calculator = calculator
        self.max_iterations = max_iterations
        self.algorithm = algorithm
        self.force_tolerance = force_tolerance
        self._is_converged = False
    
    def solve(self) -> None:
        try:
            import scipy
        except:
            raise ValueError("Missing SciPy.")
        
        result = scipy.optimize.minimize(
            self.calculator.calculate,
            self.calculator.state,
            jac=True,
            method=self.algorithm,
            options={
                "gtol": self.force_tolerance,
                "maxiter" : self.max_iterations,
            },
        )
        return result
        
    def is_converged(self) -> bool:
        return self._is_converged
    

