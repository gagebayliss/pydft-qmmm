"""Drude SCF solver."""
from __future__ import annotations

__all__ = [
    "DrudeSCFIterator",
    "CompositeSCF"
]

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .drude_data import DrudeData


class DrudeSCFIterator:
    def __init__(self,calculator,data,force_tolerance):
        self.calculator = calculator
        self.data = data
        self.force_tolerance = force_tolerance
        self.stopping_ratio = 0.90
        self._forces = np.array([np.inf])
        self._last_forces = np.array([np.inf])
        self._history = {
            "drude_positions": [],
            "drude_forces": []
        }

    def step(self) -> None:
        """Step Drude positions.
        Based on OpenMM Drude reference kernel.
        """
        print("in DrudeSCFIterator")
        self._last_forces = self._forces
        forces = self.calculate()
        self._forces = forces.copy()
        displacement_ang = (
            forces
            / self.data.force_constants.reshape((-1, 1))
        )
        # matches OpenMM implementation
        forces_squared = np.sum(forces * forces, axis=1)
        damping_mask = np.where(forces_squared > 10 * self.force_tolerance)
        displacement_ang[damping_mask] = displacement_ang[damping_mask] * 0.5
        self.state = self.state + displacement_ang

        self._history["drude_positions"].append(self.state)
        self._history["drude_forces"].append(forces)
        
    
    def calculate(self) -> NDArray[np.float64]:
        results = self.calculator.calculate()
        forces = results.forces[self.data.drude_indices,:]
        return forces
    
    def is_converged(self):
        if self.objective_function <= self.force_tolerance:
            return True
        elif self.stopping_ratio is not None:
            force_squared = float(np.sum(self._forces**2))
            last_force_squared = float(np.sum(self._last_forces**2))
            if (force_squared > self.stopping_ratio*last_force_squared):
                return True
        else:
            return False
    
    @property
    def objective_function(self) -> NDArray[np.float64]:
        return np.sqrt(np.mean(self._last_forces**2))
    
    @property
    def state(self) -> NDArray[np.float64]:
        return self.calculator.system.positions[self.data.drude_indices]
    
    @state.setter
    def state(self,drude_coordinates) -> None:
        self.calculator.system.positions[self.data.drude_indices] = drude_coordinates
    
    @property
    def history(self) -> dict[str, list[NDArray[np.float64]]]:
        return self._history
    
    def reset_history(self) -> None:
        self._history = {
            "drude_positions": [],
            "drude_forces": []
        }

class CompositeSCF:
    def __init__(self,scf_iterator,max_iterations):
        self.scf_iterator = scf_iterator
        self.max_iterations = max_iterations
        self._is_converged = False
    
    def solve(self) -> None:
        self._is_converged = False
        self.scf_iterator.reset_history()
        for iteration in range(1, self.max_iterations + 1):
            self.scf_iterator.step()
            if self.scf_iterator.is_converged():
                self._is_converged = True
                return
        self._is_converged = False
        
    def is_converged(self) -> bool:
        return self._is_converged
    

