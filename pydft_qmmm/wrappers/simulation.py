"""The class for organizing and running simulations.
"""
from __future__ import annotations

__all__ = ["Simulation"]

import logging
import time
import csv
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.calculators import CalculatorPlugin
from pydft_qmmm.utils import ELEMENT_TO_MASS
from pydft_qmmm.utils import Loggable
from pydft_qmmm.integrators import IntegratorPlugin
from pydft_qmmm.plugins import CalculatorCenter
from pydft_qmmm.plugins import CalculatorWrap
from pydft_qmmm.plugins import IntegratorWrap 
from pydft_qmmm.plugins import Stationary
from pydft_qmmm.plugins import Virtual
from pydft_qmmm.plugins import Plugin

if TYPE_CHECKING:
    from typing import Any
    from pydft_qmmm.calculators import Calculator
    from pydft_qmmm.hamiltonians import StandaloneHamiltonian
    from pydft_qmmm.integrators import Integrator
    from pydft_qmmm.system import System

logger = logging.getLogger(__name__)
logger.setLevel(20)


class Simulation(Loggable):
    """Organizes and performs simulations.

    Args:
        system: The system to simulate.
        integrator: The integrator describing how the system evolves.
        hamiltonian: The Hamiltonian describing how calculations are
            performed.
        calculator: A user-defined energy and force calculator, which
            may be provided in place of a Hamiltonian object.
        plugins: A list of plugins to apply to the calculator or
            integrator objects before simulation.
        kwargs: Additional options to provide to the logging
            functionality, see :class:`pydft_qmmm.utils.logging.Loggable`
            for a list of options.

    Attributes:
        _frame: The current frame of the simulation, defaults to zero
            upon instantiation.
    """
    _frame: int = 0

    def __init__(
            self,
            system: System,
            integrator: Integrator,
            hamiltonian: StandaloneHamiltonian | None = None,
            calculator: Calculator | None = None,
            plugins: list[Plugin] | None = None,
            **kwargs: Any,
    ) -> None:
        # Perform logging setup
        super().__init__(**kwargs)
        for handler in self.handlers:
            logger.addHandler(handler)
        # Get write intervals, if any
        self._write_interval = kwargs.get("log_write_interval",1)
        # TODO

        # Set calculator and build if necessary.
        if calculator is not None:
            self.calculator = calculator
        elif hamiltonian is not None:
            self.calculator = hamiltonian.build_calculator(system)
        else:
            raise TypeError
        # Perform additional simulation setup.
        self._offset = np.zeros(system.positions.shape)
        if system.box.any():
            pass
            self.calculator.register_plugin(CalculatorWrap(), 0)
#             integrator.register_plugin(IntegratorWrap(), 0)
        if system.select("subsystem I"):
            pass
            self.calculator.register_plugin(CalculatorCenter(), 0)

        integrator.register_plugin(Virtual(),0) 

        self.system = system
        self.integrator = integrator
        # Apply plugins.
        if plugins is not None:
            self._register_plugins(plugins)

    def run_dynamics(self, steps: int) -> None:
        """Perform a given number of simulation steps.

        Args:
            steps: The number of steps to take.
        """
        #### timing ####
#         with open('pydft_timing.csv',mode='a') as file:
# 
#             headers = ["step","step_time","integrate_time","write_time","log_time","calculate_time"]
#             writer = csv.writer(file)
#             writer.writerow(headers)
        ################

        self.calculate_energy_forces()
        for i in range(steps):
            #### timing ####
            start_time = time.perf_counter()
            ################

            new_positions, new_velocities = self.integrator.integrate(
                self.system,
            )
            #### timing ####
            integrate_time = time.perf_counter()
            ################
            self.system.positions = new_positions
            self.system.velocities = new_velocities
            #### timing ####
            write_time = time.perf_counter()
            ################

            unwrapped_positions = self.system.positions.base + self._offset

            logger.info(
                "",
                extra={
                    "frame": self._frame,
                    "positions": unwrapped_positions,
                    "box": self.system.box.base,
                },
            )

            #### timing ####
            log_time = time.perf_counter()
            ################

            self._frame += 1
            self.calculate_energy_forces()

            #### timing ####
#             calculate_time = time.perf_counter()
# 
#             step_delta_t = calculate_time - start_time
#             integrate_delta_t = integrate_time - start_time
#             write_delta_t = write_time - integrate_time 
#             log_delta_t = log_time - write_time 
#             calculate_delta_t = calculate_time - log_time
#             data = [i,step_delta_t,integrate_delta_t,write_delta_t,log_delta_t,calculate_delta_t]
#             with open('pydft_timing.csv',mode='a') as file:
#                 writer = csv.writer(file)
#                 writer.writerow(data)
            ################



    def calculate_energy_forces(self) -> None:
        """Update total system energy and forces on atoms in the system.
        """
        #### time ####
        time_0 = time.perf_counter()
        ##############

        temp = self.system.positions.base.copy()
        results = self.calculator.calculate(True,False)

        #### time ####
        time_1 = time.perf_counter()
        ##############

        self.system.forces = results.forces

        if self._frame % self._write_interval == 0:
            # we don't need to calculate this every timestep
            kinetic_energy = self.integrator.compute_kinetic_energy(
                self.system,
            )

            energy = {
                "Total Energy": kinetic_energy + results.energy,
                ".": {
                    "Kinetic Energy": kinetic_energy,
                    "Potential Energy": results.energy,
                    ".": results.components,
                },
            }

            self.energy = energy

            self._offset += temp - self.system.positions.base
            logger.info("", extra={"frame": self._frame, "energy": self.energy})

        time_2 = time.perf_counter()
        
        time_3 = time.perf_counter()
#         print(f"Calculator call: {time_1 - time_0:.5f}s")
#         print(f"Kinetic energy: {time_2 - time_1:.5f}s")

    def _register_plugins(self, plugins: list[Plugin]) -> None:
        """Dynamically load plugins for calculators and integrators.

        Args:
            plugins: A list of plugins to apply to the calculator or
                integrator objects.
        """
        for plugin in plugins:
            if isinstance(plugin, CalculatorPlugin):
                self.calculator.register_plugin(plugin)
            elif isinstance(plugin, IntegratorPlugin):
                self.integrator.register_plugin(plugin)
            else:
                raise TypeError
