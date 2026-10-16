"""Plugins for wrapping system coordinates with a PBC.
"""
from __future__ import annotations

__all__ = [
    "CalculatorWrap",
    "IntegratorWrap",
]

from typing import TYPE_CHECKING

from collections import defaultdict

try: 
    import cupy as xp
    test = xp.asarray([0,1,2])
    del test
    USE_CUPY=True
except:
    import numpy as xp
    USE_CUPY=False

from pydft_qmmm.calculators.calculator import CalculatorPlugin
from pydft_qmmm.integrators.integrator import IntegratorPlugin

if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy.typing import NDArray
    from pydft_qmmm.integrators import Returns
    from pydft_qmmm.calculators import Results
    from pydft_qmmm import System

def _wrap_positions(
        positions: NDArray[np.float64],
        system: System,
        residue_indices: dict[list] = None,
#         inverse_box: NDArray[np.float64] = None,
) -> NDArray[np.float64]:
    r"""Wrap atom positions in accord with PBC.

    Atoms are wrapped to stay inside of the periodic box.  This
    function ensures molecules are not broken up by a periodic
    boundary, since OpenMM electrostatics will be incorrect if atoms
    in a molecule are not on the same side of the periodic box.

    Args:
        positions: The positions (:math:`\mathrm{\mathring{A}}`) to
            wrap.
        system: The system whose positions will be wrapped.

    Returns:
        The new wrapped positions of the system.
    """
    if residue_indices is None:
        residue_groups = defaultdict(list)
        for residue in system.residue_map.values():
            atoms = sorted(residue)
            residue_groups[len(atoms)].append(atoms)
        residue_indices = {
            n_atoms: xp.asarray(residues,dtype=xp.int32)
            for n_atoms, residues in residue_groups.items()    
        }

    positions_gpu = xp.asarray(positions)
    box = xp.asarray(system.box.base)
    inverse_box = xp.linalg.inv(box)

    new_positions = positions_gpu.copy()

    for indices in residue_indices.values(): 
        residue_positions = positions_gpu[indices]

        centroids = xp.mean(
            residue_positions,
            axis=1,
        )

        fractional_centroids = centroids @ inverse_box.T

        masks = xp.floor(fractional_centroids)

        shifts = -(masks @ box.T)

        wrapped = residue_positions + shifts[:, None, :]

        new_positions[indices] = wrapped
    
    if USE_CUPY:
        new_positions = xp.asnumpy(new_positions)
    return new_positions, residue_indices


class CalculatorWrap(CalculatorPlugin):
    """Wrap positions before performing a calculation.
    """
    def __init__(self,wrap_frequency: int = 20):
        super().__init__()
        self.residue_indices = None
        self._step = 0
        self._wrap_frequency = wrap_frequency

    def _modify_calculate(
            self,
            calculate: Callable[[bool, bool], Results],
    ) -> Callable[[bool, bool], Results]:
        """Modify the calculate routine to perform wrapping beforehand.

        Args:
            calculate: The calculation routine to modify.

        Returns:
            The modified calculation routine which implements the PBC
            coordinate-wrapping before calculation.
        """
        def inner(
                return_forces: bool = True,
                return_components: bool = True,
        ) -> Results:
            if self._step % self._wrap_frequency == 0:
                self.calculator.system.positions[:], self.residue_indices =\
                    _wrap_positions(
                        self.calculator.system.positions,
                        self.calculator.system,
                        self.residue_indices,
                    )
            self._step += 1
            results = calculate(return_forces, return_components)
            return results
        return inner


class IntegratorWrap(IntegratorPlugin):
    """Wrap positions after performing an integration.
    """
    def __init__(self,wrap_frequency: int = 20):
        super().__init__()
        self.residue_indices = None
        self._step = 0
        self._wrap_frequency = wrap_frequency

    def _modify_integrate(
            self,
            integrate: Callable[[System], Returns],
    ) -> Callable[[System], Returns]:
        """Modify the integrate routine to perform wrapping afterward.

        Args:
            integrate: The integration routine to modify.

        Returns:
            The modified integration routine which implements the PBC
            coordinate-wrapping after integration.
        """
        def inner(system: System) -> Returns:
            positions, velocities = integrate(system)
            if self._step % self._wrap_frequency == 0:
                positions, self.residue_indices = _wrap_positions(
                        positions,
                        system,
                        self.residue_indices
                    )
            self._step += 1
            return positions, velocities
        return inner
