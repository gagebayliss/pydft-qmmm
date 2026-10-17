"""Electrostatic embedding for PySCF."""
from __future__ import annotations

__all__ = [
    "add_finite_embedding",
    "finite_embedding_forces",
]

from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from .pyscf_backend import load_submodule
from .pyscf_backend import to_numpy
from .pyscf_utils import spin_sum

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numpy.typing import NDArray
    from pydft_qmmm import System


def add_finite_embedding(
        backend: Any,
        method: Any,
        system: System,
        embed_indices: Sequence[int],
) -> Any:
    r"""Decorate a solver with the finite point-charge environment.

    Args:
        backend: The package providing the solvers, whose own QM/MM
            module has to be used: GPU4PySCF's solvers do not inherit
            from PySCF's, so PySCF's decorator rejects them.
        method: The molecular PySCF solver to decorate.
        system: The system holding the environment positions
            (:math:`\mathrm{\mathring{A}}`) and charges (:math:`e`).
        embed_indices: The original system indices of the point charges
            to embed.

    Returns:
        The decorated solver, or the original solver when there is
        nothing to embed.
    """
    indices = list(embed_indices)
    if not indices:
        return method
    qmmm = load_submodule(backend, "qmmm")
    return qmmm.add_mm_charges(
        method,
        np.asarray(system.positions)[indices],
        np.asarray(system.charges)[indices],
        unit="Angstrom",
    )


def finite_embedding_forces(
        method: Any,
        dm: NDArray[np.float64],
        qm_indices: Sequence[int],
        embed_indices: Sequence[int],
        natoms: int,
) -> NDArray[np.float64]:
    r"""Assemble the QM and MM gradients into full-system forces.

    Args:
        method: The converged PySCF solver.
        dm: The converged density matrix.
        qm_indices: The original system indices of subsystem I, in the
            order they appear in the PySCF molecule.
        embed_indices: The original system indices of the embedded
            point charges, in the order they were embedded.
        natoms: The number of atoms in the full system.

    Returns:
        The forces (:math:`\mathrm{E_h\;a_0^{-1}}`) acting on atoms in
        the system.
    """
    forces = np.zeros((natoms, 3))
    gradient = method.nuc_grad_method()
    gradient.verbose = 0
    # Atom-centered DFT grids move with the nuclei. Include their response
    # so forces differentiate the actual discretized energy used in MD.
    if hasattr(gradient, "grid_response"):
        gradient.grid_response = True
    forces[list(qm_indices)] -= to_numpy(gradient.kernel())
    if len(embed_indices):
        forces[list(embed_indices)] -= to_numpy(
            gradient.grad_hcore_mm(spin_sum(dm)),
        ) + to_numpy(gradient.grad_nuc_mm())
    return forces
