
"""Functionality for performing exclusions and generating State objects.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import time

import numpy as np

import openmm
from simtk.unit import elementary_charge
from simtk.unit import kilojoule_per_mole
from simtk.unit import nanometer

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def _generate_state(
        omm_context: openmm.Context,
        groups: set[int] | int | None = -1,
        get_energy: bool = True,
        get_forces: bool = True,
) -> openmm.State:
    """Generate an OpenMM State in order to collect energies and forces.

    Args:
        context: An OpenMM Context object containing a representation
            of the system and appropriate forces.
        groups: The force groups of the context to include in the
            State evaluation.

    Return:
        An OpenMM State object containing the energies and forces of
        the current state of the system represented within the Context
        object for the specified groups of forces.
    """
    if groups is None:
        groups = -1
    return omm_context.getState(getEnergy=get_energy, getForces=get_forces, groups=groups)


def _exclude_intramolecular(
        omm_system: openmm.System,
        atoms: frozenset[int],
) -> None:
    """Remove intramolecular interactions for a set of atoms.

