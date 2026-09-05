"""For parsing .xml forcefield files."""
from __future__ import annotations

__all__ = ["load_omm_system"]

from typing import TYPE_CHECKING

import numpy as np

import openmm

if TYPE_CHECKING:
    import openmm.System
    from pydft_qmmm.system import System

def load_omm_system(system: System, forcefield: list[str] | str) -> openmm.System:
    """Load an OpenMM system from an XML file.

    Args:
        system: The system which will be tied to the OpenMM interface.
        forcefield: The files containing forcefield and topology data for the system.
    """
    if isinstance(forcefield, str):
        forcefield = [forcefield]
    omm_box = [openmm.Vec3(*x)*openmm.unit.angstrom for x in system.box.T]
    if not all(x := [fh.endswith(".xml") for fh in forcefield]):
        raise ValueError("...")

    omm_topology = _build_omm_topology(system, forcefield)
    if np.any(system.box):
        omm_topology.setPeriodicBoxVectors(omm_box)
    omm_modeller = _build_omm_modeller(system, omm_topology)
    omm_forcefield = _build_omm_forcefield(forcefield, omm_modeller)
    omm_system = _build_omm_system(omm_forcefield, omm_modeller)
    return omm_system

def _build_omm_topology(
        system: System,
        forcefield: list[str],
) -> openmm.app.Topology:
    """Build the OpenMM Topology object.

    Args:
        system: The system which will be tied to the OpenMM interface.
        forcefield: The files containing forcefield  and topology
            data for the system.

    Returns:
        The internal representation of system topology for OpenMM.
    """
    for fh in forcefield:
        openmm.app.Topology.loadBondDefinitions(fh)
    omm_topology = openmm.app.Topology()
    chains = {x: omm_topology.addChain(x) for x in np.unique(system.chains)}
    residue_map = system.residue_map
    for i in residue_map.keys():
        atoms = sorted(residue_map[i])
        residue = omm_topology.addResidue(
            system.residue_names[atoms[0]],
            chains[system.chains[atoms[0]]],
        )
        for j in atoms:
            element = None
            if system.elements[j].upper() not in {"EP", "LP"}:
                element = openmm.app.Element.getBySymbol(system.elements[j])
            _ = omm_topology.addAtom(
                system.names[j],
                element,
                residue,
            )
    omm_topology.createStandardBonds()
    return omm_topology


def _build_omm_modeller(
        system: System,
        omm_topology: openmm.app.Topology,
) -> openmm.app.Modeller:
    """Build the OpenMM Modeller object.

    Args:
        system: The system which will be tied to the OpenMM interface.
        omm_topology: The OpenMM representation of system topology.

    Returns:
        The internal representation of the system OpenMM, integrating
        the topology and atomic positions.
    """
    omm_pos = openmm.unit.Quantity(
        [openmm.Vec3(*x) for x in system.positions],
        openmm.unit.angstrom,
    )
    omm_modeller = openmm.app.Modeller(omm_topology, omm_pos)
    return omm_modeller

def _build_omm_forcefield(
        forcefield: list[str],
        omm_modeller: openmm.app.Modeller,
) -> openmm.app.ForceField:
    """Build the OpenMM ForceField object.

    Args:
        forcefield: The files containing forcefield  and topology
            data for the system.
        omm_modeller: The OpenMM representation of the system.

    Returns:
        The internal representation of the force field for OpenMM.
    """
    omm_forcefield = openmm.app.ForceField(*forcefield)
    omm_modeller.addExtraParticles(omm_forcefield)
    return omm_forcefield


def _build_omm_system(
        omm_forcefield: openmm.app.ForceField,
        omm_modeller: openmm.app.Modeller,
) -> openmm.System:
    """Build the OpenMM System object.

    Args:
        omm_forcefield: The OpenMM representation of the forcefield.
        omm_modeller: The OpenMM representation of the system.

    Returns:
        The internal representation of forces, constraints, and
        particles for OpenMM.
    """
    omm_system = omm_forcefield.createSystem(
        omm_modeller.topology,
        rigidWater=False,
    )
    return omm_system

