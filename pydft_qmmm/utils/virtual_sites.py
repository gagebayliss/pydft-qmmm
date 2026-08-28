"""OpenMM-compatible virtual-site position construction."""
from __future__ import annotations

__all__ = [
    "VirtualSite",
    "extract_virtual_sites",
    "compute_positions",
    "distribute_forces",
]

from dataclasses import dataclass
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import openmm
import openmm.unit

from pydft_qmmm.utils.misc import wrap_positions
from pydft_qmmm.utils.misc import minimum_image_displacement

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray
    from pydft_qmmm.system import System
    from .variable import ObservedArray
    from .variable import array_float
    from .variable import array_int
    from .variable import ArrayValue

@dataclass(frozen=True)
class VirtualSite:
    """The virtual site data container.
    
    Args:
        index: The index of the virtual site in the total system.
        site_type: The kind of virtual site (three average, etc.).
        parents: The real or virtual atoms the site is derived from.
        parent_weights: Weights used for averaging parent positions.
    """

    index: int
    site_type: str
    parents: tuple[int, ...]
    parent_weights: tuple[Any, ...]


def compute_positions(system: System, positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Computes virtual site positions from parent atom positions.

    This method does not mutate its arguments.

    Args:
        system: a PyDFT-QMMM system.
        positions: an array of atomic positions, including virtual sites.

    Returns:
        an array containing updated positions.
    """
    new_positions = np.asarray(positions).copy()
    assert len(new_positions) == len(system)

    for i, site_index in enumerate(system.virtual_site_indices):
        if system.virtual_types[i] == "two_average":
            pos1 = new_positions[system.virtual_parents[i][0]]
            pos2 = new_positions[system.virtual_parents[i][1]]
            v12 = pos2 - pos1
            # v12 = minimum_image_displacement(v12,system.box)

            w1 = system.virtual_parent_weights[i][0]
            w2 = system.virtual_parent_weights[i][1]

            assert np.isclose(w1 + w2, 1.0,atol=1e-2)

            # relative to first virtual parent
            virtual_displacement = v12 * w2

            new_positions[site_index] = pos1 + virtual_displacement

        elif system.virtual_types[i] == "three_average":
            pos1 = new_positions[system.virtual_parents[i][0]]
            pos2 = new_positions[system.virtual_parents[i][1]]
            pos3 = new_positions[system.virtual_parents[i][2]]
            v12 = pos2 - pos1
            v13 = pos3 - pos1
            # v12 = minimum_image_displacement(v12,system.box)
            # v13 = minimum_image_displacement(v13,system.box)

            w1 = system.virtual_parent_weights[i][0]
            w2 = system.virtual_parent_weights[i][1]
            w3 = system.virtual_parent_weights[i][2]

            assert np.isclose(w1 + w2 + w3, 1.0,atol=1e-2)

            virtual_displacement = v12 * w2 + v13 * w3

            new_positions[site_index] = pos1 + virtual_displacement

        elif system.virtual_types[i] == "out_of_plane":
            raise NotImplementedError()
        elif system.virtual_types[i] == "local_coordinate":
            raise NotImplementedError()
        elif system.virtual_types[i] == 'symmetry':
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    return new_positions

def distribute_forces(system: System, forces: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Computes virtual site forces from parent atom forces.

    This method does not mutate its arguments.

    Calling this function multiple times on the same forces
        will cause problems because virtual site forces are 
        not zeroed.

    Args:
        system: a PyDFT-QMMM system.
        forces: an array of atomic forces, including virtual sites.

    Returns:
        an array containing updated forces. 
    """
    if not len(system.virtual_site_indices):
        return np.asarray(forces).copy()

    assert (len(forces) == len(system))

    new_forces = np.asarray(forces).copy()

    for i,site_index in enumerate(system.virtual_site_indices):
        force = new_forces[site_index]
        if system.virtual_types[i] == "two_average":
            p1 = system.virtual_parents[i][0]
            p2 = system.virtual_parents[i][1]
            w1 = system.virtual_parent_weights[i][0]
            w2 = system.virtual_parent_weights[i][1]
            new_forces[p1] += force * w1
            new_forces[p2] += force * w2

        elif system.virtual_types[i] == "three_average":
            p1 = system.virtual_parents[i][0]
            p2 = system.virtual_parents[i][1]
            p3 = system.virtual_parents[i][2]
            w1 = system.virtual_parent_weights[i][0]
            w2 = system.virtual_parent_weights[i][1]
            w3 = system.virtual_parent_weights[i][2]
            new_forces[p1] += force * w1
            new_forces[p2] += force * w2
            new_forces[p3] += force * w3

        elif system.virtual_types[i] == "out_of_plane":
            raise NotImplementedError()
        elif system.virtual_types[i] == "local_coordinate":
            raise NotImplementedError()
        elif system.virtual_types[i] == 'symmetry':
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    return new_forces


def _dependency_order(system: openmm.System) -> list[int]:
    """Topologically order virtual sites like OpenMM ReferenceVirtualSites."""
    remaining = {
        i for i in range(system.getNumParticles()) if system.isVirtualSite(i)
    }
    order = []
    while remaining:
        previous_size = len(remaining)
        for index in sorted(tuple(remaining)):
            site = system.getVirtualSite(index)
            dependencies = {site.getParticle(i) for i in range(site.getNumParticles())}
            if dependencies.isdisjoint(remaining):
                order.append(index)
                remaining.remove(index)
        if len(remaining) == previous_size:
            raise ValueError("Virtual site definitions are circular.")
    return order

def extract_virtual_sites(system: openmm.System) -> tuple[VirtualSite]:
    """Extract supported virtual sites from an OpenMM System."""
    sites = []
    for index in _dependency_order(system):
        site = system.getVirtualSite(index)
        particles = tuple(
            int(site.getParticle(i)) for i in range(site.getNumParticles())
        )
        if isinstance(site, openmm.TwoParticleAverageSite):
            sites.append(VirtualSite(
                index, "two_average", particles,
                tuple(float(site.getWeight(i)) for i in range(2)),
            ))
        elif isinstance(site, openmm.ThreeParticleAverageSite):
            sites.append(VirtualSite(
                index, "three_average", particles,
                tuple(float(site.getWeight(i)) for i in range(3)),
            ))
        else:
            raise TypeError(
                f"Unsupported OpenMM virtual-site type: {type(site).__name__}",
            )
    return tuple(sites)