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

try:
    import cupy as xp
    test = xp.asarray([0,1,2])
    del test
    USE_CUPY = True
except:
    import numpy as xp
    USE_CUPY = False

from numpy.typing import NDArray
import openmm
import openmm.unit

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray
    from numpy import float64
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


def _virtual_site_batches(system: System) -> list[tuple]:
    """Group independent virtual sites by type for vectorized operations."""
    indices = list(system.virtual_site_indices)
    types = list(system.virtual_types)
    parents = list(system.virtual_parents)
    weights = list(system.virtual_parent_weights)

    if not (len(indices) == len(types) == len(parents) == len(weights)):
        raise ValueError("Inconsistent virtual-site metadata lengths")

    virtual_indices = set(map(int, indices))
    groups = {"two_average": [], "three_average": []}

    for i, site_type in enumerate(types):
        if site_type not in groups:
            raise NotImplementedError(f"Unsupported virtual-site type: {site_type}")
        if any(int(parent) in virtual_indices for parent in parents[i]):
            raise NotImplementedError("Virtual sites depending on other virtual sites")
        expected = 2 if site_type == "two_average" else 3
        if len(parents[i]) != expected or len(weights[i]) != expected:
            raise ValueError(f"Invalid {site_type} site at index {indices[i]}")
        groups[site_type].append(i)

    batches = []
    for rows in groups.values():
        if not rows:
            continue
        site_indices = xp.asarray([indices[i] for i in rows], dtype=xp.int64)
        parent_indices = xp.stack(
            [xp.asarray(parents[i],dtype=xp.int64) for i in rows]
        )
        parent_weights = xp.stack(
            [xp.asarray(weights[i],dtype=xp.float64) for i in rows]
        )
        batches.append((site_indices, parent_indices, parent_weights))
    return batches


def compute_positions(system: System, positions: NDArray[float64]) -> NDArray[float64]:
    """Compute virtual-site coordinates in batches grouped by site type.
    """
    new_positions = xp.asarray(positions).copy()
    if len(new_positions) != len(system):
        raise ValueError("Position array length does not match system")

    for indices, parents, weights in _virtual_site_batches(system):
        # (n_sites, n_parents, 3) -> (n_sites, 3)
        new_positions[indices] = xp.sum(
            new_positions[parents] * weights[:, :, None], axis=1
        )
        if USE_CUPY:
            new_positions = xp.asnumpy(new_positions)
    return new_positions


def distribute_forces(system: System, forces: NDArray[float64]) -> NDArray[float64]:
    """Backpropagate virtual-site forces to parents with indexed scatter-add.

    Does not modify input. As in the original implementation, virtual-site
    forces are not zeroed.
    """
    new_forces = xp.asarray(forces).copy()
    if len(new_forces) != len(system):
        raise ValueError("Force array length does not match system")

    for indices, parents, weights in _virtual_site_batches(system):
        # Flatten to permit duplicate parent indices, which ordinary indexed
        # '+=' does NOT accumulate correctly for NumPy or CuPy.
        contributions = (
            new_forces[indices, None, :] * weights[:, :, None]
        ).reshape(-1, 3)
        xp.add.at(new_forces, parents.ravel(), contributions)
        if USE_CUPY:
            new_forces = xp.asnumpy(new_forces)
    return new_forces


def extract_virtual_sites(system: openmm.System) -> tuple[VirtualSite]:
    """Extract supported virtual sites from an OpenMM System."""
    sites = []
    for index in range(system.getNumParticles()):
        if not system.isVirtualSite(index):
            continue
        site = system.getVirtualSite(index)
        particles = tuple(
            int(site.getParticle(i)) for i in range(site.getNumParticles())
        )
        if any(system.isVirtualSite(parent) for parent in particles):
            raise NotImplementedError("Virtual sites depending on other virtual sites")
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
