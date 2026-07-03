"""OpenMM-compatible virtual-site position construction."""
from __future__ import annotations

__all__ = ["VirtualSiteData", "extract_virtual_sites"]

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
import openmm
import openmm.unit


@dataclass(frozen=True)
class _VirtualSite:
    """A single virtual-site definition in dependency order."""

    index: int
    kind: str
    particles: tuple[int, ...]
    parameters: tuple[Any, ...]


@dataclass(frozen=True)
class VirtualSiteData:
    """Virtual sites extracted from an OpenMM System."""

    sites: tuple[_VirtualSite, ...]

    @property
    def indices(self) -> NDArray[np.int64]:
        """Return virtual-site particle indices."""
        return np.asarray([site.index for site in self.sites], dtype=np.int64)

    def compute_positions(
            self,
            positions: NDArray[np.float64],
            box: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute virtual-site coordinates in dependency order.

        Args:
            positions: Particle positions in Angstrom.
            box: Box vectors in Angstrom, stored as row vectors.  This is
                required only by a periodic ``SymmetrySite``.
        """
        result = np.array(positions, dtype=float, copy=True)
        for site in self.sites:
            p = site.particles
            if site.kind in {"two_average", "three_average"}:
                weights = np.asarray(site.parameters[0])
                result[site.index] = weights @ result[list(p)]
            elif site.kind == "out_of_plane":
                w12, w13, wcross = site.parameters
                v12 = result[p[1]] - result[p[0]]
                v13 = result[p[2]] - result[p[0]]
                result[site.index] = (
                    result[p[0]] + w12*v12 + w13*v13
                    + wcross*np.cross(v12, v13)
                )
            elif site.kind == "local_coordinates":
                origin_weights, x_weights, y_weights, local_position = (
                    site.parameters
                )
                source = result[list(p)]
                origin = np.asarray(origin_weights) @ source
                xdir = np.asarray(x_weights) @ source
                ydir = np.asarray(y_weights) @ source
                zdir = np.cross(xdir, ydir)
                norm_x = np.linalg.norm(xdir)
                norm_z = np.linalg.norm(zdir)
                if norm_x > 0:
                    xdir /= norm_x
                if norm_z > 0:
                    zdir /= norm_z
                ydir = np.cross(zdir, xdir)
                local = np.asarray(local_position)
                result[site.index] = (
                    origin + local[0]*xdir + local[1]*ydir + local[2]*zdir
                )
            elif site.kind == "symmetry":
                rotation, offset, use_box = site.parameters
                coordinate = result[p[0]]
                if use_box:
                    if box is None or not np.any(box):
                        raise ValueError(
                            "A periodic SymmetrySite requires box vectors.",
                        )
                    coordinate = coordinate @ np.linalg.inv(box)
                coordinate = np.asarray(rotation) @ coordinate + offset
                if use_box:
                    coordinate = coordinate @ box
                result[site.index] = coordinate
            else:  # pragma: no cover - guarded by extraction
                raise TypeError(f"Unsupported virtual-site type: {site.kind}")
        return result


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


def _vec3(value: Any, unit: Any | None = None) -> tuple[float, float, float]:
    """Convert an OpenMM Vec3 or Quantity<Vec3> to a float tuple."""
    if unit is not None:
        value = value.value_in_unit(unit)
    return tuple(float(value[i]) for i in range(3))


def extract_virtual_sites(system: openmm.System) -> VirtualSiteData:
    """Extract supported virtual sites from an OpenMM System."""
    sites = []
    for index in _dependency_order(system):
        site = system.getVirtualSite(index)
        particles = tuple(
            int(site.getParticle(i)) for i in range(site.getNumParticles())
        )
        if isinstance(site, openmm.TwoParticleAverageSite):
            sites.append(_VirtualSite(
                index, "two_average", particles,
                (tuple(float(site.getWeight(i)) for i in range(2)),),
            ))
        elif isinstance(site, openmm.ThreeParticleAverageSite):
            sites.append(_VirtualSite(
                index, "three_average", particles,
                (tuple(float(site.getWeight(i)) for i in range(3)),),
            ))
        elif isinstance(site, openmm.OutOfPlaneSite):
            sites.append(_VirtualSite(
                index, "out_of_plane", particles,
                (
                    float(site.getWeight12()),
                    float(site.getWeight13()),
                    float(site.getWeightCross()),
                ),
            ))
        elif isinstance(site, openmm.LocalCoordinatesSite):
            sites.append(_VirtualSite(
                index, "local_coordinates", particles,
                (
                    tuple(float(x) for x in site.getOriginWeights()),
                    tuple(float(x) for x in site.getXWeights()),
                    tuple(float(x) for x in site.getYWeights()),
                    _vec3(site.getLocalPosition(), openmm.unit.angstrom),
                ),
            ))
        elif hasattr(openmm, "SymmetrySite") and isinstance(
                site, openmm.SymmetrySite,
        ):
            rotation = np.asarray([
                _vec3(site.getRotationMatrix()[i]) for i in range(3)
            ])
            use_box = bool(site.getUseBoxVectors())
            offset = np.asarray(_vec3(site.getOffsetVector()))
            if not use_box:
                # SymmetrySite stores Cartesian offsets in OpenMM's native nm.
                offset *= 10.0
            sites.append(_VirtualSite(
                index, "symmetry", particles,
                (
                    rotation,
                    offset,
                    use_box,
                ),
            ))
        else:
            raise TypeError(
                f"Unsupported OpenMM virtual-site type: {type(site).__name__}",
            )
    return VirtualSiteData(tuple(sites))
