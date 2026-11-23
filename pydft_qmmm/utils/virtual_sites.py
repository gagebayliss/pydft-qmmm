"""OpenMM-compatible virtual-site position construction."""
# The out-of-plane and local-coordinate formulas are translated from
# OpenMM's ReferenceVirtualSites.cpp.
# This is part of the OpenMM molecular simulation toolkit.
# See https://openmm.org/development.
#
# Portions copyright (c) 2012-2025 Stanford University and the Authors.
# Authors: Peter Eastman
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

__all__ = [
    "VirtualSite",
    "extract_virtual_sites",
    "VirtualSiteHelper",
]

from dataclasses import dataclass
from typing import Any
from typing import TYPE_CHECKING
from pydft_qmmm.utils import system_cache

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
        parameters: Parameters for the site. Out-of-plane sites store
            (w12, w13, wcross), with wcross in inverse angstroms. Local
            coordinate sites store (origin_weights, x_weights, y_weights,
            local_position), with local_position in angstroms. Drude sites store
            (charge, polarizability, spring_constant) in elementary charge,
            angstrom^3, and kJ/mol/angstrom^2, with one oscillator parent.
    """

    index: int
    site_type: str
    parents: tuple[int, ...]
    parameters: tuple[Any, ...]


def _extract_drude_sites(system: openmm.System) -> tuple[VirtualSite, ...]:
    """Read oscillator sites from the force that owns their OpenMM coordinates."""
    sites = []
    # OpenMM Coulomb constant in kJ angstrom / (mol e^2).
    coulomb_constant = 138.93545764438198 * 10
    drude_forces = [force for force in system.getForces()
                    if isinstance(force, openmm.DrudeForce)]
    if len(drude_forces) > 1:
        raise ValueError("Expected one DrudeForce in the OpenMM system.")
    if drude_forces:
        force = drude_forces[0]
        for i in range(force.getNumParticles()):
            index, parent, _, _, _, charge, alpha, _, _ = force.getParticleParameters(i)
            charge = charge.value_in_unit(openmm.unit.elementary_charge)
            alpha = alpha.value_in_unit(openmm.unit.angstrom**3)
            sites.append(VirtualSite(
                index, "drude", (parent,),
                (charge, alpha, coulomb_constant * charge**2 / alpha),
            ))
    return tuple(sites)


def extract_virtual_sites(system: openmm.System) -> tuple[VirtualSite, ...]:
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
        elif isinstance(site, openmm.OutOfPlaneSite):
            sites.append(VirtualSite(
                index, "out_of_plane", particles,
                (site.getWeight12(), site.getWeight13(),
                 site.getWeightCross() / 10.0),
            ))
        elif isinstance(site, openmm.LocalCoordinatesSite):
            sites.append(VirtualSite(
                index, "local_coordinates", particles,
                (tuple(site.getOriginWeights()), tuple(site.getXWeights()),
                 tuple(site.getYWeights()), tuple(
                     site.getLocalPosition().value_in_unit(openmm.unit.angstrom),
                 )),
            ))
        else:
            raise TypeError(
                f"Unsupported OpenMM virtual-site type: {type(site).__name__}",
            )
    sites.extend(_extract_drude_sites(system))
    return tuple(sorted(sites, key=lambda site: site.index))


class VirtualSiteHelper:
    """Manages updating virtual site positions and propagating forces.
    """
    def __init__(
            self,
            system: System,
    ) -> None:
        self.system = system

    @system_cache(
        "virtual_site_indices", "virtual_types", "virtual_parents",
        "virtual_parameters",
    )
    def _get_batches(self):
        indices = list(self.system.virtual_site_indices)
        types = list(self.system.virtual_types)
        parents = list(self.system.virtual_parents)
        parameters = list(self.system.virtual_parameters)

        if not (len(indices) == len(types) == len(parents) == len(parameters)):
            raise ValueError("Inconsistent virtual-site metadata lengths")

        geometric_indices = {
            index for index, kind in zip(indices, types) if kind != "drude"
        }
        groups = {"two_average": [], "three_average": []}

        nonlinear_sites = []
        for i, site_type in enumerate(types):
            if site_type == "drude":
                if len(parents[i]) != 1 or len(parameters[i]) != 3:
                    raise ValueError(f"Invalid drude site at index {indices[i]}")
                # SCF varies this coordinate and needs its unmodified force.
                continue
            if site_type not in (*groups, "out_of_plane", "local_coordinates"):
                raise NotImplementedError(f"Unsupported virtual-site type: {site_type}")
            if any(int(parent) in geometric_indices for parent in parents[i]):
                raise NotImplementedError("Virtual sites depending on other virtual sites")
            if site_type == "local_coordinates":
                if (
                    len(parameters[i]) != 4
                    or any(len(w) != len(parents[i]) for w in parameters[i][:3])
                    or len(parameters[i][3]) != 3
                ):
                    raise ValueError(f"Invalid {site_type} site at index {indices[i]}")
                nonlinear_sites.append(VirtualSite(
                    indices[i], site_type, tuple(parents[i]),
                    (*parameters[i][:3], xp.asarray(parameters[i][3], dtype=xp.float64)),
                ))
                continue
            expected = 2 if site_type == "two_average" else 3
            if len(parents[i]) != expected or len(parameters[i]) != expected:
                raise ValueError(f"Invalid {site_type} site at index {indices[i]}")
            if site_type == "out_of_plane":
                nonlinear_sites.append(VirtualSite(
                    indices[i], site_type, tuple(parents[i]), tuple(parameters[i]),
                ))
            else:
                groups[site_type].append(i)

        batches = []
        for rows in groups.values():
            if not rows:
                continue
            site_indices = xp.asarray([indices[i] for i in rows], dtype=xp.int64)
            parent_indices = xp.stack(
                [xp.asarray(parents[i], dtype=xp.int64) for i in rows]
            )
            batch_parameters = xp.stack(
                [xp.asarray(parameters[i], dtype=xp.float64) for i in rows]
            )
            batches.append((site_indices, parent_indices, batch_parameters))
        return batches, nonlinear_sites

    def compute_positions(self, positions: NDArray[float64]) -> NDArray[float64]:
        """Computes virtual-site coordinates.

        Args:
            positions: Array containing positions to be updated.

        Returns:
            Array with updated virtual site positions.
        """
        new_positions = xp.asarray(positions).copy()
        if len(new_positions) != len(self.system):
            raise ValueError("Position array length does not match system")

        batches, nonlinear_sites = self._get_batches()
        for indices, parents, parameters in batches:
            # (n_sites, n_parents, 3) -> (n_sites, 3)
            new_positions[indices] = xp.sum(
                new_positions[parents] * parameters[:, :, None], axis=1
            )
        for site in nonlinear_sites:
            if site.site_type == "out_of_plane":
                p1, p2, p3 = site.parents
                w12, w13, wcross = site.parameters
                v12 = new_positions[p2] - new_positions[p1]
                v13 = new_positions[p3] - new_positions[p1]
                cross = xp.cross(v12, v13)
                new_positions[site.index] = (
                    new_positions[p1] + v12*w12 + v13*w13 + cross*wcross
                )
            else:
                originWeights, xWeights, yWeights, localPosition = site.parameters
                origin = xp.zeros(3)
                xdir = xp.zeros(3)
                ydir = xp.zeros(3)
                for j, p in enumerate(site.parents):
                    pos = new_positions[p]
                    origin += pos*originWeights[j]
                    xdir += pos*xWeights[j]
                    ydir += pos*yWeights[j]
                zdir = xp.cross(xdir, ydir)
                normXdir = xp.sqrt(xdir.dot(xdir))
                normZdir = xp.sqrt(zdir.dot(zdir))
                if normXdir > 0.0:
                    xdir *= 1.0/normXdir
                if normZdir > 0.0:
                    zdir *= 1.0/normZdir
                ydir = xp.cross(zdir, xdir)
                new_positions[site.index] = (
                    origin + xdir*localPosition[0] + ydir*localPosition[1]
                    + zdir*localPosition[2]
                )
        if USE_CUPY:
            new_positions = xp.asnumpy(new_positions)
        return new_positions

    def distribute_forces(
            self,
            forces: NDArray[float64],
            positions: NDArray[float64] | None = None,
    ) -> NDArray[float64]:
        """Propagate virtual-site forces to parents.
        Forces remain on virtual sites, so this should not be called multiple
        times on the same array or on forces that have already been distributed.
        (note: OpenMM distributes virtual site forces).

        Args:
            forces: Array containing forces to be distributed.
            positions: Coordinates for evaluating position-dependent
                derivatives. Defaults to the current system positions.

        Returns:
            Array with propagated virtual site forces..
        """
        new_forces = xp.asarray(forces).copy()
        if len(new_forces) != len(self.system):
            raise ValueError("Force array length does not match system")

        batches, nonlinear_sites = self._get_batches()
        for indices, parents, parameters in batches:
            # Flatten to permit duplicate parent indices, which ordinary indexed
            # '+=' does NOT accumulate correctly for NumPy or CuPy.
            contributions = (
                new_forces[indices, None, :] * parameters[:, :, None]
            ).reshape(-1, 3)
            xp.add.at(new_forces, parents.ravel(), contributions)
        if nonlinear_sites:
            if positions is None:
                positions = self.system.positions.base
            atomCoordinates = xp.asarray(positions)
            if len(atomCoordinates) != len(self.system):
                raise ValueError("Position array length does not match system")
        for site in reversed(nonlinear_sites):
            f = new_forces[site.index]
            if site.site_type == "out_of_plane":
                p1, p2, p3 = site.parents
                w12, w13, wcross = site.parameters
                v12 = atomCoordinates[p2] - atomCoordinates[p1]
                v13 = atomCoordinates[p3] - atomCoordinates[p1]
                f2 = xp.stack((
                    w12*f[0] - wcross*v13[2]*f[1] + wcross*v13[1]*f[2],
                    wcross*v13[2]*f[0] + w12*f[1] - wcross*v13[0]*f[2],
                    -wcross*v13[1]*f[0] + wcross*v13[0]*f[1] + w12*f[2],
                ))
                f3 = xp.stack((
                    w13*f[0] + wcross*v12[2]*f[1] - wcross*v12[1]*f[2],
                    -wcross*v12[2]*f[0] + w13*f[1] + wcross*v12[0]*f[2],
                    wcross*v12[1]*f[0] - wcross*v12[0]*f[1] + w13*f[2],
                ))
                new_forces[p1] += f-f2-f3
                new_forces[p2] += f2
                new_forces[p3] += f3
            else:
                originWeights, wx, wy, localPosition = site.parameters
                xdir = xp.zeros(3)
                ydir = xp.zeros(3)
                for j, p in enumerate(site.parents):
                    pos = atomCoordinates[p]
                    xdir += pos*wx[j]
                    ydir += pos*wy[j]
                zdir = xp.cross(xdir, ydir)
                normXdir = xp.sqrt(xdir.dot(xdir))
                normZdir = xp.sqrt(zdir.dot(zdir))
                invNormXdir = 1.0/normXdir if normXdir > 0.0 else 0.0
                invNormZdir = 1.0/normZdir if normZdir > 0.0 else 0.0
                dx = xdir*invNormXdir
                dz = zdir*invNormZdir
                dy = xp.cross(dz, dx)

                # OpenMM derived these expressions with SymPy and simplified them by hand.
                wxScaled = [w*invNormXdir for w in wx]
                fp1 = localPosition*f[0]
                fp2 = localPosition*f[1]
                fp3 = localPosition*f[2]
                for j, p in enumerate(site.parents):
                    t1 = (wx[j]*ydir[0]-wy[j]*xdir[0])*invNormZdir
                    t2 = (wx[j]*ydir[1]-wy[j]*xdir[1])*invNormZdir
                    t3 = (wx[j]*ydir[2]-wy[j]*xdir[2])*invNormZdir
                    sx = t3*dz[1]-t2*dz[2]
                    sy = t1*dz[2]-t3*dz[0]
                    sz = t2*dz[0]-t1*dz[1]
                    new_forces[p, 0] += (
                        fp1[0]*wxScaled[j]*(1-dx[0]*dx[0])
                        + fp1[2]*(dz[0]*sx)
                        + fp1[1]*((-dx[0]*dy[0])*wxScaled[j] + dy[0]*sx - dx[1]*t2 - dx[2]*t3)
                        + f[0]*originWeights[j]
                    )
                    new_forces[p, 1] += (
                        fp1[0]*wxScaled[j]*(-dx[0]*dx[1])
                        + fp1[2]*(dz[0]*sy+t3)
                        + fp1[1]*((-dx[1]*dy[0]-dz[2])*wxScaled[j] + dy[0]*sy + dx[1]*t1)
                    )
                    new_forces[p, 2] += (
                        fp1[0]*wxScaled[j]*(-dx[0]*dx[2])
                        + fp1[2]*(dz[0]*sz-t2)
                        + fp1[1]*((-dx[2]*dy[0]+dz[1])*wxScaled[j] + dy[0]*sz + dx[2]*t1)
                    )
                    new_forces[p, 0] += (
                        fp2[0]*wxScaled[j]*(-dx[1]*dx[0])
                        + fp2[2]*(dz[1]*sx-t3)
                        - fp2[1]*((dx[0]*dy[1]-dz[2])*wxScaled[j] - dy[1]*sx - dx[0]*t2)
                    )
                    new_forces[p, 1] += (
                        fp2[0]*wxScaled[j]*(1-dx[1]*dx[1])
                        + fp2[2]*(dz[1]*sy)
                        - fp2[1]*((dx[1]*dy[1])*wxScaled[j] - dy[1]*sy + dx[0]*t1 + dx[2]*t3)
                        + f[1]*originWeights[j]
                    )
                    new_forces[p, 2] += (
                        fp2[0]*wxScaled[j]*(-dx[1]*dx[2])
                        + fp2[2]*(dz[1]*sz+t1)
                        - fp2[1]*((dx[2]*dy[1]+dz[0])*wxScaled[j] - dy[1]*sz - dx[2]*t2)
                    )
                    new_forces[p, 0] += (
                        fp3[0]*wxScaled[j]*(-dx[2]*dx[0])
                        + fp3[2]*(dz[2]*sx+t2)
                        + fp3[1]*((-dx[0]*dy[2]-dz[1])*wxScaled[j] + dy[2]*sx + dx[0]*t3)
                    )
                    new_forces[p, 1] += (
                        fp3[0]*wxScaled[j]*(-dx[2]*dx[1])
                        + fp3[2]*(dz[2]*sy-t1)
                        + fp3[1]*((-dx[1]*dy[2]+dz[0])*wxScaled[j] + dy[2]*sy + dx[1]*t3)
                    )
                    new_forces[p, 2] += (
                        fp3[0]*wxScaled[j]*(1-dx[2]*dx[2])
                        + fp3[2]*(dz[2]*sz)
                        + fp3[1]*((-dx[2]*dy[2])*wxScaled[j] + dy[2]*sz - dx[0]*t1 - dx[1]*t2)
                        + f[2]*originWeights[j]
                    )
        if USE_CUPY:
            new_forces = xp.asnumpy(new_forces)
        return new_forces

