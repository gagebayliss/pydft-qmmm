"""Test virtual site I/O, force distribution, and position calculation
"""
from __future__ import annotations

import pytest

import numpy as np

from pydft_qmmm import System
from pydft_qmmm import VerletIntegrator
from pydft_qmmm.utils import ELEMENT_TO_MASS
from pydft_qmmm.plugins import Virtual
from pydft_qmmm.plugins import Stationary
from pydft_qmmm.plugins import SETTLE

def test_virtual_io():
    system = System.load(
        "tests/swm4ndp_data/swm4ndp_qmmm_1024.pdb",
        "tests/swm4ndp_data/swm4ndp.xml"
    )
    assert (len(system) // 5) == len(system.virtual_site_indices)

def test_integrator_virtual_plugin_with_settle():
    """This test covers virtual_sites.compute_positions also!
    """
    system = System.load(
        "tests/swm4ndp_data/hoh.pdb",
        "tests/swm4ndp_data/swm4ndp.xml"
    )
    integrator = VerletIntegrator(1.0)
    stationary_atoms = [
        int(atom) for atom in np.where(system.masses.base == 0)[0]
    ]
    if stationary_atoms:
        query = "atom"
        for atom in stationary_atoms:
            query += f" {atom}"
            system.masses[atom] = ELEMENT_TO_MASS.get(
                system.elements[atom],
                0.1,
            )
        integrator.register_plugin(Stationary(query), 0)
    integrator.register_plugin(Virtual(),0)
    integrator.register_plugin(SETTLE())

    trans = np.array([1.0,1.0,1.0])
    zero_zero_zero = np.array([0.0,0.0,0.0])

    system.velocities[:] = np.array([
        trans,
        trans,
        trans + np.array([1.0,0.0,0.0]),
        zero_zero_zero,
        zero_zero_zero,
    ])
    system.positions[:] = reference_calculate_positions(np.asarray(system.positions))
    system.forces[:] = np.zeros_like(system.velocities)
    positions, velocities = integrator.integrate(system)
    system.positions[:] = positions
    positions, velocities = integrator.integrate(system)
    ref_pos = reference_calculate_positions(positions)
    np.testing.assert_allclose(positions,ref_pos)

def reference_calculate_positions(water_positions):
    pos = water_positions.copy()
    pos[3] = pos[0] * 0.589781071 + pos[1] * 0.2051094645 + pos[2] * 0.2051094645 
    pos[4] = np.array([0.0,0.0,0.0])
    return pos

def test_three_average_distribute_forces():
    from pydft_qmmm.utils import virtual_sites
    system = System.load(
        "tests/swm4ndp_data/hoh.pdb",
        "tests/swm4ndp_data/swm4ndp.xml"
    )
    arbitrary_force = np.array([1000,1.5,1.0])
    zero_zero_zero = np.array([0.0,0.0,0.0])
    forces = np.array([
        arbitrary_force,
        arbitrary_force,
        arbitrary_force,
        arbitrary_force,
        zero_zero_zero,
    ])
    propagated_forces = virtual_sites.distribute_forces(system,forces)
    reference_propagated_forces = reference_distribute_forces(system,forces)
    np.testing.assert_allclose(propagated_forces,reference_propagated_forces)

def reference_distribute_forces(system,forces):
    """pretty simple...
    """
    f1 = forces[0]
    f2 = forces[1]
    f3 = forces[2]
    f4 = forces[3]
    f5 = forces[4]
    f1 += f4 * 0.589781071 
    f2 += f4 * 0.2051094645
    f3 += f4 * 0.2051094645
    forces = np.array([f1,f2,f3,f4,f5])
    return forces