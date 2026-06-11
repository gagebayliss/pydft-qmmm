from __future__ import annotations

import numpy as np
import openmm
import openmm.unit
import pytest

from pydft_qmmm.plugins.drude import DrudeData
from pydft_qmmm.plugins.drude import DrudeSolver
from pydft_qmmm.plugins.drude import drude_relaxation_step
from pydft_qmmm.plugins.drude import extract_drude_data


def test_extract_drude_data():
    system = openmm.System()
    parent = system.addParticle(16.0)
    drude = system.addParticle(0.4)
    force = openmm.DrudeForce()
    force.addParticle(
        drude,
        parent,
        -1,
        -1,
        -1,
        -1.0*openmm.unit.elementary_charge,
        0.001*openmm.unit.nanometer**3,
        0.0,
        0.0,
    )
    system.addForce(force)

    data = extract_drude_data(system)

    assert data.drude_indices.tolist() == [drude]
    assert data.parent_indices.tolist() == [parent]
    assert data.charges.tolist() == [-1.0]
    assert data.polarizabilities.tolist() == [0.001]
    assert data.force_constants[0] == pytest.approx(138935.45764438197)


def test_drude_solver_relaxes_against_force_oracle():
    data = DrudeData(
        drude_indices=np.array([0]),
        parent_indices=np.array([1]),
        charges=np.array([-1.0]),
        polarizabilities=np.array([1.0]),
        force_constants=np.array([10.0]),
    )

    def oracle(positions):
        x_nm = positions[0, 0] / 10.0
        force = np.zeros((1, 3))
        force[0, 0] = -10.0*(x_nm - 0.02)
        return force

    solver = DrudeSolver(
        data,
        oracle,
        force_tolerance=1e-12,
        displacement_tolerance=1e-12,
    )
    positions, info = solver.relax(np.zeros((2, 3)))

    assert positions[0, 0] == pytest.approx(0.2)
    assert info.converged


def test_drude_relaxation_step_is_standalone():
    data = DrudeData(
        drude_indices=np.array([0]),
        parent_indices=np.array([1]),
        charges=np.array([-1.0]),
        polarizabilities=np.array([1.0]),
        force_constants=np.array([20.0]),
    )
    positions = np.zeros((2, 3))
    forces = np.array([[2.0, 0.0, 0.0]])

    updated, info = drude_relaxation_step(data, positions, forces)

    assert updated[0, 0] == pytest.approx(1.0)
    assert positions[0, 0] == 0.0
    assert info.max_force == pytest.approx(2.0)
    assert info.max_displacement == pytest.approx(1.0)
