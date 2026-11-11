"""Test virtual site I/O, force distribution, and position calculation
"""
from __future__ import annotations

import pytest

import json
import numpy as np

from pydft_qmmm import System
from pydft_qmmm import VerletIntegrator
from pydft_qmmm import MMHamiltonian
from pydft_qmmm import QMHamiltonian
from pydft_qmmm import QMMMHamiltonian
from pydft_qmmm.utils import ELEMENT_TO_MASS
from pydft_qmmm.utils import Subsystem
from pydft_qmmm.plugins import Stationary
from pydft_qmmm.plugins import SETTLE

def test_virtual_io():
    system = System.load(
        "tests/tip4pew_data/tip4pew_trimer.pdb",
    )
    assert len(system.virtual_site_indices) == 0
    #### build calculator
    mm = MMHamiltonian(
        forcefield=[
            "tests/tip4pew_data/tip4pew.xml",
        ],
        nonbonded_method="NoCutoff",
    )
    calculator = mm.build_calculator(system)
    #### system doesn't have virtual sites until we do this
    assert len(system.virtual_site_indices) == (len(system) // 4)

def test_virtual_site_positions():
    """runs a short trajectory.
    This test has been shown to fail when compute_positions
    function is broken, so it is valid...
    """
    system = System.load(
        "tests/tip4pew_data/tip4pew_trimer.pdb",
    )
    #### build calculator
    mm = MMHamiltonian(
        forcefield=[
            "tests/tip4pew_data/tip4pew.xml",
        ],
        nonbonded_method="NoCutoff",
    )
    calculator = mm.build_calculator(system)
    integrator = VerletIntegrator(1)
    for i in range(10):
        system.forces = calculator.calculate().forces
        positions, velocities = integrator.integrate(system)
        system.positions = positions
        system.velocities = velocities
        reference_positions = reference_calculate_positions(positions)
        np.testing.assert_allclose(system.positions.base,reference_positions)
    

def test_three_average_distribute_forces():
    """tests force distribution.
    This test has been shown to fail when the force distribution 
    function is broken, so it is valid...
    """
    system = System.load(
        "tests/tip4pew_data/tip4pew_trimer.pdb",
    )
    # we have no virtual sites until we build calculator
    mm = MMHamiltonian(
        forcefield=[
            "tests/tip4pew_data/tip4pew.xml",
        ],
        nonbonded_method="NoCutoff",
    )
    calculator = mm.build_calculator(system)
    # (but calculator is not used in this test)

    rng = np.random.default_rng(seed=42)
    forces = rng.random((len(system),3))

    propagated_forces = system.vsite_helper.distribute_forces(forces)
    reference_propagated_forces = reference_distribute_forces(forces)

    np.testing.assert_allclose(propagated_forces,reference_propagated_forces)


def reference_calculate_positions(water_positions):
    pos = water_positions.copy()
    for i in range(len(water_positions) // 4):
        print(i)
        w1 = 0.786646558
        w2 = 0.106676721
        w3 = w2
        pos[i*4 + 3] = pos[i*4+0] * w1 + pos[i*4+1] * w2  + pos[i*4+2] * w3
    return pos

def reference_distribute_forces(forces):
    """pretty simple...
    """
    _forces = forces.copy()
    w1 = 0.786646558
    w2 = 0.106676721
    w3 = w2
    assert(w1+w2+w3 - 1 <= 0.000001)
    for i in range(len(_forces) // 4):
        f4 = _forces[i*4+3]
        _forces[i*4+0] += f4 * w1
        _forces[i*4+1] += f4 * w2
        _forces[i*4+2] += f4 * w3
    return _forces




##### these.....

def test_calculator():
    system = System.load(
        "tests/tip4pew_data/tip4pew_trimer.pdb",
    )
    with open("tests/tip4pew_data/trimer_ss_ii.json") as fh:
        embedding_list = json.load(fh)
    for atom in embedding_list:
        system.subsystems[atom] = Subsystem.II
    mm = MMHamiltonian(
        forcefield=[
            "tests/tip4pew_data/tip4pew.xml",
            # "tests/tip4pew_data/tip4pew_residues.xml",
        ],
        # pme_gridnumber=30,
        # pme_alpha=5.0,
        nonbonded_method="NoCutoff",
    )
    qm = QMHamiltonian(
        basis="sto-3g",
        functional="HF",
        charge=0,
        multiplicity=1,
        guess="read",
    )
    I_III = "none"
    qmmm = QMMMHamiltonian(
        "electrostatic",
        I_III,
        partition=None,
    )
    total = mm[4:] + qm[:4] + qmmm
    calculator = total.build_calculator(system)
    results = calculator.calculate()
    forces = results.forces
    print()
    np.set_printoptions(
        precision=4,       # digits after decimal
        suppress=True,     # avoid scientific notation for small values
        # linewidth=120,     # characters before wrapping
        # threshold=1000,    # number of elements before abbreviating with ...
    )
    print("tip4pew")
    print(f"I_III: {I_III}")
    print("subsystem I")
    print(forces[:4])
    print("subsystem II")
    print(forces[4:8])
    print("subsystem III")
    print(forces[8:])


def test_calculator_spce():
    system = System.load(
        "tests/spce_data/hoh_123.pdb",
    )
    mm = MMHamiltonian(
        forcefield=[
            "tests/spce_data/spce.xml",
            "tests/spce_data/spce_residues.xml",
        ],
        nonbonded_method="NoCutoff",
    )
    qm = QMHamiltonian(
        basis="def2-SVP",
        functional="PBE",
        charge=0,
        multiplicity=1,
        guess="SAD",
    )
    # I_II = "electrostatic"
    # I_II = "mechanical"
    I_II = "none"
    I_III = "none"
    # I_III = "mechanical"
    qmmm = QMMMHamiltonian(
        I_II,
        I_III,
        partition=None,
    )
    total = mm[3:] + qm[:3] + qmmm
    calculator = total.build_calculator(system)
    
    with open("tests/spce_data/trimer_ss_ii.json") as fh:
        embedding_list = json.load(fh)
    for atom in embedding_list:
        calculator.system.subsystems[atom] = Subsystem.II
        
    results = calculator.calculate()
    forces = results.forces
    
    np.set_printoptions(
        precision=4,       # digits after decimal
        suppress=True,     # avoid scientific notation for small values
    )
    print()
    print("spce, QM/MM")
    print(f"I_II: {I_II}")
    print(f"I_III: {I_III}")
    print("subsystem I")
    print(forces[:3])
    print("subsystem II")
    print(forces[3:6])
    print("subsystem III")
    print(forces[6:])


def test_calculator_spce_mm_only():
    sys_name = "hoh_123"
    system = System.load(
        f"tests/spce_data/{sys_name}.pdb",
    )
    mm = MMHamiltonian(
        forcefield=[
            "tests/spce_data/spce.xml",
            "tests/spce_data/spce_residues.xml",
        ],
        nonbonded_method="NoCutoff",
    )
    total = mm
    calculator = total.build_calculator(system)
    results = calculator.calculate()
    forces = results.forces
    np.set_printoptions(
        precision=4,       # digits after decimal
        suppress=True,     # avoid scientific notation for small values
    )
    print()
    print("spce, MM only")
    print("subsystem I")
    print(forces[:3])
    print("subsystem II")
    print(forces[3:6])
    print("subsystem III")
    print(forces[6:])
    np.save(f"tests/spce_data/reference/{sys_name}_forces.npy",forces)

