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

def test_calculator():
    system = System.load(
        "tests/tip4pew_data/tip4pew_trimer.pdb",
        "tests/tip4pew_data/tip4pew.xml"
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
        "tests/spce_data/hoh_trimer.pdb",
        "tests/spce_data/spce.xml"
    )
    mm = MMHamiltonian(
        forcefield=[
            "tests/spce_data/spce.xml",
            "tests/spce_data/spce_residues.xml",
        ],
        # pme_gridnumber=30,
        # pme_alpha=5.0,
        nonbonded_method="NoCutoff",
    )
    qm = QMHamiltonian(
        basis="def2-SVP",
        functional="PBE",
        charge=0,
        multiplicity=1,
        guess="SAD",
    )
    I_II = "electrostatic"
    # I_II = "mechanical"
    # I_II = "none"
    # I_III = "none"
    I_III = "mechanical"
    qmmm = QMMMHamiltonian(
        I_II,
        I_III,
        partition=None,
    )
    total = mm[3:] + qm[:3] + qmmm
    calculator = total.build_calculator(system) # takes system by reference?
    with open("tests/spce_data/trimer_ss_ii.json") as fh:
        embedding_list = json.load(fh)
    for atom in embedding_list:
        system.subsystems[atom] = Subsystem.II
    results = calculator.calculate()
    forces = results.forces
    print()
    np.set_printoptions(
        precision=4,       # digits after decimal
        suppress=True,     # avoid scientific notation for small values
        # linewidth=120,     # characters before wrapping
        # threshold=1000,    # number of elements before abbreviating with ...
    )
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
    # return
    system = System.load(
        "tests/spce_data/hoh_trimer_dimer.pdb",
        "tests/spce_data/spce.xml"
    )
    # with open("tests/tip4pew_data/trimer_ss_ii.json") as fh:
    #     embedding_list = json.load(fh)
    # for atom in embedding_list:
    #     system.subsystems[atom] = Subsystem.II
    mm = MMHamiltonian(
        forcefield=[
            "tests/spce_data/spce.xml",
            "tests/spce_data/spce_residues.xml",
        ],
        # pme_gridnumber=30,
        # pme_alpha=5.0,
        nonbonded_method="NoCutoff",
    )
    total = mm
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
    print("spce, MM only")
    print("subsystem I")
    print(forces[:3])
    print("subsystem II")
    print(forces[3:6])
    print("subsystem III")
    print(forces[6:])