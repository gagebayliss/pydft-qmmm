from __future__ import annotations

import atexit
import json
import os
from tempfile import TemporaryDirectory

import psi4
import pytest

from pydft_qmmm import MMHamiltonian
from pydft_qmmm import QMHamiltonian
from pydft_qmmm import System
from pydft_qmmm import VerletIntegrator
from pydft_qmmm.plugins import SETTLE
from pydft_qmmm.utils import Subsystem

def _finalize_psi4():
    # Psi4 appends timer.dat in the current directory during interpreter exit.
    previous = os.getcwd()
    with TemporaryDirectory() as directory:
        os.chdir(directory)
        try:
            psi4.core.finalize()
        finally:
            os.chdir(previous)


atexit.unregister(psi4.core.finalize)
atexit.register(_finalize_psi4)


@pytest.fixture
def swm4ndp_system():
    return System.load(
        "tests/swm4ndp_data/swm4ndp_qmmm_1024.pdb",
    )

@pytest.fixture
def swm4ndp_qmmm_system(swm4ndp_system):
    with open("tests/swm4ndp_data/swm4ndp_qmmm_region_ii.json") as fh:
        embedding_list = json.load(fh)
    for atom in embedding_list:
        swm4ndp_system.subsystems[atom] = Subsystem.II
    return swm4ndp_system

### FINITE
@pytest.fixture
def swm4ndp_finite_system():
    return System.load(
        "tests/swm4ndp_data/swm4ndp_qmmm_231.pdb",
    )

@pytest.fixture
def swm4ndp_finite_qmmm_system(swm4ndp_finite_system):
    with open("tests/swm4ndp_data/swm4ndp_finite_qmmm_region_ii.json") as fh:
        embedding_list = json.load(fh)
    for atom in embedding_list:
        swm4ndp_finite_system.subsystems[atom] = Subsystem.II
    return swm4ndp_finite_system


@pytest.fixture
def mm_swm4ndp():
    return MMHamiltonian(
        forcefield=[
            "tests/swm4ndp_data/swm4ndp.xml",
            "tests/swm4ndp_data/swm4ndp_residues.xml",
        ],
        pme_gridnumber=30,
        pme_alpha=5.0,
    )


@pytest.fixture
def spce_system():
    return System.load(
        "tests/spce_data/spce_qmmm.pdb",
    )


# @pytest.fixture
# def spce_dimer_system():
#    return System.load(
#        "tests/spce_data/hoh_dimer.pdb",
#    )


@pytest.fixture
def spce_qmmm_system(spce_system):
    with open("tests/spce_data/spce_qmmm_region_ii.json") as fh:
        embedding_list = json.load(fh)
    for atom in embedding_list:
        spce_system.subsystems[atom] = Subsystem.II
    return spce_system


@pytest.fixture
def qm_water():
    return QMHamiltonian(
        basis="def2-SVP",
        functional="PBE",
        charge=0,
        multiplicity=1,
        guess="read",
    )


@pytest.fixture
def mm_spce():
    return MMHamiltonian(
        forcefield=[
            "tests/spce_data/spce.xml",
            "tests/spce_data/spce_residues.xml",
        ],
        pme_gridnumber=30,
        pme_alpha=5.0,
    )


@pytest.fixture
def mm_spce_no_lj():
    return MMHamiltonian(
        forcefield=[
            "tests/spce_data/spce_no_lj.xml",
            "tests/spce_data/spce_residues.xml",
        ],
        pme_gridnumber=30,
        pme_alpha=5.0,
    )


@pytest.fixture
def spce_plugins():
    return [SETTLE()]


@pytest.fixture
def verlet():
    return VerletIntegrator(1)


@pytest.fixture
def no_logging():
    return {
        "log_write": False,
        "csv_write": False,
        "dcd_write": False,
    }
