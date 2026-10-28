from __future__ import annotations

import pytest

import time

from pydft_qmmm import System
from pydft_qmmm import MMHamiltonian
from pydft_qmmm import QMHamiltonian
from pydft_qmmm import QMMMHamiltonian
from pydft_qmmm.utils import numerical_gradient


def test_pyscf_electrostatic_mechanical_gradients():
    system = System.load("tests/swm4ndp_data/swm4ndp_qmmm_1024.pdb")
    qm = QMHamiltonian(
       interface="pyscf-mol",
       device="gpu",
       basis="def2-TZVPP",
       functional="B3LYP",
       charge=0,
       multiplicity=1,
       guess="read",
       density_fit=True,
       conv_tol=1e-6,
    )
    mm = MMHamiltonian(
        forcefield=[
            "tests/swm4ndp_data/swm4ndp.xml",
            "tests/swm4ndp_data/swm4ndp_residues.xml",
        ],
        pme_gridnumber=30,
        pme_alpha=5.0,
        platform="CUDA",
    )
    qmmm = QMMMHamiltonian("electrostatic", "mechanical")
    total = mm[5:] + qm[0:5] + qmmm
    calculator = total.build_calculator(spce_qmmm_system)

    time_0 = time.perf_counter()
    
    force_indices = np.array([0,5,10])
    analytical = -calculator.calculate().forces[force_indices]

    time_1 = time.perf_counter()

    numerical = numerical_gradient(calculator, set(force_indices))

    time_2 = time.perf_counter()
    
    # write timings and errors to disk here...
    # want an output directory. want some reasonable format with relevant information
    # really, easier to specify output directory in script, right?
    # writing a json blob would be fine. just need a way to name these...
    calculator_time = time_1 - time_0
    numgrad_time = time_2 - time_1
    error = analytical - numerical
    MAE = np.average(np.abs(error))
    max_abs_error = np.abs(error).max()


    assert analytical - numerical == pytest.approx(0, abs=1)
