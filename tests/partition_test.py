
from __future__ import annotations

import pytest

import numpy as np

from pydft_qmmm import QMMMHamiltonian
from pydft_qmmm.utils import numerical_gradient
from pydft_qmmm.plugins import CentroidPartition

def test_centroid_partition(
        spce_qmmm_system,
        mm_spce,
        qm_water,
):
    system =spce_qmmm_system
    qmmm = QMMMHamiltonian(
        "electrostatic",
        "none",
        partition=CentroidPartition("all",12.0),
        )
    total = mm_spce[3:] + qm_water[0:3] + qmmm
    calculator = total.build_calculator(system)
    # partition is a calculator modifier, so we have to run a calculation
    # to see how the system is partitioned.
    ss_i = np.array(sorted(system.select("subsystem I")),dtype=int)
    ss_ii = np.array(sorted(system.select("subsystem II")),dtype=int) # something is broken here.
    ss_iii = np.array(sorted(system.select("subsystem III")),dtype=int)
    
    print("before calculator")
    print(f"len subsystem II: {len(ss_ii)}")
    print(f"len subsystem III: {len(ss_iii)}")
    
    analytical = calculator.calculate().forces
    
    ss_i = np.array(sorted(system.select("subsystem I")),dtype=int)
    ss_ii = np.array(sorted(system.select("subsystem II")),dtype=int) # something is broken here.
    ss_iii = np.array(sorted(system.select("subsystem III")),dtype=int)
    
    print("after")
    print(f"len subsystem II: {len(ss_ii)}")
    print(f"len subsystem III: {len(ss_iii)}")
    return
