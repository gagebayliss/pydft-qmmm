"""Test virtual site I/O, force distribution, and position calculation
"""
from __future__ import annotations

import pytest

import numpy as np

from pydft_qmmm import System

def test_virtual_io():
    system = System.load(
        "tests/swm4ndp_data/swm4ndp_qmmm_1024.pdb",
        "tests/swm4ndp_data/swm4ndp.xml"
    )
    print(len(system))
    assert len(system.virtual_site_indices) == (len(system) // 5)

