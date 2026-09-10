"""Integration tests; run with the pydft environment's pytest."""
from pathlib import Path

import numpy as np
import pytest

from gradient_check import gradient_sweep, load_backends


@pytest.fixture(scope="module")
def backends():
    return load_backends(Path(__file__).resolve().parent)


def test_post_scf_agreement(backends):
    a, b = [backend.solve(backend.initial) for backend in backends[:2]]
    indices = backends[0].drudes
    assert abs(a[0] - b[0]) < 1e-8
    np.testing.assert_allclose(a[2][indices], b[2][indices], atol=1e-9, rtol=0)
    np.testing.assert_allclose(a[1][indices], b[1][indices], atol=1e-7, rtol=0)
    for _, forces, _ in (a, b):
        assert np.max(np.abs(forces[indices])) < 1e-6


@pytest.mark.parametrize("index", [0, 1, 2], ids=["pydft", "openmm", "reference"])
def test_energy_gradient(backends, index):
    # CPU nonbonded arithmetic limits finite differences; Reference allows
    # much smaller displacements. All 27 real and 9 Drude components are tested.
    step = 3e-5 if index == 2 else 1e-2
    limits = [3e-6, 1e-7] if index == 2 else [0.3, 0.04]
    report = gradient_sweep(backends[index], [step])
    for row, limit in zip(report["gradients"], limits):
        assert row["max_error_kjmol_A"] < limit, row
