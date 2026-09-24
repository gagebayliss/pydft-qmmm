"""Gradient and convergence checks for fixed-force Drude relaxation."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import openmm
import pytest

from pydft_qmmm import MMHamiltonian, QMHamiltonian, QMMMHamiltonian, System
from pydft_qmmm.plugins import (
    CachedForceCalculator, CompositeSCF, DrudeCalculator, DrudeSCF,
    IntegratorDrudeSCF, QMForceCache,
    extract_drude_data,
)
from pydft_qmmm.utils import Subsystem, load_omm_system, virtual_sites

FORCEFIELD = [
    "tests/swm4ndp_data/swm4ndp.xml",
    "tests/swm4ndp_data/swm4ndp_residues.xml",
]


@pytest.fixture
def drude_dimer():
    system = System.load("tests/swm4ndp_data/hoh_dimer.pdb", *FORCEFIELD)
    system.positions[:] = virtual_sites.compute_positions(system, system.positions)
    return system


@pytest.fixture(params=[False, True], ids=["mm", "qmmm"])
def drude_problem(request, drude_dimer, tmp_path):
    system = drude_dimer
    mm = MMHamiltonian(
        forcefield=FORCEFIELD, nonbonded_method="NoCutoff", platform="Reference",
    )
    external_potential = None
    if request.param:
        psi4 = pytest.importorskip("psi4")
        psi4.set_num_threads(1)
        qm = QMHamiltonian(
            basis="sto-3g", functional="HF", charge=0, multiplicity=1,
            guess="sad", e_convergence=12, d_convergence=12,
            output_file=str(tmp_path / "psi4.log"),
        )
        coupling = QMMMHamiltonian("electrostatic", "none", partition=None)
        calculator = (mm[5:] + qm[:5] + coupling).build_calculator(system)
        system.subsystems[5:] = Subsystem.II
        mm_calculator, qm_calculator = calculator.calculators
        observer = QMForceCache()
        qm_calculator.register_plugin(observer)
        qm_calculator.calculate()
        external_potential = CachedForceCalculator(system, observer)
    else:
        calculator = mm.build_calculator(system)
        mm_calculator = calculator
    data = extract_drude_data(load_omm_system(system, FORCEFIELD))
    adapter = DrudeCalculator(mm_calculator, data, external_potential)
    return calculator, adapter


@pytest.mark.parametrize("step", [1e-4, 5e-5])
def test_fixed_force_drude_gradient(drude_problem, step):
    """Check every Drude component, including the frozen-force energy term."""
    _, adapter = drude_problem
    reference = adapter.state
    _, analytical = adapter.calculate(reference)
    numerical = np.zeros_like(reference)
    try:
        for index in range(len(reference)):
            plus = reference.copy()
            minus = reference.copy()
            plus[index] += step
            minus[index] -= step
            numerical[index] = (
                adapter.calculate(plus)[0] - adapter.calculate(minus)[0]
            ) / (2 * step)
    finally:
        adapter.state = reference
    error = np.max(np.abs(analytical - numerical))
    print(f"Fixed-force Drude gradient: h={step:g} A, max error={error:.3e}")
    np.testing.assert_allclose(analytical, numerical, atol=2e-5, rtol=0)


@pytest.mark.parametrize("algorithm", ["BFGS", "L-BFGS-B"])
def test_fixed_force_relaxation(drude_problem, algorithm):
    calculator, adapter = drude_problem
    original = calculator.system.positions.base.copy()
    initial_energy, _ = adapter.calculate(adapter.state)
    solver = DrudeSCF(adapter, algorithm=algorithm, force_tolerance=1e-5,
                      max_iterations=100)
    energy, converged = solver.solve()
    assert converged, solver.result.message
    assert energy < initial_energy
    final_energy, gradient = adapter.calculate(adapter.state)
    assert energy == pytest.approx(final_energy)
    assert np.max(np.abs(gradient)) <= solver.force_tolerance
    nuclear_indices = sorted(set(range(len(original))) - set(adapter.indices))
    np.testing.assert_array_equal(
        calculator.system.positions[nuclear_indices], original[nuclear_indices],
    )


def test_drude_metadata_units(drude_dimer):
    data = extract_drude_data(load_omm_system(drude_dimer, FORCEFIELD))
    np.testing.assert_array_equal(data.drude_indices, [4, 9])
    np.testing.assert_array_equal(data.parent_indices, [0, 5])
    np.testing.assert_allclose(data.polarizabilities, 0.978253)
    np.testing.assert_allclose(data.force_constants, 4183.874769695973, rtol=1e-12)
    assert extract_drude_data(openmm.System()) is None


def test_composite_scf_requires_converged_children():
    class Solver:
        def __init__(self, converged):
            self.converged = converged
            self.calls = 0

        def solve(self):
            self.calls += 1
            return 2.0, self.converged

    first, second = Solver(True), Solver(False)
    solver = CompositeSCF([first, second], 0, maxiter=3)
    assert solver.solve() == (2.0, False)
    assert first.calls == second.calls == 3
    second.converged = True
    assert solver.solve() == (2.0, True)
    assert first.calls == second.calls == 5


def test_solver_checks_final_forces_and_restores_accepted_point(
        drude_problem, monkeypatch,
):
    _, adapter = drude_problem
    accepted = adapter.state

    def minimize(objective, initial, **kwargs):
        objective(initial + 0.1)  # Simulate a rejected line-search trial.
        return SimpleNamespace(x=accepted, success=True)

    monkeypatch.setattr("scipy.optimize.minimize", minimize)
    solver = DrudeSCF(adapter, force_tolerance=1e-10)
    _, converged = solver.solve()
    assert not converged  # A success flag cannot replace the force check.
    np.testing.assert_array_equal(adapter.state, accepted)



@pytest.mark.parametrize("step", [1e-4, 5e-5])
def test_full_energy_gradient_before_freezing(drude_problem, step):
    """Check the physical QM/MM forces used to construct the frozen field.

    This checks a single geometry, not the derivative of the approximate
    trajectory produced by using the previous step's forces.
    """
    from pydft_qmmm.utils import numerical_gradient

    calculator, _ = drude_problem
    atoms = frozenset({0, 1, 2, 4, 5, 6, 7, 9})
    analytical = -calculator.calculate().forces[sorted(atoms)]
    numerical = numerical_gradient(calculator, atoms, dist=step)
    error = np.max(np.abs(analytical - numerical))
    print(f"Full energy gradient: h={step:g} A, max error={error:.3e}")
    np.testing.assert_allclose(analytical, numerical, atol=2e-3, rtol=0)


def test_integrator_reuses_previous_qm_forces(drude_problem, monkeypatch):
    from pydft_qmmm import VerletIntegrator
    from pydft_qmmm.calculators import PotentialCalculator
    from pydft_qmmm.plugins import Virtual

    calculator, adapter = drude_problem
    system = calculator.system
    plugin = IntegratorDrudeSCF(
        calculator, FORCEFIELD, force_tolerance=1e-5, max_iterations=100,
    )
    system.forces[:] = calculator.calculate().forces
    original = system.positions.base.copy()
    previous_qm_forces = system.forces - adapter.calculator.calculate().forces
    # A small physical step; virtual sites are rebuilt before relaxation.
    system.masses[system.virtual_site_indices] = 0.1
    integrator = VerletIntegrator(0.1)
    integrator.register_plugin(Virtual())
    integrator.register_plugin(plugin, 0)
    original_calculate = PotentialCalculator.calculate
    qm_calls = 0

    def count_qm_calls(self, *args, **kwargs):
        nonlocal qm_calls
        if self.calculator_group == "QM":
            qm_calls += 1
            raise AssertionError("Drude relaxation must not evaluate QM.")
        return original_calculate(self, *args, **kwargs)

    monkeypatch.setattr(PotentialCalculator, "calculate", count_qm_calls)
    positions, velocities = integrator.integrate(system)
    assert qm_calls == 0
    np.testing.assert_array_equal(system.positions, original)
    if plugin.qm_force_cache is not None:
        np.testing.assert_allclose(
            plugin.qm_force_cache.forces, previous_qm_forces, atol=1e-12,
        )
        external = plugin.scf.calculator.external_potential
        assert external.calculate().forces is plugin.qm_force_cache.forces
    np.testing.assert_array_equal(velocities[adapter.indices], 0)
    np.testing.assert_allclose(
        positions, virtual_sites.compute_positions(system, positions),
    )
    try:
        system.positions[:] = positions
        _, gradient = plugin.scf.calculator.calculate(
            positions[adapter.indices].reshape(-1),
        )
        assert np.max(np.abs(gradient)) <= 1e-5
    finally:
        system.positions[:] = original


@pytest.mark.parametrize("failure", ["exception", "nonconvergence"])
def test_integrator_restores_positions_on_failure(drude_problem, monkeypatch, failure):
    calculator, _ = drude_problem
    system = calculator.system
    plugin = IntegratorDrudeSCF(calculator, FORCEFIELD)
    system.forces[:] = calculator.calculate().forces
    original = system.positions.base.copy()

    def fail():
        system.positions[:] = original + 0.5
        if failure == "exception":
            raise RuntimeError("backend failure")
        return 0.0, False

    monkeypatch.setattr(plugin.scf, "solve", fail)
    integrate = plugin._modify_integrate(
        lambda system: (original + 0.1, np.zeros_like(original)),
    )
    with pytest.raises(RuntimeError):
        integrate(system)
    np.testing.assert_array_equal(system.positions, original)


def test_cache_observes_forces_without_recalculating(drude_dimer):
    from pydft_qmmm.calculators import Results

    system = drude_dimer
    observer = QMForceCache()
    observer.modify(SimpleNamespace(system=system))
    external = CachedForceCalculator(system, observer)
    with pytest.raises(RuntimeError, match="not been cached"):
        external.calculate()

    force_values = np.ones(system.positions.shape)

    def calculate(return_forces, return_components):
        return Results(123.0, force_values if return_forces else np.empty(0))

    observed_calculate = observer._modify_calculate(calculate)
    observed_calculate(True, True)
    snapshot = observer.forces
    assert snapshot is not force_values  # Own the cached data.
    force_values[:] = 2.0
    observed_calculate(False, True)
    assert observer.forces is snapshot  # Energy-only calls do not overwrite.
    assert external.calculate().forces is snapshot
    system.positions[:] += 0.1
    assert external.calculate(False).energy == pytest.approx(-3.0)
    assert external.calculate(False).forces.size == 0

    observed_calculate(True, False)
    assert observer.forces is not snapshot
    np.testing.assert_array_equal(external.calculate().forces, 2.0)
    assert external.calculate().energy == pytest.approx(0.0)


def test_general_external_calculator(drude_dimer):
    """External potentials may depend on coordinates, not just cached forces."""
    from pydft_qmmm.calculators import Results

    system = drude_dimer
    reference = system.positions.base.copy()
    data = extract_drude_data(load_omm_system(system, FORCEFIELD))

    class HarmonicCalculator:
        def __init__(self, center):
            self.system = system
            self.center = center

        def calculate(self, return_forces=True, return_components=True):
            displacement = self.system.positions - self.center
            return Results(float(np.sum(displacement**2)), -2 * displacement)

    base = HarmonicCalculator(reference)
    external = HarmonicCalculator(reference + 0.2)
    adapter = DrudeCalculator(base, data, external_potential=external)
    initial = adapter.state
    _, analytical = adapter.calculate(initial)
    numerical = np.zeros_like(initial)
    for index in range(len(initial)):
        delta = np.zeros_like(initial)
        delta[index] = 1e-5
        numerical[index] = (
            adapter.calculate(initial + delta)[0]
            - adapter.calculate(initial - delta)[0]
        ) / 2e-5
    adapter.state = initial
    np.testing.assert_allclose(analytical, numerical, atol=1e-9)
    _, converged = DrudeSCF(adapter, force_tolerance=1e-9).solve()
    assert converged
    np.testing.assert_allclose(adapter.state, initial + 0.1, atol=1e-9)
