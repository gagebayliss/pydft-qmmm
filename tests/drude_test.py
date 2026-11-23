"""Gradient and convergence checks for frozen-density Drude relaxation."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import openmm
import pytest

from pydft_qmmm import MMHamiltonian, QMHamiltonian, QMMMHamiltonian, System
from pydft_qmmm.plugins import (
    CachedForceCalculator, CalculatorDrudeSCF, CompositeSCF, DrudeCalculator, DrudeSCF,
    QMForceCache,
)
from pydft_qmmm.system.atom import Atom
from pydft_qmmm.utils import KJMOL_PER_EH, Subsystem, VirtualSite, extract_virtual_sites

FORCEFIELD = [
    "tests/swm4ndp_data/swm4ndp.xml",
    "tests/swm4ndp_data/swm4ndp_residues.xml",
]


@pytest.fixture
def drude_dimer():
    return System.load("tests/swm4ndp_data/hoh_dimer.pdb")


@pytest.fixture(params=[False, True], ids=["mm", "qmmm"])
def drude_problem(request, drude_dimer, tmp_path):
    system = drude_dimer
    mm = MMHamiltonian(
        forcefield=FORCEFIELD, nonbonded_method="NoCutoff", platform="Reference",
    )
    qm_interface = None
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
        qm_interface = qm_calculator.potential
    else:
        calculator = mm.build_calculator(system)
        mm_calculator = calculator
    adapter = DrudeCalculator(mm_calculator, qm_interface=qm_interface)
    adapter.freeze_density()
    return calculator, adapter


@pytest.mark.parametrize("step", [1e-4, 5e-5])
def test_frozen_density_drude_gradient(drude_problem, step):
    """Check every Drude component, including the frozen-density embedding energy."""
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
    print(f"Frozen-density Drude gradient: h={step:g} A, max error={error:.3e}")
    np.testing.assert_allclose(analytical, numerical, atol=2e-5, rtol=0)


@pytest.mark.parametrize("algorithm", ["BFGS", "L-BFGS-B"])
def test_frozen_density_relaxation(drude_problem, algorithm):
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
    MMHamiltonian(
        forcefield=FORCEFIELD, nonbonded_method="NoCutoff", platform="Reference",
    ).build_calculator(drude_dimer)
    mask = drude_dimer.virtual_types.base == "drude"
    np.testing.assert_array_equal(drude_dimer.virtual_site_indices.base[mask], [4, 9])
    assert list(drude_dimer.virtual_parents.base[mask]) == [(0,), (5,)]
    for _, alpha, spring in drude_dimer.virtual_parameters.base[mask]:
        assert alpha == pytest.approx(0.978253)
        assert spring == pytest.approx(4183.874769695973, rel=1e-12)
    assert extract_virtual_sites(openmm.System()) == ()


def test_drude_relaxation_with_coordinate_wrapping():
    """Trial displacements remain valid when calculation shifts the system."""
    from pydft_qmmm.calculators import Results

    system = System([
        Atom(position=np.array([10., 20., 30.])),
        Atom(position=np.array([10.1, 20.1, 30.1])),
    ])
    system.build_virtual_sites([VirtualSite(1, "drude", (0,), (-1., 1., 1389.35))])

    class WrappedCalculator:
        def __init__(self):
            self.system = system

        def calculate(self, return_components=False):
            system.positions[:] -= 10.0
            displacement = system.positions[1] - system.positions[0]
            forces = np.zeros((2, 3))
            forces[1] = -2 * displacement
            return Results(float(np.sum(displacement**2)), forces)

    adapter = DrudeCalculator(WrappedCalculator())
    np.testing.assert_allclose(adapter.state, 0.1)
    energy, converged = DrudeSCF(adapter, force_tolerance=1e-9).solve()
    assert converged
    assert energy == pytest.approx(0.0, abs=1e-18)
    np.testing.assert_allclose(adapter.state, 0.0, atol=1e-9)


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
    np.testing.assert_allclose(adapter.state, accepted, atol=1e-14, rtol=0)



@pytest.mark.parametrize("step", [1e-4, 5e-5])
def test_full_energy_gradient_before_freezing(drude_problem, step):
    """Check the full QM/MM gradient with the density relaxed at each geometry."""
    from pydft_qmmm.utils import numerical_gradient

    calculator, _ = drude_problem
    atoms = frozenset({0, 1, 2, 4, 5, 6, 7, 9})
    analytical = -calculator.calculate().forces[sorted(atoms)]
    numerical = numerical_gradient(calculator, atoms, dist=step)
    error = np.max(np.abs(analytical - numerical))
    print(f"Full energy gradient: h={step:g} A, max error={error:.3e}")
    np.testing.assert_allclose(analytical, numerical, atol=2e-3, rtol=0)


def test_embedding_at_fixed_density(drude_problem, monkeypatch):
    calculator, adapter = drude_problem
    qm = adapter.qm_interface
    if qm is None:
        pytest.skip("Requires QM embedding.")

    system = calculator.system
    drude = adapter.indices[-1]
    density = adapter.density_matrix
    qm.return_external_ixn_component()
    embedding = qm.compute_components()["External Interaction Energy"]
    wfn = qm._generate_wavefunction()
    external = wfn.external_pot()
    expected = density.vector_dot(external.computePotentialMatrix(wfn.basisset()))
    expected += external.computeNuclearEnergy(wfn.molecule())
    assert embedding == pytest.approx(expected * KJMOL_PER_EH)
    assert qm.compute_embedding_energy(density) == pytest.approx(embedding)
    field = qm.compute_electric_field(system.positions[[drude]])
    np.testing.assert_allclose(field[0] * system.charges[drude],
                               qm.compute_forces()[drude], atol=1e-6)

    def unexpected_wavefunction(*args, **kwargs):
        pytest.fail("A fixed-density evaluation requested a wavefunction.")

    monkeypatch.setattr(type(qm), "_generate_wavefunction", unexpected_wavefunction)
    system.positions[drude, 0] += 0.02
    assert qm.compute_embedding_energy(density) != pytest.approx(embedding)
    new_field = qm.compute_electric_field(system.positions[[drude]], density)
    assert not np.allclose(new_field, field)
    # Drude trials must skip the arbitrary potential even when one is installed.
    qm.potentials.append(object())
    monkeypatch.setattr(type(qm), "_generate_external_potentials", unexpected_wavefunction)
    adapter.calculate(adapter.state)


@pytest.mark.parametrize("drude_problem", [True], indirect=True)
def test_embedding_components(drude_problem, monkeypatch):
    import psi4

    _, adapter = drude_problem
    qm = adapter.qm_interface
    wfn = qm._generate_wavefunction()
    density = adapter.density_matrix
    external = wfn.external_pot()
    point_energy = qm.compute_embedding_energy(density)
    matrix = psi4.core.Matrix.from_array(np.eye(wfn.basisset().nbf()))
    external.setMatrix(matrix)
    matrix_energy = density.vector_dot(matrix) * KJMOL_PER_EH
    gaussian_molecule = psi4.geometry("He 4 0 0\nno_reorient\nno_com")
    gaussian_basis = psi4.core.BasisSet.build(gaussian_molecule, "BASIS", "sto-3g")
    external.addGaussian(gaussian_basis, psi4.core.Vector.from_array(np.array([0.2])))

    full = density.vector_dot(external.computePotentialMatrix(wfn.basisset()))
    full += external.computeNuclearEnergy(wfn.molecule())
    full *= KJMOL_PER_EH
    assert qm.compute_embedding_energy() == pytest.approx(full)
    assert qm.compute_embedding_energy(diffuse_charges=False) == pytest.approx(
        point_energy + matrix_energy,
    )
    assert qm.compute_embedding_energy(point_charges=False) == pytest.approx(full - point_energy)
    assert qm.compute_embedding_energy(arbitrary_potential=False) == pytest.approx(full - matrix_energy)

    components = qm.compute_components()
    electronic = qm.compute_energy() - components["Nuclear Energy"]
    assert components["Electronic Energy"] == pytest.approx(electronic)
    qm.return_external_ixn_component()
    components = qm.compute_components()
    assert components["External Interaction Energy"] == pytest.approx(full)
    assert components["Electronic Energy"] == pytest.approx(electronic - full)

    qm.potentials.append(object())
    monkeypatch.setattr(
        type(qm), "_generate_external_potentials", lambda self: [None, None, matrix.np],
    )
    assert qm.compute_embedding_energy(density, point_charges=False) == pytest.approx(matrix_energy)
    assert qm.compute_embedding_energy(density, arbitrary_potential=False) == pytest.approx(point_energy)


def test_calculator_relaxes_drudes(drude_problem, monkeypatch):
    calculator, adapter = drude_problem
    if adapter.qm_interface is not None:
        import psi4
        from scipy.optimize import minimize

        def unexpected_qm(*args, **kwargs):
            pytest.fail("The inner minimizer invoked QM SCF or gradients.")

        def minimize_with_frozen_density(*args, **kwargs):
            with monkeypatch.context() as patch:
                patch.setattr(psi4, "energy", unexpected_qm)
                patch.setattr(psi4, "gradient", unexpected_qm)
                return minimize(*args, **kwargs)

        monkeypatch.setattr("scipy.optimize.minimize", minimize_with_frozen_density)
    expected_charges = calculator.system.charges.base.copy()
    if adapter.qm_interface is not None:
        # Only the first water is in subsystem I.
        assert expected_charges[4] != 0.0
        expected_charges[0] += expected_charges[4]
        expected_charges[4] = 0.0
    plugin = CalculatorDrudeSCF(strict=True)
    calculator.register_plugin(plugin)
    np.testing.assert_array_equal(calculator.system.charges, expected_charges)
    results = calculator.calculate()
    assert np.isfinite(results.energy)
    residual = results.forces[plugin.drude_scf.calculator.indices]
    assert np.max(np.abs(residual)) <= plugin.force_tolerance


@pytest.mark.parametrize("failure", ["exception", "nonconvergence"])
def test_calculator_retries_after_failure(drude_problem, monkeypatch, failure):
    calculator, _ = drude_problem
    plugin = CalculatorDrudeSCF(strict=True)
    calculator.register_plugin(plugin)
    attempts = 0

    def solve():
        nonlocal attempts
        attempts += 1
        if attempts == 1 and failure == "exception":
            raise RuntimeError("backend failure")
        return 0.0, attempts > 1

    monkeypatch.setattr(plugin.scf, "solve", solve)
    with pytest.raises(RuntimeError):
        calculator.calculate()
    calculator.calculate()
    assert attempts == 2


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
    MMHamiltonian(
        forcefield=FORCEFIELD, nonbonded_method="NoCutoff", platform="Reference",
    ).build_calculator(system)
    reference = system.positions.base.copy()

    class HarmonicCalculator:
        def __init__(self, center):
            self.system = system
            self.center = center

        def calculate(self, return_forces=True, return_components=True):
            displacement = self.system.positions - self.center
            return Results(float(np.sum(displacement**2)), -2 * displacement)

    base = HarmonicCalculator(reference)
    external = HarmonicCalculator(reference + 0.2)
    adapter = DrudeCalculator(base, external_potential=external)
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


def test_scf_plugin_mass_transfer_and_warm_start(drude_dimer, monkeypatch):
    system = drude_dimer
    calculator = MMHamiltonian(
        forcefield=FORCEFIELD, nonbonded_method="NoCutoff", platform="Reference",
    ).build_calculator(system)
    indices = [4, 9]
    parents = [0, 5]
    np.testing.assert_array_equal(system.virtual_site_indices, [3, 4, 8, 9])
    np.testing.assert_array_equal(
        system.virtual_types, ["three_average", "drude", "three_average", "drude"],
    )
    for row, parent in zip([1, 3], parents):
        assert system.virtual_parents[row] == (parent,)
        charge, alpha, spring = system.virtual_parameters[row]
        assert charge == pytest.approx(system.charges[system.virtual_site_indices[row]])
        assert alpha == pytest.approx(0.978253)
        assert spring == pytest.approx(4183.874769695973)
    copied = system.copy()
    np.testing.assert_array_equal(copied.virtual_types, system.virtual_types)
    np.testing.assert_array_equal(copied.virtual_parameters, system.virtual_parameters)
    initial_masses = system.masses.base.copy()
    plugin = CalculatorDrudeSCF(strict=True)
    calculator.register_plugin(plugin)
    np.testing.assert_allclose(system.masses[indices], 0.0)
    np.testing.assert_allclose(
        system.masses[parents],
        initial_masses[parents] + initial_masses[indices],
    )
    assert sum(system.masses) == pytest.approx(sum(initial_masses))

    calculator.calculate()
    previous = plugin.scf.calculator.state
    system.positions[parents] = system.positions.base[parents] + 0.01
    assert not np.allclose(plugin.scf.calculator.state, previous)

    from scipy.optimize import minimize

    def check_initial(objective, initial, **kwargs):
        np.testing.assert_allclose(initial, previous, atol=1e-14)
        return minimize(objective, initial, **kwargs)

    monkeypatch.setattr("scipy.optimize.minimize", check_initial)
    calculator.calculate()
