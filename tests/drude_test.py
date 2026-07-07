from __future__ import annotations

import numpy as np
import openmm
import openmm.unit
import pytest

from pydft_qmmm.plugins.drude import OpenMMDrudeForceOracle
from pydft_qmmm.plugins.drude import DrudeData
from pydft_qmmm.plugins.drude import DrudeSolver
from pydft_qmmm.plugins.drude import drude_relaxation_step
from pydft_qmmm.plugins.drude import extract_drude_data
from pydft_qmmm.plugins.drude import extract_virtual_sites
from pydft_qmmm.integrators import DrudeLangevinIntegrator
from pydft_qmmm.integrators import DrudeSCFIntegrator
from pydft_qmmm.integrators import VerletIntegrator
from pydft_qmmm import Atom
from pydft_qmmm import System
from pydft_qmmm.plugins import SETTLE


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


def test_drude_solver_can_accept_openmm_style_stagnation():
    data = DrudeData(
        drude_indices=np.array([0]),
        parent_indices=np.array([1]),
        charges=np.array([-1.0]),
        polarizabilities=np.array([1.0]),
        force_constants=np.array([10.0]),
    )

    def stalled_oracle(positions):
        return np.array([[1.0, 0.0, 0.0]])

    solver = DrudeSolver(
        data,
        stalled_oracle,
        force_tolerance=1e-12,
        displacement_tolerance=1e-12,
        max_iterations=10,
        stagnation_ratio=0.9,
    )
    positions, info = solver.relax(np.zeros((2, 3)))

    assert info.converged
    assert info.iterations == 2
    assert positions[0, 0] == pytest.approx(2.0)


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


def test_openmm_drude_force_oracle_masks_selected_source_charges():
    omm_system = openmm.System()
    for _ in range(3):
        omm_system.addParticle(1.0)
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    nonbonded.addParticle(
        1.0*openmm.unit.elementary_charge,
        1.0*openmm.unit.nanometer,
        0.0*openmm.unit.kilojoule_per_mole,
    )
    for _ in range(2):
        nonbonded.addParticle(
            -1.0*openmm.unit.elementary_charge,
            1.0*openmm.unit.nanometer,
            0.0*openmm.unit.kilojoule_per_mole,
        )
    omm_system.addForce(nonbonded)
    context = openmm.Context(
        omm_system,
        openmm.VerletIntegrator(1.0*openmm.unit.femtosecond),
        openmm.Platform.getPlatformByName("Reference"),
    )

    class Potential:
        base_context = context

        def update_positions(self, positions):
            context.setPositions(
                openmm.unit.Quantity(
                    [openmm.Vec3(*row) for row in positions],
                    openmm.unit.angstrom,
                ),
            )

    data = DrudeData(
        drude_indices=np.array([1, 2]),
        parent_indices=np.array([1, 2]),
        charges=np.array([-1.0, -1.0]),
        polarizabilities=np.array([1.0, 1.0]),
        force_constants=np.array([1.0, 1.0]),
    )
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ],
    )

    unmasked = OpenMMDrudeForceOracle(Potential(), data)(positions)
    masked = OpenMMDrudeForceOracle(
        Potential(),
        data,
        zero_charge_atoms={0, 2},
        masked_drude_indices={1},
    )(positions)
    restored = OpenMMDrudeForceOracle(Potential(), data)(positions)

    assert abs(unmasked[0, 0]) > 1.0
    assert masked[0, 0] == pytest.approx(0.0)
    assert masked[1, 1] == pytest.approx(unmasked[1, 1])
    assert restored == pytest.approx(unmasked)


def test_virtual_site_positions_match_openmm():
    system = openmm.System()
    for mass in (1.0, 1.0, 0.0, 0.0):
        system.addParticle(mass)
    system.setVirtualSite(2, openmm.TwoParticleAverageSite(0, 1, 0.25, 0.75))
    system.setVirtualSite(
        3,
        openmm.OutOfPlaneSite(0, 1, 2, 0.2, 0.3, 0.1),
    )
    positions = np.array([
        [0.1, 0.2, 0.3],
        [1.1, -0.4, 0.7],
        [9.0, 9.0, 9.0],
        [8.0, 8.0, 8.0],
    ])
    context = openmm.Context(system, openmm.VerletIntegrator(0.001))
    context.setPositions(positions*openmm.unit.angstrom)
    context.computeVirtualSites()
    expected = context.getState(getPositions=True).getPositions(asNumpy=True)
    expected = np.asarray(expected.value_in_unit(openmm.unit.angstrom))
    actual = extract_virtual_sites(system).compute_positions(positions)
    assert np.allclose(actual, expected, atol=1e-13)


def test_local_coordinate_virtual_site_matches_openmm():
    system = openmm.System()
    for mass in (1.0, 1.0, 1.0, 0.0):
        system.addParticle(mass)
    system.setVirtualSite(
        3,
        openmm.LocalCoordinatesSite(
            [0, 1, 2],
            [1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
            openmm.Vec3(0.4, 0.3, 0.2),
        ),
    )
    positions = np.array([
        [0.1, 0.2, 0.3],
        [1.1, -0.4, 0.7],
        [0.4, 1.2, -0.2],
        [9.0, 9.0, 9.0],
    ])
    context = openmm.Context(system, openmm.VerletIntegrator(0.001))
    context.setPositions(positions*openmm.unit.angstrom)
    context.computeVirtualSites()
    expected = context.getState(getPositions=True).getPositions(asNumpy=True)
    expected = np.asarray(expected.value_in_unit(openmm.unit.angstrom))
    actual = extract_virtual_sites(system).compute_positions(positions)
    assert np.allclose(actual, expected, atol=1e-13)


def test_drude_langevin_deterministic_step_matches_openmm():
    omm_system = openmm.System()
    drude = omm_system.addParticle(0.4)
    parent = omm_system.addParticle(15.6)
    normal = omm_system.addParticle(1.0)
    drude_force = openmm.DrudeForce()
    drude_force.addParticle(drude, parent, -1, -1, -1, -1.0, 0.001, 1.0, 1.0)
    omm_system.addForce(drude_force)
    external = openmm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
    external.addGlobalParameter("k", 10.0)
    for index in range(3):
        external.addParticle(index, [])
    omm_system.addForce(external)
    omm_system.addForce(openmm.CMMotionRemover(1))

    positions = np.array([
        [0.02, 0.00, 0.00],
        [0.00, 0.00, 0.00],
        [0.10, 0.20, -0.10],
    ])
    velocities = np.array([
        [0.01, -0.02, 0.03],
        [-0.01, 0.01, 0.00],
        [0.04, 0.02, -0.03],
    ])
    reference_integrator = openmm.DrudeLangevinIntegrator(
        0.0, 1.0, 0.0, 20.0, 0.001,
    )
    reference_integrator.setMaxDrudeDistance(0.0)
    context = openmm.Context(omm_system, reference_integrator)
    context.setPositions(positions*openmm.unit.nanometer)
    context.setVelocities(
        velocities*openmm.unit.nanometer/openmm.unit.picosecond,
    )
    initial = context.getState(getForces=True)
    forces = np.asarray(initial.getForces(asNumpy=True).value_in_unit(
        openmm.unit.kilojoule_per_mole/openmm.unit.nanometer,
    ))
    reference_integrator.step(1)
    expected = context.getState(getPositions=True, getVelocities=True)

    atoms = [
        Atom(
            position=position*10,
            velocity=velocity*0.01,
            force=force/10,
            mass=mass,
            element="H",
            name=f"A{index}",
            residue_name="MOL",
        )
        for index, (position, velocity, force, mass) in enumerate(zip(
            positions, velocities, forces, (0.4, 15.6, 1.0),
        ))
    ]
    system = System(atoms)
    integrator = DrudeLangevinIntegrator(
        timestep=1.0,
        temperature=0.0,
        friction=0.001,
        drude_temperature=0.0,
        drude_friction=0.02,
        max_drude_distance=0.0,
    )

    class Potential:
        base_context = context

    class Calculator:
        potential = Potential()

    integrator.bind(Calculator())
    actual_positions, actual_velocities = integrator.integrate(system)
    expected_positions = np.asarray(
        expected.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom),
    )
    expected_velocities = np.asarray(
        expected.getVelocities(asNumpy=True).value_in_unit(
            openmm.unit.angstrom/openmm.unit.femtosecond,
        ),
    )
    assert np.allclose(actual_positions, expected_positions, atol=2e-7)
    assert np.allclose(actual_velocities, expected_velocities, atol=2e-7)


def test_settle_handles_drude_water_and_updates_virtual_site():
    omm_system = openmm.System()
    for mass in (15.6, 1.0, 1.0, 0.0, 0.4):
        omm_system.addParticle(mass)
    weights = (0.589781071, 0.2051094645, 0.2051094645)
    omm_system.setVirtualSite(
        3,
        openmm.ThreeParticleAverageSite(0, 1, 2, *weights),
    )
    drude_force = openmm.DrudeForce()
    drude_force.addParticle(4, 0, -1, -1, -1, -1.0, 0.001, 1.0, 1.0)
    omm_system.addForce(drude_force)
    context = openmm.Context(omm_system, openmm.VerletIntegrator(0.001))

    angle = np.deg2rad(104.52)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.9572, 0.0, 0.0],
        [0.9572*np.cos(angle), 0.9572*np.sin(angle), 0.0],
        [9.0, 9.0, 9.0],
        [0.01, 0.0, 0.0],
    ])
    masses = (15.6, 1.0, 1.0, 0.0, 0.4)
    names = ("O", "H1", "H2", "M", "OD")
    elements = ("O", "H", "H", "EP", "EP")
    atoms = [
        Atom(
            position=position,
            velocity=np.array([0.001*index, -0.001*index, 0.0]),
            mass=mass,
            residue=0,
            element=element,
            name=name,
            residue_name="HOH",
        )
        for index, (position, mass, name, element) in enumerate(zip(
            positions, masses, names, elements,
        ))
    ]
    system = System(atoms)
    integrator = DrudeLangevinIntegrator(
        1.0, 0.0, 0.0, 0.0, 0.0, max_drude_distance=0.0,
    )

    class Potential:
        base_context = context

    class Calculator:
        potential = Potential()

    integrator.bind(Calculator())
    settle = SETTLE(oh_distance=0.9572, hh_distance=1.5139006545)
    integrator.register_plugin(settle)
    updated, _ = integrator.integrate(system)

    assert settle._get_hoh_residues(system) == [[0, 1, 2]]
    assert np.linalg.norm(updated[1] - updated[0]) == pytest.approx(0.9572)
    assert np.linalg.norm(updated[2] - updated[0]) == pytest.approx(0.9572)
    assert np.linalg.norm(updated[2] - updated[1]) == pytest.approx(1.5139006545)
    assert updated[3] == pytest.approx(np.asarray(weights) @ updated[:3])


def test_drude_scf_integrator_separates_eom_from_relaxation():
    omm_system = openmm.System()
    drude = omm_system.addParticle(0.4)
    parent = omm_system.addParticle(10.0)
    drude_force = openmm.DrudeForce()
    drude_force.addParticle(drude, parent, -1, -1, -1, -1.0, 0.001, 1.0, 1.0)
    omm_system.addForce(drude_force)
    context = openmm.Context(omm_system, openmm.VerletIntegrator(0.001))

    atoms = [
        Atom(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.5, 0.0, 0.0]),
            force=np.array([1000.0, 0.0, 0.0]),
            mass=0.4,
            element="EP",
            name="D",
            residue_name="MOL",
        ),
        Atom(
            position=np.array([1.0, 0.0, 0.0]),
            velocity=np.array([0.1, 0.0, 0.0]),
            force=np.array([2.0, 0.0, 0.0]),
            mass=10.0,
            element="O",
            name="P",
            residue_name="MOL",
        ),
    ]
    system = System(atoms)

    class Potential:
        base_context = context

    class Calculator:
        potential = Potential()
        calculate_calls = 0

        def calculate(self):
            self.calculate_calls += 1

    calculator = Calculator()
    integrator = DrudeSCFIntegrator(VerletIntegrator(1.0))
    integrator.bind(calculator)
    positions, velocities = integrator.integrate(system)

    # The EOM stage neither calls the calculator nor moves the Drude.
    assert calculator.calculate_calls == 0
    assert positions[drude] == pytest.approx(system.positions[drude])
    assert velocities[drude] == pytest.approx(np.zeros(3))
    # The physical parent follows the wrapped Verlet EOM.
    assert velocities[parent, 0] == pytest.approx(0.10002)
    assert positions[parent, 0] == pytest.approx(1.10002)


def test_drude_scf_integrator_finds_mm_potential_in_composite():
    omm_system = openmm.System()
    omm_system.addParticle(0.4)
    omm_system.addParticle(10.0)
    force = openmm.DrudeForce()
    force.addParticle(0, 1, -1, -1, -1, -1.0, 0.001, 1.0, 1.0)
    omm_system.addForce(force)
    context = openmm.Context(omm_system, openmm.VerletIntegrator(0.001))

    class MM:
        class Potential:
            base_context = context
        potential = Potential()

    class QM:
        potential = object()

    class Composite:
        calculators = [QM(), MM()]

    integrator = DrudeSCFIntegrator(VerletIntegrator(1.0))
    integrator.bind(Composite())

    assert integrator.drude_indices.tolist() == [0]
