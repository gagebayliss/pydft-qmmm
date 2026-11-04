"""Temperature and pressure coupling without external energy engines."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from pydft_qmmm import Atom, System, VerletIntegrator
from pydft_qmmm.calculators import Calculator, Results
from pydft_qmmm.plugins import AndersenThermostat, MonteCarloBarostat, Stationary
from pydft_qmmm.utils import KB, pluggable_method


@dataclass(frozen=True)
class ToyCalculator(Calculator):
    energies: list = field(default_factory=lambda: [0., 0., 0.])

    @property
    def name(self):
        return "toy"

    @pluggable_method
    def calculate(self, return_forces=True, return_components=True):
        value = self.energies.pop(0)
        if isinstance(value, Exception):
            raise value
        return Results(value, np.ones_like(self.system.positions))


def make_system():
    return System([
        Atom(position=np.array([1., 1., 1.]), mass=16., residue=0),
        Atom(position=np.array([2., 1., 1.]), mass=1., residue=0),
        Atom(position=np.array([4., 4., 4.]), mass=0., residue=1),
    ], np.eye(3) * 10.)


def test_andersen_distribution_and_massless():
    system = System([Atom(mass=4., residue=i) for i in range(20000)]
                    + [Atom(mass=0., residue=20000)])
    integrator = VerletIntegrator(1.)
    integrator.register_plugin(AndersenThermostat(300., 100., seed=5))
    positions, velocities = integrator.integrate(system)
    np.testing.assert_allclose(positions, velocities)
    assert np.all(system.velocities == 0.)
    assert np.all(velocities[-1] == 0.)
    variance = KB * 300. * 1.e-7 / 4.
    assert np.var(velocities[:-1]) == pytest.approx(variance, rel=.025)
    assert abs(np.mean(velocities[:-1])) < .015 * np.sqrt(variance)


def test_collision_probability_and_selection():
    system = System([Atom(mass=1., residue=i) for i in range(10000)])
    integrator = VerletIntegrator(2.)
    integrator.register_plugin(AndersenThermostat(300., .2, seed=7))
    _, velocities = integrator.integrate(system)
    fraction = np.count_nonzero(np.any(velocities != 0., axis=1)) / len(system)
    assert fraction == pytest.approx(1. - np.exp(-.4), abs=.02)
    integrator = VerletIntegrator(1.)
    integrator.register_plugin(AndersenThermostat(300., 100., query="index 0", seed=1))
    _, velocities = integrator.integrate(system)
    assert np.any(velocities[0] != 0.)
    assert np.all(velocities[1:] == 0.)


def test_stationary_outer_wrapper():
    system = make_system()
    integrator = VerletIntegrator(1.)
    integrator.register_plugin(Stationary("all"))
    integrator.register_plugin(AndersenThermostat(300., 100., seed=1))
    positions, velocities = integrator.integrate(system)
    np.testing.assert_array_equal(positions, system.positions)
    assert np.all(velocities == 0.)


@pytest.mark.parametrize("outcome", [0., 1.e10, RuntimeError("trial failed")])
def test_volume_trial_transaction(outcome):
    system = make_system()
    original = [np.array(getattr(system, name), copy=True)
                for name in ("box", "positions", "velocities", "forces")]
    calculator = ToyCalculator(system, calculator_group="MM", energies=[0., outcome, 0.])
    plugin = MonteCarloBarostat(calculator, 0., 300., interval=1, seed=1)
    if isinstance(outcome, Exception):
        with pytest.raises(RuntimeError, match="trial failed"):
            plugin._attempt(system)
    else:
        plugin._attempt(system)
    if outcome == 0.:
        assert np.linalg.det(system.box) != pytest.approx(1000., abs=.001)
        np.testing.assert_allclose(system.positions[1] - system.positions[0], [1., 0., 0.])
        np.testing.assert_array_equal(system.forces, np.ones((3, 3)))
    else:
        for name, expected in zip(("box", "positions", "velocities", "forces"), original):
            np.testing.assert_array_equal(getattr(system, name), expected)


def test_barostat_interval_and_force_refresh():
    system = make_system()
    calculator = ToyCalculator(system, calculator_group="MM")
    integrator = VerletIntegrator(1.)
    integrator.register_plugin(MonteCarloBarostat(calculator, 0., 300., interval=2, seed=1))
    integrator.integrate(system)
    assert len(calculator.energies) == 3
    _, velocities = integrator.integrate(system)
    np.testing.assert_allclose(velocities[0], np.ones(3) * 1.e-4 / 16.)
    assert not calculator.energies
    with pytest.raises(ValueError, match="integrated system"):
        integrator.integrate(make_system())


@pytest.mark.parametrize("kwargs", [{"temperature": -1.}, {"collision_frequency": -1.},
                                   {"temperature": np.nan}])
def test_invalid_thermostat(kwargs):
    args = dict(temperature=300., collision_frequency=.01)
    args.update(kwargs)
    with pytest.raises(ValueError):
        AndersenThermostat(**args)


@pytest.mark.parametrize("kwargs", [{"temperature": 0.}, {"interval": 0},
                                   {"volume_step": -1.}, {"pressure": np.nan}])
def test_invalid_barostat(kwargs):
    args = dict(calculator=ToyCalculator(make_system(), calculator_group="MM"),
                temperature=300., pressure=1.)
    args.update(kwargs)
    with pytest.raises(ValueError):
        MonteCarloBarostat(**args)


def test_openmm_box_callback():
    import openmm
    import openmm.unit
    from types import SimpleNamespace
    from pydft_qmmm.interfaces.openmm.openmm_interface import OpenMMInterface

    contexts = []
    integrators = []
    for _ in range(2):
        omm_system = openmm.System()
        omm_system.addParticle(1.)
        integrators.append(openmm.VerletIntegrator(.001))
        contexts.append(openmm.Context(omm_system, integrators[-1],
                                      openmm.Platform.getPlatformByName("Reference")))
    interface = SimpleNamespace(base_context=contexts[0], aux_context=contexts[1])
    box = np.array([[20., 1., 2.], [0., 21., 3.], [0., 0., 22.]])
    OpenMMInterface.update_box(interface, box)
    for context in contexts:
        vectors = context.getState().getPeriodicBoxVectors(asNumpy=True)
        np.testing.assert_allclose(vectors.value_in_unit(openmm.unit.angstrom), box.T)


@pytest.mark.parametrize("pressure,draw,accepted", [(0., .9, True),
                                                   (1.e6, .9, False),
                                                   (0., .995, False)])
def test_pressure_and_molecular_jacobian(pressure, draw, accepted):
    from types import SimpleNamespace
    system = make_system()
    calculator = ToyCalculator(system, calculator_group="MM")
    plugin = MonteCarloBarostat(calculator, pressure, 300.)
    # A 1% contraction has acceptance .99**2 at zero pressure because
    # there are two molecules, despite there being three particles.
    # Use expansion for the pressure-work case.
    delta = 10. if pressure else -10.
    plugin._rng = SimpleNamespace(uniform=lambda low, high: delta,
                                  random=lambda: draw)
    plugin._attempt(system)
    assert len(calculator.energies) == (0 if accepted else 1)


def test_force_evaluation_failure_restores_accepted_trial():
    system = make_system()
    calculator = ToyCalculator(system, calculator_group="MM",
                               energies=[0., 0., RuntimeError("forces failed")])
    plugin = MonteCarloBarostat(calculator, 0., 300., seed=1)
    positions = np.array(system.positions, copy=True)
    with pytest.raises(RuntimeError, match="forces failed"):
        plugin._attempt(system)
    np.testing.assert_array_equal(system.positions, positions)
    np.testing.assert_array_equal(system.box, np.eye(3) * 10.)


def test_zero_collisions_and_nonperiodic_box():
    system = make_system()
    integrator = VerletIntegrator(1.)
    integrator.register_plugin(AndersenThermostat(300., 0., seed=1))
    _, velocities = integrator.integrate(system)
    np.testing.assert_array_equal(velocities, np.zeros((3, 3)))
    system.box[:] = 0.
    calculator = ToyCalculator(system, calculator_group="MM")
    with pytest.raises(ValueError, match="periodic volume"):
        MonteCarloBarostat(calculator, 1., 300.)._attempt(system)


def test_thermostat_with_settle_and_barostat():
    from pydft_qmmm.plugins import SETTLE
    height = np.sqrt(1. - (1.632981 / 2.)**2)
    system = System([
        Atom(position=np.array([2., 2., 2.]), mass=16., residue_name="HOH"),
        Atom(position=np.array([2. + height, 2. + 1.632981 / 2., 2.]),
             mass=1., residue_name="HOH"),
        Atom(position=np.array([2. + height, 2. - 1.632981 / 2., 2.]),
             mass=1., residue_name="HOH"),
    ], np.eye(3) * 10.)
    calculator = ToyCalculator(system, calculator_group="MM")
    integrator = VerletIntegrator(1.)
    integrator.register_plugin(MonteCarloBarostat(calculator, 0., 300., interval=1, seed=1))
    integrator.register_plugin(SETTLE())
    integrator.register_plugin(AndersenThermostat(300., 100., seed=1))
    positions, velocities = integrator.integrate(system)
    assert np.any(velocities != 0.)
    assert np.linalg.norm(positions[1] - positions[0]) == pytest.approx(1., abs=1.e-6)
    assert np.linalg.norm(positions[2] - positions[0]) == pytest.approx(1., abs=1.e-6)
    assert np.linalg.norm(positions[2] - positions[1]) == pytest.approx(1.632981, abs=1.e-6)
    np.testing.assert_array_equal(system.velocities, np.zeros((3, 3)))
