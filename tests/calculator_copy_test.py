"""Copy isolation and fixed-charge DeltaECV regression checks."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import openmm
import pytest

from pydft_qmmm import Atom, System
from pydft_qmmm.calculators import (
    CalculatorPlugin, CompositeCalculator, PotentialCalculator,
)
from pydft_qmmm.interfaces.openmm.openmm_interface import OpenMMPotential
from pydft_qmmm.plugins.plumed import DeltaECV, PLUMED


class RecordingPlugin(CalculatorPlugin):
    def __init__(self):
        self.energies = []

    def _modify_calculate(self, calculate):
        def inner(return_forces=True, return_components=True):
            result = calculate(return_forces, return_components)
            self.energies.append(result.energy)
            return result
        return inner


class QuadraticPotential:
    """Atomic terms plus a charge-dependent pair interaction."""
    def __init__(self, system):
        self.system = system

    def compute_energy(self):
        q = self.system.charges.base
        x = self.system.positions.base
        energy = np.sum(q[:, None] * x**2)
        if len(q) == 2:
            energy += q[0] * q[1] * np.sum((x[0] - x[1])**2)
        return float(energy)

    def compute_forces(self):
        q = self.system.charges.base
        x = self.system.positions.base
        forces = -2 * q[:, None] * x
        if len(q) == 2:
            pair = -2 * q[0] * q[1] * (x[0] - x[1])
            forces[0] += pair
            forces[1] -= pair
        return forces

    def compute_components(self):
        return {}


def make_calculator(system):
    return PotentialCalculator(
        system, QuadraticPotential(system), calculator_group="MM",
    )


def make_system():
    return System([
        Atom(position=np.array([1., 0., 0.]), charge=1., residue=0),
        Atom(position=np.array([3., 0., 0.]), charge=-1., residue=1),
    ], np.eye(3) * 20)


def test_composite_copy_preserves_current_state_and_isolates_plugins():
    system = make_system()
    child = make_calculator(system)
    child.register_plugin(RecordingPlugin())
    calculator = CompositeCalculator(system, [child, child])
    calculator.register_plugin(RecordingPlugin())
    calculator.calculate()  # Populate wrappers before copying.
    system.positions[0, 0] = 2.
    system.charges[0] = 0.5
    notifications = []
    system.positions.register_notifier(lambda _: notifications.append(True))

    copied = calculator.copy()
    assert copied.system is not system
    assert copied.calculators[0] is copied.calculators[1]
    copied_child = copied.calculators[0]
    assert copied_child.system is copied.system
    assert copied_child.potential.system is copied.system
    for original, clone in ((calculator, copied), (child, copied_child)):
        assert clone.active_plugins()[0] is not original.active_plugins()[0]
        assert clone.active_plugins()[0].calculator is clone
        assert clone.active_plugins()[0].energies == original.active_plugins()[0].energies
        assert clone.active_plugins()[0].energies is not original.active_plugins()[0].energies
    np.testing.assert_array_equal(copied.system.charges, system.charges)
    np.testing.assert_array_equal(copied.system.positions, system.positions)
    assert copied.calculate().energy == pytest.approx(calculator.calculate().energy)
    old_calls = len(child.active_plugins()[0].energies)
    copied.system.positions[0, 0] = 5.
    copied.system.box[0, 0] = 30.
    copied.system.charges[0] = 2.
    copied.calculate()
    assert len(child.active_plugins()[0].energies) == old_calls
    assert notifications == []
    assert system.positions[0, 0] == 2.
    assert system.box[0, 0] == 20.
    assert system.charges[0] == 0.5


@pytest.mark.parametrize("exception_scale", [None, 0.5, 0.25])
def test_openmm_copy_rebuilds_contexts_and_charge_observers(exception_scale, monkeypatch):
    system = make_system()
    native_system = openmm.System()
    force = openmm.NonbondedForce()
    for charge in system.charges:
        native_system.addParticle(1.)
        force.addParticle(float(charge), 1., 0.)
    if exception_scale is not None:
        force.addException(0, 1, -exception_scale, 1., 0.)
    native_system.addForce(force)
    auxiliary = openmm.System()
    for _ in system:
        auxiliary.addParticle(1.)
    platform = openmm.Platform.getPlatformByName("Reference")
    contexts = [
        openmm.Context(s, openmm.VerletIntegrator(0.001), platform)
        for s in (native_system, auxiliary)
    ]
    for context in contexts:
        context.setPositions(system.positions.base * 0.1)
    potential = OpenMMPotential(
        system, *contexts, *(np.ones((2, 3)) for _ in range(3)),
    )
    calculator = PotentialCalculator(system, potential, calculator_group="MM")
    system.charges[:] = system.charges.base.copy()  # Populate source cache.
    original = calculator.calculate()
    copied = calculator.copy()
    assert copied.potential.base_context is not potential.base_context
    assert copied.potential.aux_context is not potential.aux_context
    assert "_base_state_cache" not in vars(copied.potential)
    assert copied.calculate().energy == pytest.approx(original.energy)
    copied.system.charges[0] = 2.
    assert copied.potential._exceptions_cache is not potential._exceptions_cache
    assert copied.potential._exceptions_cache[0] is not potential._exceptions_cache[0]
    assert copied.calculate().energy == pytest.approx(2 * original.energy)
    assert calculator.calculate().energy == pytest.approx(original.energy)
    copied.system.positions[1, 0] = 5.
    assert copied.calculate().energy == pytest.approx(original.energy)
    np.testing.assert_allclose(calculator.calculate().forces, original.forces)
    np.testing.assert_array_equal(system.charges, [1., -1.])
    cached_exceptions = copied.potential._exceptions_cache[0]

    def unexpected_scan(*args):
        raise AssertionError("Repeated updates must reuse the exception cache")

    with monkeypatch.context() as patch:
        patch.setattr(openmm.NonbondedForce, "getExceptionParameters", unexpected_scan)
        copied.system.charges[1] = -2.
    assert copied.potential._exceptions_cache[0] is cached_exceptions
    assert copied.calculate().energy == pytest.approx(2 * original.energy)
    assert calculator.calculate().energy == pytest.approx(original.energy)


def test_delta_e_copies_plugins_without_recursion_or_repeated_charge_updates(monkeypatch):
    engines = []

    class Engine:
        def __init__(self):
            self.steps = []
            engines.append(self)

        def __deepcopy__(self, memo):
            raise AssertionError("Native PLUMED engines must not be deep-copied")

        def cmd(self, command, value=None):
            if command == "setStep":
                self.steps.append(value)

    monkeypatch.setitem(sys.modules, "plumed", SimpleNamespace(Plumed=Engine))

    def build_aux(self, system, forcefield):
        aux = System([Atom(charge={"neutral": 1., "anion": 2.}[forcefield])])
        return make_calculator(aux)

    monkeypatch.setattr(DeltaECV, "_build_aux_calculator", build_aux)
    system = make_system()
    calculator = make_calculator(system)
    cv = DeltaECV(
        "resid 0", "anion - neutral",
        states_forcefield={"neutral": "neutral", "anion": "anion"},
    )
    bias = PLUMED("", "unused.log", {"gap": cv})
    calculator.register_plugin(bias)
    calculator.register_plugin(RecordingPlugin())
    source_updates = []
    system.charges.register_notifier(lambda _: source_updates.append(True))
    calculator.calculate()
    alternate = cv._state_calculators["anion"]
    assert cv._state_calculators["neutral"] is None
    assert alternate.system is not system
    assert alternate.active_plugins()[0] is not bias
    assert alternate.active_plugins()[0].calculator is alternate
    assert alternate.active_plugins()[0]._bypass_plumed
    assert alternate.active_plugins()[0].plumed is None
    assert alternate.active_plugins()[0].arbitrary_cvs["gap"] is not cv
    assert alternate.active_plugins()[0].arbitrary_cvs["gap"]._state_calculators == {}
    alternate_updates = []
    alternate.system.charges.register_notifier(lambda _: alternate_updates.append(True))

    for position in (2., 4.):
        system.positions[0, 0] = position
        # Drive through PLUMED, then independently check the scalar and gradient.
        calculator.calculate()
        result = make_calculator(system).calculate()
        assert cv.get_scalar_cv(calculator, result) == pytest.approx(-(position - 3)**2)
        expected_gradient = np.zeros((2, 3))
        expected_gradient[:, 0] = [-2 * (position - 3), 2 * (position - 3)]
        np.testing.assert_allclose(cv.chain_rule(), expected_gradient)
        np.testing.assert_array_equal(alternate.system.positions, system.positions)
        for aux in cv._aux_calculators.values():
            np.testing.assert_array_equal(aux.system.positions, system.positions[[0]])
    assert source_updates == []
    assert alternate_updates == []
    assert len(engines) == 1
    assert engines[0].steps == [0, 1, 2]
    assert not bias._bypass_plumed
    np.testing.assert_array_equal(system.charges, [1., -1.])
