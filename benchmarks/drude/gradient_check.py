"""Finite differences of the relaxed Drude energy and post-SCF forces.

Real-atom displacements re-solve SCF and recompute virtual sites. Drude-only
partial derivatives start at the SCF solution but hold other coordinates fixed;
re-solving those displacements would erase the derivative being tested.
All positions are Angstrom, energies kJ/mol, and forces kJ/mol/Angstrom.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import runpy

import numpy as np
import openmm
from openmm import unit

FORCE_UNIT = unit.kilojoule_per_mole / unit.angstrom


class Backend:
    def __init__(self, kind, namespace, scf_tolerance=1e-3):
        self.kind = kind
        self.namespace = namespace
        self.scf_tolerance = scf_tolerance
        self.max_residual = 0.
        self.solve_calls = 0
        if kind == 'pydft':
            self.system = namespace['system']
            self.calculator = namespace['calculator']
            self.integrator = namespace['integrator']
            self.context = self.calculator.potential.base_context
        else:
            self.context = namespace['simmd'].context
            self.integrator = namespace['integrator']
            if kind == 'reference':
                system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(self.context.getSystem()))
                self.integrator = openmm.DrudeSCFIntegrator(0.001)
                self.integrator.setMinimizationErrorTolerance(1e-8)
                self.context = openmm.Context(system, self.integrator,
                                               openmm.Platform.getPlatformByName('Reference'))
                self.context.setPositions(namespace['state'].getPositions())
                self.scf_tolerance = 1e-8
        omm_system = self.context.getSystem()
        drude_force = next(f for f in omm_system.getForces() if isinstance(f, openmm.DrudeForce))
        self.drudes = np.array([drude_force.getParticleParameters(i)[0]
                               for i in range(drude_force.getNumParticles())])
        self.virtuals = np.array([i for i in range(omm_system.getNumParticles())
                                 if omm_system.isVirtualSite(i)], dtype=int)
        self.real = np.array([i for i in range(omm_system.getNumParticles())
                             if i not in self.drudes and i not in self.virtuals])
        self.initial = np.array(self.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom))

    def set_positions(self, positions):
        if self.kind == 'pydft':
            from pydft_qmmm.utils.virtual_sites import compute_positions
            self.system.positions[:] = compute_positions(self.system, positions)
            self.system.velocities[:] = 0.
        else:
            self.context.setPositions(positions * unit.angstrom)
            self.context.computeVirtualSites()

    def evaluate(self):
        # Always a fresh evaluation, including after the last SCF position update.
        if self.kind == 'pydft':
            result = self.calculator.calculate(return_components=False)
            return float(result.energy), np.array(result.forces), np.array(self.system.positions)
        state = self.context.getState(getEnergy=True, getForces=True, getPositions=True)
        return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                np.array(state.getForces(asNumpy=True).value_in_unit(FORCE_UNIT)),
                np.array(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)))

    def solve(self, positions):
        self.solve_calls += 1
        self.set_positions(positions)
        for _ in range(20):
            if self.kind == 'pydft':
                proposed, velocities = self.integrator.integrate(self.system)
                self.system.positions[:] = proposed
                self.system.velocities[:] = velocities
            else:
                self.integrator.step(1)
            energy, forces, final = self.evaluate()
            if not np.isfinite(energy) or not np.isfinite(forces).all():
                raise AssertionError(f'{self.kind}: non-finite post-SCF energy/forces')
            residual = float(np.sqrt(np.mean(forces[self.drudes]**2)))
            if residual <= self.scf_tolerance:
                self.max_residual = max(self.max_residual, residual)
                np.testing.assert_allclose(final[self.real], positions[self.real], atol=1e-13, rtol=0)
                return energy, forces, final
        raise AssertionError(f'{self.kind}: post-SCF RMS force {residual} exceeds {self.scf_tolerance}')


def load_backends(root):
    openmm.Platform.getPlatformByName('CPU').setPropertyDefaultValue('Threads', '1')
    namespaces = {}
    for name, script in [('pydft', 'run_pydft/run_pydft.py'), ('openmm', 'run_openmm/run_openMM.py')]:
        with contextlib.redirect_stdout(io.StringIO()):
            namespaces[name] = runpy.run_path(str(root / script))
    return [Backend('pydft', namespaces['pydft']), Backend('openmm', namespaces['openmm']),
            Backend('reference', namespaces['openmm'])]


def gradient_sweep(backend, steps):
    energy, analytic, equilibrium = backend.solve(backend.initial)
    rows = []
    for step in steps:
        for mode, indices in [('relaxed_real', backend.real), ('drude_partial', backend.drudes)]:
            numerical = np.empty((len(indices), 3))
            for row, atom in enumerate(indices):
                for axis in range(3):
                    energies = []
                    for sign in [1, -1]:
                        displaced = equilibrium.copy()
                        displaced[atom, axis] += sign * step
                        if mode == 'relaxed_real':
                            e, _, _ = backend.solve(displaced)
                        else:
                            backend.set_positions(displaced)
                            e, _, _ = backend.evaluate()
                        energies.append(e)
                    numerical[row, axis] = -(energies[0] - energies[1]) / (2 * step)
            error = numerical - analytic[indices]
            worst = np.unravel_index(np.argmax(np.abs(error)), error.shape)
            rows.append(dict(mode=mode, step_A=step, max_error_kjmol_A=float(np.max(np.abs(error))),
                             rms_error_kjmol_A=float(np.sqrt(np.mean(error**2))),
                             worst_atom=int(indices[worst[0]]), worst_axis='xyz'[worst[1]],
                             analytic_forces_kjmol_A=analytic[indices].tolist(),
                             finite_difference_forces_kjmol_A=numerical.tolist()))
    backend.solve(equilibrium)
    return dict(energy_kjmol=energy, drude_positions_A=equilibrium[backend.drudes].tolist(),
                drude_forces_kjmol_A=analytic[backend.drudes].tolist(),
                max_post_scf_rms_force_kjmol_A=backend.max_residual,
                scf_solves=backend.solve_calls, gradients=rows)


def run(root, steps):
    backends = load_backends(root)
    report = {'steps_A': steps, 'backends': {}}
    for backend in backends:
        print(f'Checking {backend.kind}...', flush=True)
        report['backends'][backend.kind] = gradient_sweep(backend, steps)
        for row in report['backends'][backend.kind]['gradients']:
            print(f"  {row['mode']} h={row['step_A']:g}: max |F + dE/dx|={row['max_error_kjmol_A']:.6g}", flush=True)
    a, b = (report['backends'][name] for name in ['pydft', 'openmm'])
    report['agreement'] = dict(
        energy_difference_kjmol=abs(a['energy_kjmol']-b['energy_kjmol']),
        max_drude_position_difference_A=float(np.max(np.abs(np.array(a['drude_positions_A'])-b['drude_positions_A']))),
        max_drude_force_difference_kjmol_A=float(np.max(np.abs(np.array(a['drude_forces_kjmol_A'])-b['drude_forces_kjmol_A']))),
    )
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--steps', type=float, nargs='+', default=[1e-2, 3e-3, 1e-3, 3e-4])
    parser.add_argument('--output', type=Path, default=Path('drude-gradients.json'))
    args = parser.parse_args()
    report = run(args.benchmark_root.resolve(), args.steps)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['agreement'], indent=2))
