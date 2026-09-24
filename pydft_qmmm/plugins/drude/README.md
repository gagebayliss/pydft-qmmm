# Drude relaxation with fixed QM forces

`IntegratorDrudeSCF` relaxes Drude positions after each nuclear integration
step. It reuses the previous step's QM forces on the Drudes and evaluates
only the MM calculator during minimization. It does not converge the QM
wavefunction at each trial position.

`QMForceCache` observes forces from normal QM calculator evaluations.
`CachedForceCalculator` exposes those forces by reference and supplies their
linear potential energy. `DrudeCalculator` accepts this through its optional
`external_potential` argument, which can also be any other calculator
implementing the usual energy/force interface. The optimizer is independent
of how the external contribution is obtained.

For the cached-force implementation, minimization uses

```
E_trial(r_D) = E_MM(r_D) - F_QM_previous · (r_D - r_D_reference)
```

The corresponding Drude forces are `F_MM(r_D) + F_QM_previous`. The reference
coordinates set an arbitrary energy offset; the trial energy is not the
physical QM/MM total energy. After integration, the normal simulation force
calculation updates the electronic state.

This lagged-force approximation differs from mutual electronic/Drude SCF.
Convergence means the residual of the fixed-force problem is small, not that
the forces of a freshly converged QM calculation vanish on the Drudes.

## Usage

Build the MM or QM/MM calculator and simulation as usual, then register the
plugin on the simulation's integrator:

```python
from pydft_qmmm.plugins import IntegratorDrudeSCF

plugin = IntegratorDrudeSCF(
    simulation.calculator,
    forcefield=["swm4ndp.xml", "swm4ndp_residues.xml"],
    algorithm="L-BFGS-B",
    force_tolerance=0.10,  # kJ/mol/angstrom, maximum absolute component
    max_iterations=100,
)
simulation.integrator.register_plugin(plugin, index=0)
simulation.run_dynamics(10)
```

Registration at index 0 makes this the outermost integration wrapper, so it
relaxes Drudes after any existing SETTLE, stationary-atom, and virtual-site
updates. `run_dynamics` computes the initial total forces before integration.
The observer is installed when `IntegratorDrudeSCF` is constructed, so the
first force evaluation must occur after construction. For manual integration,
calculate total forces first and refresh them after each accepted step.
Energy-only evaluations do not overwrite the last force snapshot. Trying to
relax QM/MM Drudes before any forces have been observed raises `RuntimeError`.

The plugin supports an MM calculator or a composite with one `MM` and one
`QM` calculator, using the existing `calculator_group` labels. Composites
with additional correction terms are rejected because they require an
explicit decision about which forces change during optimization.

Only Drude positions are optimized. Their returned velocities are zero.
The input system positions are restored before the plugin returns, including
on exceptions. Nonconvergence raises `RuntimeError`; inspect `plugin.scf.result`
or enable `debug_log` for SciPy diagnostics.

For a standalone fixed-force minimization:

```python
from pydft_qmmm.plugins import (
    CachedForceCalculator, DrudeCalculator, DrudeSCF, QMForceCache,
)

observer = QMForceCache()
qm_calculator.register_plugin(observer)
qm_calculator.calculate()  # Normal QM evaluation, outside optimization.
external = CachedForceCalculator(system, observer)
adapter = DrudeCalculator(
    mm_calculator, drude_data, external_potential=external,
)
energy, converged = DrudeSCF(adapter).solve()
```

A standalone solve leaves the system at the accepted Drude coordinates and
returns `(trial_energy, converged)`. Omitting `external_potential` performs
MM relaxation. A general external calculator must return consistent energy
and forces at the current geometry; only the cached-force adapter uses the
linear approximation. The cache must remain unchanged during a solve.

## Gradient checks

Run `python -m pytest tests/drude_test.py -q -s` in an environment containing
OpenMM, SciPy, and Psi4. The tests use the SWM4-NDP water dimer with the OpenMM
Reference platform and, for QM/MM, HF/STO-3G electrostatic embedding.

Central differences at 0.0001 and 0.00005 angstrom check both the physical
energy before freezing and the linearized objective during relaxation.
The latter holds the QM force array fixed for every displacement. Integration
tests also prohibit QM calculator calls, verify the previous-force snapshot,
and check restoration after failures. These checks do not establish energy
conservation of the lagged-force dynamics.
