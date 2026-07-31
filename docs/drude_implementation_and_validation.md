# Drude Support in pydft-qmmm: Implementation and Validation

Last updated: 2026-07-03

## Scope and baseline

This document records all pydft-qmmm changes made for Drude oscillator
support on branch `drude-scf-openmm`. The pre-Drude baseline is release
`0.4.2`, commit `8a4a4fb`. The implementation consists of four commits:

| Revision | Purpose |
|---|---|
| `6639dfa` | Initial OpenMM Drude-force and delegated Drude-SCF support |
| `27f3bc5` | Native Drude metadata, force oracle, SCF solver, and plugin |
| `0297e7b` | Selective source-charge masking for QM/MM coupling |
| `798f47e` | Virtual sites, SETTLE support, Drude Langevin dynamics, and separated Drude-SCF EOM propagation |

The implementation deliberately supports two distinct physical models:

1. Extended-Lagrangian Drude dynamics through `DrudeLangevinIntegrator`.
2. Variational Drude-SCF dynamics through `DrudeSCFIntegrator`, where nuclear
   EOM propagation and Drude relaxation are separate operations.

The QM/MM path uses the second model. Drude coordinates are not propagated by
the nuclear EOM. They are optimized by the calculator, including the coupled
QM electronic/Drude SCF when `QMMMDrudeSCFPlugin` is installed.

## Implementation summary

### Public exports

`pydft_qmmm/__init__.py` now exports:

- `DrudeLangevinIntegrator`
- `DrudeSCFIntegrator`

`pydft_qmmm/integrators/__init__.py` exports both implementation modules.
`pydft_qmmm/plugins/__init__.py` exports the Drude plugin package.

### OpenMM interface construction

`pydft_qmmm/interfaces/openmm/openmm_factory.py` was extended to:

- accept `openmm.DrudeForce` as a supported force;
- accept explicit extra particles (`EP` and `LP`) with no chemical element;
- add force-field extra particles through `Modeller.addExtraParticles()`;
- expose `drude_engine` with values `"openmm"` and `"native"`;
- expose `drude_scf_tolerance`;
- construct an OpenMM `DrudeSCFIntegrator` for delegated relaxation when
  `drude_engine="openmm"` and a Drude force is present;
- construct a plain OpenMM Verlet context when `drude_engine="native"`, so
  pydft-qmmm or the QM SCF owns relaxation;
- transfer force-field masses and charges, including Drude and virtual-site
  particles, back into the pydft-qmmm `System`.

The `native` setting is required for both native MM Drude-SCF and coupled
QM/MM Drude-SCF. It prevents an OpenMM context integrator from independently
moving the same Drudes.

### Delegated OpenMM Drude-SCF

`pydft_qmmm/interfaces/openmm/openmm_interface.py` gained
`OpenMMPotential._relax_drude_positions()`. When the context owns an OpenMM
`DrudeSCFIntegrator`, it takes one integrator step, reads relaxed positions,
and synchronizes them into the pydft-qmmm system before energy, force, or
component evaluation. The method is a no-op for native contexts.

### OpenMM force utilities

`pydft_qmmm/interfaces/openmm/openmm_utils.py` was generalized so force
filtering and force-group state generation work with Drude-containing systems.
The implementation preserves Drude forces required by native relaxation while
continuing to support the existing QM/MM force masks.

### Drude metadata

`pydft_qmmm/plugins/drude/drude_data.py` defines immutable `DrudeData` with:

- Drude particle indices;
- parent particle indices;
- Drude charges in elementary charge;
- polarizabilities in nm^3;
- diagonal spring constants in kJ/mol/nm^2.

`extract_drude_data()` reads one OpenMM `DrudeForce` and calculates the
isotropic diagonal curvature as

```text
k = ONE_4PI_EPS0 * q^2 / polarizability
```

OpenMM remains responsible for the complete force, including spring,
electrostatics, PME, anisotropy, exclusions, and screened-pair terms.

### Native force oracle

`pydft_qmmm/plugins/drude/openmm_oracle.py` defines
`OpenMMDrudeForceOracle`. For each trial configuration it:

1. pushes pydft-qmmm positions into the OpenMM context;
2. recomputes virtual-site coordinates;
3. requests OpenMM forces;
4. returns the Drude rows in kJ/mol/nm.

For QM/MM bookkeeping it optionally supports:

- `zero_charge_atoms`: temporarily zero selected `NonbondedForce` particle
  charges and corresponding exception charge products;
- `masked_drude_indices`: replace only selected Drude force rows with forces
  obtained under the charge mask;
- exact restoration of original particle and exception parameters after the
  masked force evaluation.

This allows parent-subsystem-II Drudes to exclude mechanical subsystem-I MM
charge contributions while retaining the unmasked environment for other
Drudes.

### Native fixed-point Drude solver

`pydft_qmmm/plugins/drude/drude_solver.py` provides:

- `drude_relaxation_step()` for one diagonal-Newton update;
- `DrudeSolver.step()` for reusable one-step coupling to a QM SCF;
- `DrudeSolver.relax()` for converged standalone relaxation;
- `DrudeStepInfo` and `DrudeSCFInfo` diagnostics.

The update is

```text
drude_position += damping * drude_force / spring_constant
```

with conversion from nm to angstrom at the pydft-qmmm boundary. By default,
convergence follows OpenMM: the Cartesian-component RMS force tolerance is
1 kJ/mol/nm, at most 50 iterations are performed, and a `stagnation_ratio` of
`0.9` accepts a step when the sum of squared Drude forces no longer decreases
by at least 10%. Reaching the iteration limit returns the latest coordinates
with `converged=False` instead of raising. Maximum-step displacement
convergence remains available as an explicit extension but is disabled by
default.

The solver also provides optional diagonally preconditioned conjugate
gradient. `DrudeCGState` retains the force, spring-scaled force, and search
direction. `drude_conjugate_gradient_step()` uses a directional force probe,
Polak--Ribiere+ updates, downhill restarts, and residual-force backtracking.
The original diagonal update remains the default; select CG with
`DrudeSCF(algorithm="cg")`.

### Standalone MM Drude-SCF plugin

`pydft_qmmm/plugins/drude/drude_scf.py` defines the calculator plugin
`DrudeSCF`. Before the wrapped calculator evaluates energy and forces it:

1. verifies that OpenMM is configured with `drude_engine="native"`;
2. extracts and caches Drude topology;
3. builds an OpenMM force oracle and `DrudeSolver`;
4. relaxes Drudes at fixed physical-nuclear coordinates;
5. writes relaxed coordinates into the shared system;
6. recomputes virtual sites.

The plugin exposes force, displacement, iteration, damping, and optional
OpenMM-style stagnation settings. Its `last_info` attribute records the most
recent relaxation diagnostics.

### Virtual-site support

`pydft_qmmm/plugins/drude/virtual_sites.py` implements OpenMM-compatible
virtual-site position construction. `extract_virtual_sites()` topologically
orders dependent sites and rejects circular definitions. `VirtualSiteData`
supports:

- `TwoParticleAverageSite`;
- `ThreeParticleAverageSite`;
- `OutOfPlaneSite`;
- `LocalCoordinatesSite`;
- `SymmetrySite`, including periodic fractional-coordinate operation.

Coordinates are handled in angstrom and box-vector orientation is converted
between pydft-qmmm and OpenMM conventions. OpenMM already distributes MM
virtual-site forces to their parent particles during force evaluation, so the
native dynamics code only needs coordinate reconstruction.

### Extended-Lagrangian Drude dynamics

`pydft_qmmm/integrators/drude_langevin_integrator.py` implements OpenMM's
dual-thermostat `DrudeLangevinIntegrator` equations in pydft-qmmm units. It:

- separates each Drude-parent pair into center-of-mass and relative modes;
- thermostats the center-of-mass mode at the physical temperature;
- thermostats the relative Drude mode at the Drude temperature;
- applies forces and stochastic noise in the two modes;
- propagates normal massive particles;
- implements the optional maximum-Drude-distance hard wall;
- reproduces `CMMotionRemover` frequency behavior;
- applies constraints before hard-wall and virtual-site finalization;
- excludes massless virtual sites safely from integration and kinetic energy.

NumPy and OpenMM do not share a random-number stream. Deterministic steps can
be compared directly, while finite-temperature validation must compare
statistical observables.

### Variational Drude-SCF EOM propagation

`pydft_qmmm/integrators/drude_scf_integrator.py` is the QM/MM dynamics path.
It wraps an ordinary pydft-qmmm integrator such as `VerletIntegrator` and
strictly separates the operations:

```text
calculate/relax Drudes -> propagate physical nuclei -> apply constraints
-> reconstruct virtual sites -> calculate/relax Drudes -> ...
```

During `integrate()` it creates an isolated propagation view of the system,
zeros force and velocity on Drudes and virtual sites, and invokes the wrapped
integrator. It then restores the pre-EOM Drude coordinates and sets auxiliary
velocities to zero. It does not invoke a force oracle or calculator and cannot
relax a Drude.

The subsequent calculator evaluation owns relaxation:

- MM-only calculations use `DrudeSCF`;
- QM/MM calculations use `QMMMDrudeSCFPlugin`, which updates Drudes within the
  electronic SCF.

The wrapper can locate the OpenMM Drude potential in either a standalone
calculator or a composite QM/MM calculator. It requires a native OpenMM
context and excludes Drudes and virtual sites from physical kinetic energy.

### SETTLE integration

`pydft_qmmm/plugins/settle/settle.py` was updated for polarizable water:

- selected water residues may contain Drudes and virtual sites;
- Drude and virtual-site indices are excluded before identifying the
  three-atom SETTLE cluster;
- the heaviest remaining particle is placed first as oxygen;
- residue topology is cached for trajectory performance;
- massless sites no longer cause division by zero in shifted velocities;
- SCF auxiliary particles are excluded from kinetic energy;
- Drude post-step work is deferred until after constraints;
- virtual sites are reconstructed after constrained physical positions;
- Langevin hard-wall handling occurs in OpenMM's post-constraint order.

For SWM4-NDP the plugin is configured with O-H distance `0.9572 A` and H-H
distance `1.5139006545 A`.

### Simulation binding and zero-mass handling

`pydft_qmmm/wrappers/simulation.py` now calls an optional integrator `bind()`
method after calculator construction. This lets Drude integrators discover
OpenMM topology even when a Hamiltonian builds the calculator internally.

Massless virtual sites advertised by the bound integrator remain truly
massless and are not converted to stationary 0.1-Da placeholder particles.
Other zero-mass particles retain the previous stationary-particle behavior.

## QM/MM coupling behavior

The complementary `QMMMDrudeSCFPlugin` is implemented in the
qmmm-image-solver repository, but the pydft-qmmm changes above provide its
required APIs and force semantics. The current parent-subsystem rules are:

- all Drudes are active polarization variables;
- parent-II Drudes feel the QM electric field;
- parent-II Drudes exclude subsystem-I mechanical MM charge forces;
- parent-III Drudes do not feel the QM field and retain their normal MM
  environment;
- parent-I Drudes are stepped normally;
- parent-II Drude charges are removed from the static embedding path and fed
  into the Fock matrix through the dynamic Drude electronic potential.

At each MD step, `DrudeSCFIntegrator` propagates physical nuclei only. The
next composite calculator evaluation jointly converges the electronic density
and Drude polarization through the QM plugin.

## Usage

### Native MM Drude-SCF dynamics

```python
from pydft_qmmm import DrudeSCFIntegrator
from pydft_qmmm import MMHamiltonian
from pydft_qmmm import Simulation
from pydft_qmmm import VerletIntegrator
from pydft_qmmm.plugins import DrudeSCF
from pydft_qmmm.plugins import SETTLE

hamiltonian = MMHamiltonian(
    forcefield="swm4ndp.xml",
    nonbonded_method="PME",
    drude_engine="native",
)
integrator = DrudeSCFIntegrator(VerletIntegrator(1.0))
plugins = [
    DrudeSCF(
        force_tolerance=0.1,
        displacement_tolerance=2e-6,
        damping=0.5,
        stagnation_ratio=0.9,
    ),
    SETTLE(oh_distance=0.9572, hh_distance=1.5139006545),
]
simulation = Simulation(
    system,
    integrator,
    hamiltonian=hamiltonian,
    plugins=plugins,
)
simulation.run_dynamics(steps)
```

### QM/MM Drude-SCF dynamics

```python
integrator = DrudeSCFIntegrator(VerletIntegrator(1.0))
plugins = [
    QMMMDrudeSCFPlugin(),
    SETTLE(oh_distance=0.9572, hh_distance=1.5139006545),
]
```

The MM component of the QM/MM Hamiltonian must use `drude_engine="native"`.
`DrudeSCF` should not be stacked with `QMMMDrudeSCFPlugin`; the latter owns the
coupled relaxation.

## Validation

### Focused unit and integration tests

`tests/drude_test.py` validates:

- extraction of Drude indices, parents, charges, polarizabilities, and spring
  constants;
- native fixed-point convergence against an analytic force oracle;
- standalone one-step relaxation behavior;
- optional OpenMM-style stagnation termination;
- selective source-charge masking and exact parameter restoration;
- dependent average and out-of-plane virtual sites against OpenMM;
- local-coordinate virtual sites against OpenMM;
- deterministic `DrudeLangevinIntegrator` position and velocity agreement
  with OpenMM;
- SWM4-NDP-style SETTLE with a Drude and virtual M-site;
- strict separation of EOM propagation from Drude relaxation;
- discovery of an MM Drude potential inside a composite QM/MM calculator.

Final focused result:

```text
11 passed
```

### Full pydft-qmmm regression

After all Drude and trajectory changes:

```text
20 passed, 1 warning
```

The warning is the pre-existing warning for unstable I-II mechanical with
I-III none embedding.

### QM/MM Drude plugin regression

The qmmm-image-solver focused suite was run against the modified editable
pydft-qmmm source:

```text
14 passed
```

### Native MM single-point comparison

The original native solver validation used equilibrated SWM4-NDP water and
compared native relaxation against OpenMM Drude-SCF. Representative results
were:

- energy difference approximately `1e-3` to `4e-3 kJ/mol`;
- maximum Drude coordinate difference approximately `8e-7` to `1e-6 A`.

### Stepwise trajectory validation

The benchmark is located at:

```text
drude/benchmarks/swm4ndp_drude_scf_trajectory_validation/
```

It starts from equilibrated 64-, 256-, 512-, and 1024-water SWM4-NDP cells.
Each deterministic NVE benchmark runs pydft-qmmm and OpenMM in lockstep with
matched initial constrained velocities. It records both coordinate arrays,
energies, relaxation diagnostics, and the Drude movement during the EOM stage.

| Cell | Steps | Physical max delta | Drude max delta | EOM Drude move |
|---|---:|---:|---:|---:|
| 64 waters | 10 | `3.67e-6 A` | `4.95e-6 A` | `0 A` |
| 64 waters | 100 | `1.26e-4 A` | `1.22e-4 A` | `0 A` |
| 64 waters | 1000 (1 ps) | `2.26e-2 A` | `1.20e-2 A` | `0 A` |
| 256 waters | 25 | `1.28e-5 A` | `1.45e-5 A` | `0 A` |
| 256 waters | 100 | `2.71e-4 A` | `1.38e-4 A` | `0 A` |
| 512 waters | 25 | `4.24e-5 A` | `2.42e-5 A` | `0 A` |
| 1024 waters | 10 | `1.61e-5 A` | `1.42e-5 A` | `0 A` |

For the 64-water 1 ps run:

- physical RMS coordinate separation: `0.00353 A`;
- Drude RMS coordinate separation: `0.00286 A`;
- pydft-qmmm total-energy change: `2.59 kJ/mol`;
- OpenMM total-energy change: `3.39 kJ/mol`;
- SCF iterations at reported frames: 14 to 19;
- maximum EOM-stage Drude displacement: exactly `0 A`.

For the 1024-water 10-step run:

- physical RMS coordinate separation: `7.63e-7 A`;
- Drude RMS coordinate separation: `1.04e-6 A`;
- pydft-qmmm/OpenMM total-energy changes: `-120.08/-120.23 kJ/mol`.

The larger maximum separation at 1 ps is accumulated deterministic trajectory
divergence. Short-time agreement, RMS behavior, matched energy evolution, and
zero EOM-stage Drude displacement demonstrate that the two-stage SCF dynamics
cycle is operating as intended.

### Conjugate-gradient QM/MM validation

The QM/MM integration optionally nests CG updates at each fixed electronic
density. The QM field is evaluated once per electronic iteration and held
fixed during the inner CG probes, then reevaluated after the density changes.

Focused tests passed: 13 pydft-qmmm Drude tests and 19 qmmm-image-solver Drude
tests. A coupled two-Drude quadratic is solved below `1e-12 kJ/mol/nm` in two
CG updates. Tighter Drude convergence did not improve NVE energy conservation:

| System and embedding | Relaxation | Final force range (kJ/mol/nm) | Delta total energy over 1.5 fs (kJ/mol) |
|---|---|---:|---:|
| 256 water, cutoff | diagonal | 0.93--2.01 | +95.051 |
| 256 water, cutoff | CG | 0.027--0.756 | +95.048 |
| 1024 water, cutoff | diagonal | 2.55--4.06 | -67.544 |
| 1024 water, cutoff | CG | 0.048--0.894 | -68.319 |
| 1024 water, mechanical | diagonal | 2.54--4.06 | -64.165 |
| 1024 water, mechanical | CG | 0.047--0.920 | -64.963 |

CG is about three times slower in the 1024-water benchmark. Incomplete
diagonal relaxation is therefore not the source of the force--energy
inconsistency, and CG remains opt-in.

## Known limitations and follow-up validation

- The source-charge mask covers `NonbondedForce` particle charges and matching
  exception charge products. It does not remove arbitrary screened-pair terms
  from `DrudeForce`.
- OpenMM and NumPy stochastic streams differ, so finite-temperature Langevin
  trajectories cannot be compared coordinate by coordinate after noise is
  applied. Temperature, diffusion, structural distributions, polarization,
  and energy distributions should be compared statistically.
- The current trajectory benchmarks are MM-only tests of the dynamics and
  relaxation machinery. Full production QM/MM trajectories remain necessary
  to validate coupled electronic/Drude convergence and long-time observables.
- The shared qmmm-image-solver state currently has one active slot. Combining
  independently stacked image-charge and Drude SCF plugins may require unified
  shared-state ownership.
- The OpenMM-style stagnation criterion can terminate with maximum individual
  Drude force above the nominal tolerance, matching OpenMM behavior. Both the
  termination reason and resulting observables should be monitored in long
  production runs.

## Verbatim diffs

The following sections contain every pydft-qmmm Drude implementation diff,
verbatim and in application order. Generated benchmark outputs and changes in
the separate qmmm-image-solver repository are intentionally excluded.

### Commit `6639dfa`: initial OpenMM Drude support

```diff
diff --git a/pydft_qmmm/interfaces/openmm/openmm_factory.py b/pydft_qmmm/interfaces/openmm/openmm_factory.py
index fc3eb41..3759483 100644
--- a/pydft_qmmm/interfaces/openmm/openmm_factory.py
+++ b/pydft_qmmm/interfaces/openmm/openmm_factory.py
@@ -28,10 +28,12 @@ NEEDS_CUTOFF = ("PME", "EWALD", "CUTOFFPERIODIC", "CUTOFFNONPERIODIC")
 PERIODIC = ("PME", "EWALD", "CUTOFFPERIODIC")
 SUPPORTED_FORCES = (
     openmm.CMMotionRemover,
+    openmm.CMAPTorsionForce,
     openmm.CustomNonbondedForce,
     openmm.CustomBondForce,
     openmm.HarmonicAngleForce,
     openmm.HarmonicBondForce,
+    openmm.DrudeForce,
     openmm.NonbondedForce,
     openmm.PeriodicTorsionForce,
     openmm.RBTorsionForce,
@@ -161,9 +163,12 @@ def _build_omm_topology(
             chains[system.chains[atoms[0]]],
         )
         for j in atoms:
+            element = None
+            if system.elements[j].upper() not in {"EP", "LP"}:
+                element = openmm.app.Element.getBySymbol(system.elements[j])
             _ = omm_topology.addAtom(
                 system.names[j],
-                openmm.app.Element.getBySymbol(system.elements[j]),
+                element,
                 residue,
             )
     omm_topology.createStandardBonds()
@@ -330,7 +335,11 @@ def _build_omm_context(
         calculations, containing the System object and the specific
         platform to use, which is currently just the CPU platform.
     """
-    omm_integrator = openmm.VerletIntegrator(1. * openmm.unit.femtosecond)
+    if any(isinstance(f, openmm.DrudeForce) for f in omm_system.getForces()):
+        omm_integrator = openmm.DrudeSCFIntegrator(0.001 * openmm.unit.femtosecond)
+        omm_integrator.setMinimizationErrorTolerance(1e-6)
+    else:
+        omm_integrator = openmm.VerletIntegrator(1. * openmm.unit.femtosecond)
     # We currently only support the CPU platform.
     omm_platform = openmm.Platform.getPlatformByName("CPU")
     omm_context = openmm.Context(omm_system, omm_integrator, omm_platform)
diff --git a/pydft_qmmm/interfaces/openmm/openmm_interface.py b/pydft_qmmm/interfaces/openmm/openmm_interface.py
index 35647b2..6e51cb2 100644
--- a/pydft_qmmm/interfaces/openmm/openmm_interface.py
+++ b/pydft_qmmm/interfaces/openmm/openmm_interface.py
@@ -324,12 +324,26 @@ class OpenMMPotential(OpenMMInterface, AtomicPotential):
             whose masked forces will be included.
     """
 
+    def _relax_drude_positions(self) -> None:
+        """Relax Drude particles in the OpenMM context and sync positions."""
+        integrator = self.base_context.getIntegrator()
+        if not isinstance(integrator, openmm.DrudeSCFIntegrator):
+            return
+        integrator.step(1)
+        state = self.base_context.getState(getPositions=True)
+        positions = (
+            state.getPositions(asNumpy=True)
+            / openmm.unit.angstrom
+        )
+        self.system.positions[:] = positions
+
     def compute_energy(self) -> float:
         r"""Compute the energy of the system using OpenMM.
 
         Returns:
             The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the system.
         """
+        self._relax_drude_positions()
         base_state = openmm_utils._generate_state(self.base_context)
         energy = (
             base_state.getPotentialEnergy()
@@ -353,6 +367,7 @@ class OpenMMPotential(OpenMMInterface, AtomicPotential):
             (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) acting
             on atoms in the system.
         """
+        self._relax_drude_positions()
         base_state = openmm_utils._generate_state(self.base_context)
         forces = (
             self.base_force_mask * base_state.getForces(asNumpy=True)
@@ -387,6 +402,7 @@ class OpenMMPotential(OpenMMInterface, AtomicPotential):
             The components of the energy (:math:`\mathrm{kJ\;mol^{-1}}`)
             of the system.
         """
+        self._relax_drude_positions()
         components = {}
         for force in range(self.base_context.getSystem().getNumForces()):
             key = type(
diff --git a/pydft_qmmm/interfaces/openmm/openmm_utils.py b/pydft_qmmm/interfaces/openmm/openmm_utils.py
index 2c33f13..bc15eee 100644
--- a/pydft_qmmm/interfaces/openmm/openmm_utils.py
+++ b/pydft_qmmm/interfaces/openmm/openmm_utils.py
@@ -448,6 +448,20 @@ def _exclude_custom_nonbonded(
             other_atoms,
         )
 
+_exceptions_cache = {}
+
+def _get_non_zero_exceptions(force: openmm.nonbondedForce) -> list[int]:
+    global _exceptions_cache
+    if (n := force.getForceGroup()) in _exceptions_cache:
+        return _exceptions_cache[n]
+    exceptions = []
+    for i in range(force.getNumExceptions()):
+        x = force.getExceptionParameters(i)
+        if x[2]:
+            exceptions.append((i, x))
+    _exceptions_cache[n] = exceptions
+    return exceptions
+
 
 def _update_exceptions(
         force: openmm.nonbondedForce,
@@ -459,16 +473,11 @@ def _update_exceptions(
         force: The OpenMM NonbondedForce with exceptions to update.
         new_charges: The new partial charge (:math:`e`) of the atoms.
     """
-    exceptions = [
-        force.getExceptionParameters(
-            i,
-        ) for i in range(force.getNumExceptions())
-    ]
-    for i, x in enumerate(exceptions):
-        if x[2] / (elementary_charge**2):
-            q0, _, _ = force.getParticleParameters(x[0])
-            q1, _, _ = force.getParticleParameters(x[1])
-            qprod_old = q0 * q1 / (elementary_charge**2)
-            qprod_new = new_charges[x[0]] * new_charges[x[1]]
-            x[2] *= (qprod_new / qprod_old)
-            force.setExceptionParameters(i, *x)
+    for exception in _get_non_zero_exceptions(force):
+        i, x = exception
+        q0, _, _ = force.getParticleParameters(x[0])
+        q1, _, _ = force.getParticleParameters(x[1])
+        qprod_old = q0 * q1 / (elementary_charge**2)
+        qprod_new = new_charges[x[0]] * new_charges[x[1]]
+        x[2] *= (qprod_new / qprod_old)
+        force.setExceptionParameters(i, *x)
```

### Commit `27f3bc5`: native Drude-SCF relaxation

```diff
diff --git a/pydft_qmmm/interfaces/openmm/openmm_factory.py b/pydft_qmmm/interfaces/openmm/openmm_factory.py
index 3759483..da9ea90 100644
--- a/pydft_qmmm/interfaces/openmm/openmm_factory.py
+++ b/pydft_qmmm/interfaces/openmm/openmm_factory.py
@@ -48,6 +48,8 @@ def openmm_interface_factory(
         nonbonded_cutoff: float | int = 14.,
         pme_gridnumber: int | tuple[int, int, int] | None = None,
         pme_alpha: float | int | None = None,
+        drude_engine: str = "openmm",
+        drude_scf_tolerance: float | int = 1e-6,
 ) -> openmm_interface.OpenMMPotential:
     r"""Build the interface to OpenMM.
 
@@ -64,6 +66,12 @@ def openmm_interface_factory(
             lattice edge in PME summation.
         pme_alpha: The Gaussian width parameter in Ewald summation
             (:math:`\mathrm{nm^{-1}}`).
+        drude_engine: The engine used to relax Drude particles.  The
+            default, "openmm", uses OpenMM's DrudeSCFIntegrator.  The
+            "native" option leaves the context on a plain Verlet
+            integrator so a pydft-qmmm plugin can drive Drude-SCF.
+        drude_scf_tolerance: The minimization tolerance for OpenMM's
+            DrudeSCFIntegrator when drude_engine is "openmm".
 
     Returns:
         The OpenMM interface.
@@ -124,7 +132,12 @@ def openmm_interface_factory(
     )
     _adjust_system(system, base_system)
     aux_system = _empty_omm_system(system)
-    base_context = _build_omm_context(base_system, omm_modeller)
+    base_context = _build_omm_context(
+        base_system,
+        omm_modeller,
+        drude_engine,
+        drude_scf_tolerance,
+    )
     aux_context = _build_omm_context(aux_system, omm_modeller)
     wrapper = openmm_interface.OpenMMPotential(
         system,
@@ -189,7 +202,10 @@ def _build_omm_modeller(
         The internal representation of the system OpenMM, integrating
         the topology and atomic positions.
     """
-    omm_pos = [openmm.Vec3(*x)*openmm.unit.angstrom for x in system.positions]
+    omm_pos = openmm.unit.Quantity(
+        [openmm.Vec3(*x) for x in system.positions],
+        openmm.unit.angstrom,
+    )
     omm_modeller = openmm.app.Modeller(omm_topology, omm_pos)
     return omm_modeller
 
@@ -209,7 +225,7 @@ def _build_omm_forcefield(
         The internal representation of the force field for OpenMM.
     """
     omm_forcefield = openmm.app.ForceField(*forcefield)
-    # modeller.addExtraParticles(forcefield)
+    omm_modeller.addExtraParticles(omm_forcefield)
     return omm_forcefield
 
 
@@ -322,6 +338,8 @@ def _adjust_system(
 def _build_omm_context(
         omm_system: openmm.System,
         omm_modeller: openmm.app.Modeller,
+        drude_engine: str = "openmm",
+        drude_scf_tolerance: float | int = 1e-6,
 ) -> openmm.Context:
     """Build the OpenMM Context object.
 
@@ -329,15 +347,26 @@ def _build_omm_context(
         omm_system: The OpenMM representation of forces, constraints,
             and particles.
         omm_modeller: The OpenMM representation of the system.
+        drude_engine: The engine used to relax Drude particles.
+        drude_scf_tolerance: The minimization tolerance for OpenMM's
+            DrudeSCFIntegrator when drude_engine is "openmm".
 
     Returns:
         The OpenMM machinery required to perform energy and force
         calculations, containing the System object and the specific
         platform to use, which is currently just the CPU platform.
     """
-    if any(isinstance(f, openmm.DrudeForce) for f in omm_system.getForces()):
-        omm_integrator = openmm.DrudeSCFIntegrator(0.001 * openmm.unit.femtosecond)
-        omm_integrator.setMinimizationErrorTolerance(1e-6)
+    drude_engine = drude_engine.lower()
+    if drude_engine not in {"openmm", "native"}:
+        raise ValueError(f"Unsupported Drude engine: {drude_engine}")
+    if (
+            drude_engine == "openmm"
+            and any(isinstance(f, openmm.DrudeForce) for f in omm_system.getForces())
+    ):
+        omm_integrator = openmm.DrudeSCFIntegrator(
+            0.001 * openmm.unit.femtosecond,
+        )
+        omm_integrator.setMinimizationErrorTolerance(drude_scf_tolerance)
     else:
         omm_integrator = openmm.VerletIntegrator(1. * openmm.unit.femtosecond)
     # We currently only support the CPU platform.
diff --git a/pydft_qmmm/plugins/__init__.py b/pydft_qmmm/plugins/__init__.py
index 3279fcf..b2d1e41 100644
--- a/pydft_qmmm/plugins/__init__.py
+++ b/pydft_qmmm/plugins/__init__.py
@@ -8,6 +8,7 @@ __author__ = "John Pederson"
 from .atom_partition import *
 from .center import *
 from .centroid_partition import *
+from .drude import *
 from .firstatom_partition import *
 from .plumed import *
 from .rigid import *
diff --git a/pydft_qmmm/plugins/drude/__init__.py b/pydft_qmmm/plugins/drude/__init__.py
new file mode 100644
index 0000000..8420cd8
--- /dev/null
+++ b/pydft_qmmm/plugins/drude/__init__.py
@@ -0,0 +1,20 @@
+"""Plugins and helpers for Drude oscillator relaxation."""
+from __future__ import annotations
+
+__all__ = [
+    "DrudeData",
+    "DrudeSCF",
+    "DrudeSolver",
+    "DrudeStepInfo",
+    "OpenMMDrudeForceOracle",
+    "drude_relaxation_step",
+    "extract_drude_data",
+]
+
+from .drude_data import DrudeData
+from .drude_data import extract_drude_data
+from .drude_scf import DrudeSCF
+from .drude_solver import DrudeSolver
+from .drude_solver import DrudeStepInfo
+from .drude_solver import drude_relaxation_step
+from .openmm_oracle import OpenMMDrudeForceOracle
diff --git a/pydft_qmmm/plugins/drude/drude_data.py b/pydft_qmmm/plugins/drude/drude_data.py
new file mode 100644
index 0000000..a9c27d5
--- /dev/null
+++ b/pydft_qmmm/plugins/drude/drude_data.py
@@ -0,0 +1,78 @@
+"""Drude oscillator metadata extracted from OpenMM systems."""
+from __future__ import annotations
+
+__all__ = ["DrudeData", "extract_drude_data"]
+
+from dataclasses import dataclass
+
+import numpy as np
+from numpy.typing import NDArray
+import openmm
+import openmm.unit
+
+ONE_4PI_EPS0 = 138.93545764438198
+
+
+@dataclass(frozen=True)
+class DrudeData:
+    """Metadata needed to relax Drude oscillator positions.
+
+    Attributes:
+        drude_indices: Particle indices for Drude particles.
+        parent_indices: Particle indices for each Drude parent.
+        charges: Drude particle charges in elementary charge.
+        polarizabilities: Drude polarizabilities in nm^3.
+        force_constants: Harmonic spring constants in kJ/mol/nm^2.
+    """
+    drude_indices: NDArray[np.int64]
+    parent_indices: NDArray[np.int64]
+    charges: NDArray[np.float64]
+    polarizabilities: NDArray[np.float64]
+    force_constants: NDArray[np.float64]
+
+    def __len__(self) -> int:
+        """Get the number of Drude oscillators."""
+        return len(self.drude_indices)
+
+
+def extract_drude_data(omm_system: openmm.System) -> DrudeData:
+    """Extract Drude oscillator metadata from an OpenMM system.
+
+    Args:
+        omm_system: The OpenMM system containing one DrudeForce.
+
+    Returns:
+        Drude oscillator metadata for fixed-point relaxation.
+    """
+    drude_forces = [
+        force for force in omm_system.getForces()
+        if isinstance(force, openmm.DrudeForce)
+    ]
+    if not drude_forces:
+        raise ValueError("The OpenMM system does not contain a DrudeForce.")
+    if len(drude_forces) > 1:
+        raise ValueError("Expected one DrudeForce in the OpenMM system.")
+    drude_force = drude_forces[0]
+    drude_indices = []
+    parent_indices = []
+    charges = []
+    polarizabilities = []
+    force_constants = []
+    for i in range(drude_force.getNumParticles()):
+        particle, parent, *_rest, charge, polarizability, _a12, _a34 = (
+            drude_force.getParticleParameters(i)
+        )
+        charge_e = charge / openmm.unit.elementary_charge
+        alpha_nm3 = polarizability / openmm.unit.nanometer**3
+        drude_indices.append(int(particle))
+        parent_indices.append(int(parent))
+        charges.append(float(charge_e))
+        polarizabilities.append(float(alpha_nm3))
+        force_constants.append(float(ONE_4PI_EPS0 * charge_e**2 / alpha_nm3))
+    return DrudeData(
+        drude_indices=np.array(drude_indices, dtype=np.int64),
+        parent_indices=np.array(parent_indices, dtype=np.int64),
+        charges=np.array(charges, dtype=float),
+        polarizabilities=np.array(polarizabilities, dtype=float),
+        force_constants=np.array(force_constants, dtype=float),
+    )
diff --git a/pydft_qmmm/plugins/drude/drude_scf.py b/pydft_qmmm/plugins/drude/drude_scf.py
new file mode 100644
index 0000000..79e944e
--- /dev/null
+++ b/pydft_qmmm/plugins/drude/drude_scf.py
@@ -0,0 +1,91 @@
+"""Calculator plugin for native Drude-SCF relaxation."""
+from __future__ import annotations
+
+__all__ = ["DrudeSCF"]
+
+from collections.abc import Callable
+from typing import TYPE_CHECKING
+
+import openmm
+
+from pydft_qmmm.calculators import CalculatorPlugin
+
+from .drude_data import extract_drude_data
+from .drude_solver import DrudeSolver
+from .openmm_oracle import OpenMMDrudeForceOracle
+
+if TYPE_CHECKING:
+    from pydft_qmmm.calculators import Results
+    from .drude_solver import DrudeSCFInfo
+
+
+class DrudeSCF(CalculatorPlugin):
+    """Relax Drude oscillators before calculator evaluations.
+
+    Args:
+        force_tolerance: Maximum Drude-particle force norm required for
+            convergence, in kJ/mol/nm.
+        displacement_tolerance: Maximum Drude-particle displacement in
+            one iteration required for convergence, in Angstrom.
+        max_iterations: Maximum fixed-point iterations.
+        damping: Scalar multiplier applied to each diagonal-Newton step.
+    """
+
+    def __init__(
+            self,
+            force_tolerance: float = 5e-2,
+            displacement_tolerance: float = 1e-6,
+            max_iterations: int = 100,
+            damping: float = 1.0,
+    ) -> None:
+        self.force_tolerance = force_tolerance
+        self.displacement_tolerance = displacement_tolerance
+        self.max_iterations = max_iterations
+        self.damping = damping
+        self._solver: DrudeSolver | None = None
+        self.last_info: DrudeSCFInfo | None = None
+
+    def _get_solver(self) -> DrudeSolver:
+        """Build or retrieve the Drude solver for this calculator."""
+        if self._solver is not None:
+            return self._solver
+        potential = self.calculator.potential
+        integrator = potential.base_context.getIntegrator()
+        if isinstance(integrator, openmm.DrudeSCFIntegrator):
+            raise RuntimeError(
+                "DrudeSCF requires an OpenMM context with "
+                'drude_engine="native"; the current context uses '
+                "OpenMM's DrudeSCFIntegrator.",
+            )
+        data = extract_drude_data(potential.base_context.getSystem())
+        oracle = OpenMMDrudeForceOracle(potential, data)
+        self._solver = DrudeSolver(
+            data,
+            oracle,
+            force_tolerance=self.force_tolerance,
+            displacement_tolerance=self.displacement_tolerance,
+            max_iterations=self.max_iterations,
+            damping=self.damping,
+        )
+        return self._solver
+
+    def relax(self) -> None:
+        """Relax Drude positions and update the calculator system."""
+        solver = self._get_solver()
+        positions, info = solver.relax(self.calculator.system.positions)
+        self.calculator.system.positions[:] = positions
+        self.calculator.potential.base_context.computeVirtualSites()
+        self.last_info = info
+
+    def _modify_calculate(
+            self,
+            calculate: Callable[[bool, bool], Results],
+    ) -> Callable[[bool, bool], Results]:
+        """Modify the calculate routine to relax Drudes beforehand."""
+        def inner(
+                return_forces: bool = True,
+                return_components: bool = True,
+        ) -> Results:
+            self.relax()
+            return calculate(return_forces, return_components)
+        return inner
diff --git a/pydft_qmmm/plugins/drude/drude_solver.py b/pydft_qmmm/plugins/drude/drude_solver.py
new file mode 100644
index 0000000..b9fb6d8
--- /dev/null
+++ b/pydft_qmmm/plugins/drude/drude_solver.py
@@ -0,0 +1,159 @@
+"""Native Drude-SCF fixed-point solver."""
+from __future__ import annotations
+
+__all__ = [
+    "DrudeSCFInfo",
+    "DrudeStepInfo",
+    "DrudeSolver",
+    "drude_relaxation_step",
+]
+
+from collections.abc import Callable
+from dataclasses import dataclass
+
+import numpy as np
+from numpy.typing import NDArray
+
+from .drude_data import DrudeData
+
+
+@dataclass(frozen=True)
+class DrudeStepInfo:
+    """Diagnostics from one Drude relaxation step."""
+    max_force: float
+    max_displacement: float
+
+
+@dataclass(frozen=True)
+class DrudeSCFInfo:
+    """Convergence information from a Drude relaxation."""
+    iterations: int
+    final_max_force: float
+    final_max_displacement: float
+    converged: bool
+
+
+def drude_relaxation_step(
+        data: DrudeData,
+        positions: NDArray[np.float64],
+        forces: NDArray[np.float64],
+        *,
+        damping: float = 1.0,
+) -> tuple[NDArray[np.float64], DrudeStepInfo]:
+    """Apply one diagonal-Newton Drude relaxation step.
+
+    Args:
+        data: Drude oscillator metadata.
+        positions: Full system positions in Angstrom.
+        forces: Forces on Drude particles in kJ/mol/nm.
+        damping: Scalar multiplier applied to the diagonal-Newton step.
+
+    Returns:
+        Updated full-system positions in Angstrom and step diagnostics.
+    """
+    relaxed = np.array(positions, dtype=float, copy=True)
+    displacement_nm = (
+        damping
+        * forces
+        / data.force_constants.reshape((-1, 1))
+    )
+    displacement_ang = 10.0*displacement_nm
+    relaxed[data.drude_indices, :] += displacement_ang
+    force_norms = np.linalg.norm(forces, axis=1)
+    displacement_norms = np.linalg.norm(displacement_ang, axis=1)
+    return relaxed, DrudeStepInfo(
+        max_force=float(np.max(force_norms, initial=0.0)),
+        max_displacement=float(np.max(displacement_norms, initial=0.0)),
+    )
+
+
+class DrudeSolver:
+    """Relax Drude oscillator positions at fixed real-atom positions.
+
+    Args:
+        data: Drude oscillator metadata.
+        force_oracle: Callable returning forces on Drude particles in
+            kJ/mol/nm for a full positions array in Angstrom.
+        force_tolerance: Maximum Drude-particle force norm required for
+            convergence, in kJ/mol/nm.
+        displacement_tolerance: Maximum Drude-particle displacement in
+            one iteration required for convergence, in Angstrom.
+        max_iterations: Maximum fixed-point iterations.
+        damping: Scalar multiplier applied to each diagonal-Newton step.
+    """
+
+    def __init__(
+            self,
+            data: DrudeData,
+            force_oracle: Callable[[NDArray[np.float64]], NDArray[np.float64]],
+            *,
+            force_tolerance: float = 5e-2,
+            displacement_tolerance: float = 1e-6,
+            max_iterations: int = 100,
+            damping: float = 1.0,
+    ) -> None:
+        self.data = data
+        self.force_oracle = force_oracle
+        self.force_tolerance = force_tolerance
+        self.displacement_tolerance = displacement_tolerance
+        self.max_iterations = max_iterations
+        self.damping = damping
+
+    def step(
+            self,
+            positions: NDArray[np.float64],
+    ) -> tuple[NDArray[np.float64], DrudeStepInfo]:
+        """Apply one Drude relaxation step.
+
+        Args:
+            positions: Full system positions in Angstrom.
+
+        Returns:
+            Updated full-system positions in Angstrom and step
+            diagnostics.
+        """
+        forces = self.force_oracle(positions)
+        return drude_relaxation_step(
+            self.data,
+            positions,
+            forces,
+            damping=self.damping,
+        )
+
+    def relax(
+            self,
+            positions: NDArray[np.float64],
+    ) -> tuple[NDArray[np.float64], DrudeSCFInfo]:
+        """Relax Drude positions.
+
+        Args:
+            positions: Full system positions in Angstrom.
+
+        Returns:
+            Relaxed full-system positions in Angstrom and convergence
+            diagnostics.
+        """
+        relaxed = np.array(positions, dtype=float, copy=True)
+        final_max_force = np.inf
+        final_max_displacement = np.inf
+        for iteration in range(1, self.max_iterations + 1):
+            stepped, step_info = self.step(relaxed)
+            final_max_force = step_info.max_force
+            final_max_displacement = step_info.max_displacement
+            if (
+                    final_max_force < self.force_tolerance
+                    or final_max_displacement < self.displacement_tolerance
+            ):
+                return relaxed, DrudeSCFInfo(
+                    iterations=iteration,
+                    final_max_force=final_max_force,
+                    final_max_displacement=final_max_displacement,
+                    converged=True,
+                )
+            relaxed = stepped
+        raise RuntimeError(
+            "Drude SCF did not converge after "
+            f"{self.max_iterations} iterations; "
+            f"max force = {final_max_force:.6g} kJ/mol/nm, "
+            f"max displacement = {final_max_displacement:.6g} A",
+        )
diff --git a/pydft_qmmm/plugins/drude/openmm_oracle.py b/pydft_qmmm/plugins/drude/openmm_oracle.py
new file mode 100644
index 0000000..af8b1b9
--- /dev/null
+++ b/pydft_qmmm/plugins/drude/openmm_oracle.py
@@ -0,0 +1,41 @@
+"""OpenMM force oracle for native Drude-SCF relaxation."""
+from __future__ import annotations
+
+__all__ = ["OpenMMDrudeForceOracle"]
+
+from typing import TYPE_CHECKING
+
+import openmm.unit
+
+from pydft_qmmm.interfaces.openmm import openmm_utils
+
+if TYPE_CHECKING:
+    import numpy as np
+    from numpy.typing import NDArray
+    from pydft_qmmm.interfaces.openmm.openmm_interface import OpenMMPotential
+    from .drude_data import DrudeData
+
+
+class OpenMMDrudeForceOracle:
+    """Read Drude particle forces from an OpenMM-backed potential."""
+
+    def __init__(
+            self,
+            potential: OpenMMPotential,
+            data: DrudeData,
+    ) -> None:
+        self.potential = potential
+        self.data = data
+
+    def __call__(
+            self,
+            positions: NDArray[np.float64],
+    ) -> NDArray[np.float64]:
+        """Return forces on Drude particles in kJ/mol/nm."""
+        self.potential.update_positions(positions)
+        self.potential.base_context.computeVirtualSites()
+        state = openmm_utils._generate_state(self.potential.base_context)
+        forces = state.getForces(asNumpy=True).value_in_unit(
+            openmm.unit.kilojoule_per_mole/openmm.unit.nanometer,
+        )
+        return forces[self.data.drude_indices, :]
diff --git a/tests/drude_test.py b/tests/drude_test.py
new file mode 100644
index 0000000..73d5db2
--- /dev/null
+++ b/tests/drude_test.py
@@ -0,0 +1,84 @@
+from __future__ import annotations
+
+import numpy as np
+import openmm
+import openmm.unit
+import pytest
+
+from pydft_qmmm.plugins.drude import DrudeData
+from pydft_qmmm.plugins.drude import DrudeSolver
+from pydft_qmmm.plugins.drude import drude_relaxation_step
+from pydft_qmmm.plugins.drude import extract_drude_data
+
+
+def test_extract_drude_data():
+    system = openmm.System()
+    parent = system.addParticle(16.0)
+    drude = system.addParticle(0.4)
+    force = openmm.DrudeForce()
+    force.addParticle(
+        drude,
+        parent,
+        -1,
+        -1,
+        -1,
+        -1.0*openmm.unit.elementary_charge,
+        0.001*openmm.unit.nanometer**3,
+        0.0,
+        0.0,
+    )
+    system.addForce(force)
+
+    data = extract_drude_data(system)
+
+    assert data.drude_indices.tolist() == [drude]
+    assert data.parent_indices.tolist() == [parent]
+    assert data.charges.tolist() == [-1.0]
+    assert data.polarizabilities.tolist() == [0.001]
+    assert data.force_constants[0] == pytest.approx(138935.45764438197)
+
+
+def test_drude_solver_relaxes_against_force_oracle():
+    data = DrudeData(
+        drude_indices=np.array([0]),
+        parent_indices=np.array([1]),
+        charges=np.array([-1.0]),
+        polarizabilities=np.array([1.0]),
+        force_constants=np.array([10.0]),
+    )
+
+    def oracle(positions):
+        x_nm = positions[0, 0] / 10.0
+        force = np.zeros((1, 3))
+        force[0, 0] = -10.0*(x_nm - 0.02)
+        return force
+
+    solver = DrudeSolver(
+        data,
+        oracle,
+        force_tolerance=1e-12,
+        displacement_tolerance=1e-12,
+    )
+    positions, info = solver.relax(np.zeros((2, 3)))
+
+    assert positions[0, 0] == pytest.approx(0.2)
+    assert info.converged
+
+
+def test_drude_relaxation_step_is_standalone():
+    data = DrudeData(
+        drude_indices=np.array([0]),
+        parent_indices=np.array([1]),
+        charges=np.array([-1.0]),
+        polarizabilities=np.array([1.0]),
+        force_constants=np.array([20.0]),
+    )
+    positions = np.zeros((2, 3))
+    forces = np.array([[2.0, 0.0, 0.0]])
+
+    updated, info = drude_relaxation_step(data, positions, forces)
+
+    assert updated[0, 0] == pytest.approx(1.0)
+    assert positions[0, 0] == 0.0
+    assert info.max_force == pytest.approx(2.0)
+    assert info.max_displacement == pytest.approx(1.0)
```

### Commit `0297e7b`: QM/MM force masking

```diff
diff --git a/pydft_qmmm/plugins/drude/openmm_oracle.py b/pydft_qmmm/plugins/drude/openmm_oracle.py
index af8b1b9..90013e8 100644
--- a/pydft_qmmm/plugins/drude/openmm_oracle.py
+++ b/pydft_qmmm/plugins/drude/openmm_oracle.py
@@ -3,15 +3,18 @@ from __future__ import annotations
 
 __all__ = ["OpenMMDrudeForceOracle"]
 
+from collections.abc import Iterable
+from contextlib import contextmanager
 from typing import TYPE_CHECKING
 
+import numpy as np
+import openmm
 import openmm.unit
+from numpy.typing import NDArray
 
 from pydft_qmmm.interfaces.openmm import openmm_utils
 
 if TYPE_CHECKING:
-    import numpy as np
-    from numpy.typing import NDArray
     from pydft_qmmm.interfaces.openmm.openmm_interface import OpenMMPotential
     from .drude_data import DrudeData
 
@@ -23,9 +26,73 @@ class OpenMMDrudeForceOracle:
             self,
             potential: OpenMMPotential,
             data: DrudeData,
+            *,
+            zero_charge_atoms: Iterable[int] | None = None,
+            masked_drude_indices: Iterable[int] | None = None,
     ) -> None:
         self.potential = potential
         self.data = data
+        self.zero_charge_atoms = frozenset(zero_charge_atoms or ())
+        self.masked_drude_indices = frozenset(masked_drude_indices or ())
+
+    def _forces(self) -> NDArray[np.float64]:
+        """Return all OpenMM forces in kJ/mol/nm."""
+        self.potential.base_context.computeVirtualSites()
+        state = openmm_utils._generate_state(self.potential.base_context)
+        return state.getForces(asNumpy=True).value_in_unit(
+            openmm.unit.kilojoule_per_mole/openmm.unit.nanometer,
+        )
+
+    @contextmanager
+    def _zeroed_charges(self):
+        """Temporarily zero selected NonbondedForce particle charges."""
+        if not self.zero_charge_atoms:
+            yield
+            return
+        base_system = self.potential.base_context.getSystem()
+        nonbonded_forces = [
+            force for force in base_system.getForces()
+            if isinstance(force, openmm.NonbondedForce)
+        ]
+        originals = []
+        try:
+            for force in nonbonded_forces:
+                particle_params = []
+                for atom in range(force.getNumParticles()):
+                    charge, sigma, epsilon = force.getParticleParameters(atom)
+                    particle_params.append((atom, charge, sigma, epsilon))
+                    if atom in self.zero_charge_atoms:
+                        force.setParticleParameters(atom, 0.0, sigma, epsilon)
+                exception_params = []
+                for exception in range(force.getNumExceptions()):
+                    p1, p2, chargeprod, sigma, epsilon = (
+                        force.getExceptionParameters(exception)
+                    )
+                    exception_params.append(
+                        (exception, p1, p2, chargeprod, sigma, epsilon),
+                    )
+                    if (
+                            p1 in self.zero_charge_atoms
+                            or p2 in self.zero_charge_atoms
+                    ):
+                        force.setExceptionParameters(
+                            exception,
+                            p1,
+                            p2,
+                            0.0,
+                            sigma,
+                            epsilon,
+                        )
+                force.updateParametersInContext(self.potential.base_context)
+                originals.append((force, particle_params, exception_params))
+            yield
+        finally:
+            for force, particle_params, exception_params in originals:
+                for atom, charge, sigma, epsilon in particle_params:
+                    force.setParticleParameters(atom, charge, sigma, epsilon)
+                for params in exception_params:
+                    force.setExceptionParameters(*params)
+                force.updateParametersInContext(self.potential.base_context)
 
     def __call__(
             self,
@@ -33,9 +100,11 @@ class OpenMMDrudeForceOracle:
     ) -> NDArray[np.float64]:
         """Return forces on Drude particles in kJ/mol/nm."""
         self.potential.update_positions(positions)
-        self.potential.base_context.computeVirtualSites()
-        state = openmm_utils._generate_state(self.potential.base_context)
-        forces = state.getForces(asNumpy=True).value_in_unit(
-            openmm.unit.kilojoule_per_mole/openmm.unit.nanometer,
-        )
+        forces = self._forces()
+        if self.masked_drude_indices and self.zero_charge_atoms:
+            with self._zeroed_charges():
+                masked_forces = self._forces()
+            for atom_index in self.data.drude_indices:
+                if int(atom_index) in self.masked_drude_indices:
+                    forces[atom_index, :] = masked_forces[atom_index, :]
         return forces[self.data.drude_indices, :]
diff --git a/tests/drude_test.py b/tests/drude_test.py
index 73d5db2..b21e3b2 100644
--- a/tests/drude_test.py
+++ b/tests/drude_test.py
@@ -5,6 +5,7 @@ import openmm
 import openmm.unit
 import pytest
 
+from pydft_qmmm.plugins.drude import OpenMMDrudeForceOracle
 from pydft_qmmm.plugins.drude import DrudeData
 from pydft_qmmm.plugins.drude import DrudeSolver
 from pydft_qmmm.plugins.drude import drude_relaxation_step
@@ -82,3 +83,68 @@ def test_drude_relaxation_step_is_standalone():
     assert positions[0, 0] == 0.0
     assert info.max_force == pytest.approx(2.0)
     assert info.max_displacement == pytest.approx(1.0)
+
+
+def test_openmm_drude_force_oracle_masks_selected_source_charges():
+    omm_system = openmm.System()
+    for _ in range(3):
+        omm_system.addParticle(1.0)
+    nonbonded = openmm.NonbondedForce()
+    nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
+    nonbonded.addParticle(
+        1.0*openmm.unit.elementary_charge,
+        1.0*openmm.unit.nanometer,
+        0.0*openmm.unit.kilojoule_per_mole,
+    )
+    for _ in range(2):
+        nonbonded.addParticle(
+            -1.0*openmm.unit.elementary_charge,
+            1.0*openmm.unit.nanometer,
+            0.0*openmm.unit.kilojoule_per_mole,
+        )
+    omm_system.addForce(nonbonded)
+    context = openmm.Context(
+        omm_system,
+        openmm.VerletIntegrator(1.0*openmm.unit.femtosecond),
+        openmm.Platform.getPlatformByName("Reference"),
+    )
+
+    class Potential:
+        base_context = context
+
+        def update_positions(self, positions):
+            context.setPositions(
+                openmm.unit.Quantity(
+                    [openmm.Vec3(*row) for row in positions],
+                    openmm.unit.angstrom,
+                ),
+            )
+
+    data = DrudeData(
+        drude_indices=np.array([1, 2]),
+        parent_indices=np.array([1, 2]),
+        charges=np.array([-1.0, -1.0]),
+        polarizabilities=np.array([1.0, 1.0]),
+        force_constants=np.array([1.0, 1.0]),
+    )
+    positions = np.array(
+        [
+            [0.0, 0.0, 0.0],
+            [5.0, 0.0, 0.0],
+            [0.0, 5.0, 0.0],
+        ],
+    )
+
+    unmasked = OpenMMDrudeForceOracle(Potential(), data)(positions)
+    masked = OpenMMDrudeForceOracle(
+        Potential(),
+        data,
+        zero_charge_atoms={0, 2},
+        masked_drude_indices={1},
+    )(positions)
+    restored = OpenMMDrudeForceOracle(Potential(), data)(positions)
+
+    assert abs(unmasked[0, 0]) > 1.0
+    assert masked[0, 0] == pytest.approx(0.0)
+    assert masked[1, 1] == pytest.approx(unmasked[1, 1])
+    assert restored == pytest.approx(unmasked)
```

### Commit `798f47e`: Drude dynamics and SCF propagation

```diff
diff --git a/pydft_qmmm/__init__.py b/pydft_qmmm/__init__.py
index 86256cc..1f79d47 100644
--- a/pydft_qmmm/__init__.py
+++ b/pydft_qmmm/__init__.py
@@ -19,6 +19,8 @@ from .hamiltonians import MMHamiltonian
 from .hamiltonians import QMHamiltonian
 from .hamiltonians import QMMMHamiltonian
 from .integrators import LangevinIntegrator
+from .integrators import DrudeLangevinIntegrator
+from .integrators import DrudeSCFIntegrator
 from .integrators import VerletIntegrator
 from .system import Atom
 from .system import System
diff --git a/pydft_qmmm/integrators/__init__.py b/pydft_qmmm/integrators/__init__.py
index d10b445..92d9869 100644
--- a/pydft_qmmm/integrators/__init__.py
+++ b/pydft_qmmm/integrators/__init__.py
@@ -5,5 +5,7 @@ from __future__ import annotations
 __author__ = "John Pederson"
 
 from .integrator import *
+from .drude_langevin_integrator import *
+from .drude_scf_integrator import *
 from .langevin_integrator import *
 from .verlet_integrator import *
diff --git a/pydft_qmmm/integrators/drude_langevin_integrator.py b/pydft_qmmm/integrators/drude_langevin_integrator.py
new file mode 100644
index 0000000..aac7bb6
--- /dev/null
+++ b/pydft_qmmm/integrators/drude_langevin_integrator.py
@@ -0,0 +1,330 @@
+"""Dual-thermostat extended-Lagrangian Drude dynamics."""
+from __future__ import annotations
+
+__all__ = ["DrudeLangevinIntegrator"]
+
+from dataclasses import dataclass
+from dataclasses import field
+from typing import TYPE_CHECKING
+
+import numpy as np
+from numpy.typing import NDArray
+import openmm
+
+from pydft_qmmm.utils import KB
+from pydft_qmmm.utils import pluggable_method
+from pydft_qmmm.plugins.drude.drude_data import extract_drude_data
+from pydft_qmmm.plugins.drude.virtual_sites import VirtualSiteData
+from pydft_qmmm.plugins.drude.virtual_sites import extract_virtual_sites
+
+from .integrator import Integrator
+
+if TYPE_CHECKING:
+    from pydft_qmmm import System
+    from pydft_qmmm.calculators import Calculator
+    from .integrator import Returns
+
+
+@dataclass(frozen=True)
+class DrudeLangevinIntegrator(Integrator):
+    """Integrate Drude pairs with separate center-of-mass and relative baths.
+
+    The update follows OpenMM's ``DrudeLangevinIntegrator`` reference
+    algorithm.  Temperature is in kelvin, friction coefficients are in
+    inverse femtoseconds, and the random seed controls NumPy's generator.
+
+    Metadata is bound from the OpenMM calculator by ``Simulation``.  The
+    integrator can therefore be constructed before the Hamiltonian builds its
+    calculator, like the other pydft-qmmm integrators.
+    """
+
+    temperature: float | int
+    friction: float | int
+    drude_temperature: float | int = 1.0
+    drude_friction: float | int = 0.02
+    max_drude_distance: float | int = 0.2
+    random_seed: int | None = None
+    _drude_indices: NDArray[np.int64] | None = field(default=None, init=False)
+    _parent_indices: NDArray[np.int64] | None = field(default=None, init=False)
+    _normal_indices: NDArray[np.int64] | None = field(default=None, init=False)
+    _virtual_sites: VirtualSiteData | None = field(default=None, init=False)
+    _cm_motion_frequency: int | None = field(default=None, init=False)
+    _step_count: int = field(default=0, init=False)
+    _defer_post_step: bool = field(default=False, init=False)
+    _rng: np.random.Generator = field(init=False, repr=False, compare=False)
+
+    def __post_init__(self) -> None:
+        if self.temperature < 0 or self.drude_temperature < 0:
+            raise ValueError("Temperatures must be nonnegative.")
+        object.__setattr__(self, "_rng", np.random.default_rng(self.random_seed))
+
+    @property
+    def drude_indices(self) -> NDArray[np.int64]:
+        """Indices of Drude particles after calculator binding."""
+        self._require_bound()
+        return self._drude_indices  # type: ignore[return-value]
+
+    @property
+    def parent_indices(self) -> NDArray[np.int64]:
+        """Indices of Drude parent particles after calculator binding."""
+        self._require_bound()
+        return self._parent_indices  # type: ignore[return-value]
+
+    @property
+    def virtual_site_indices(self) -> NDArray[np.int64]:
+        """Indices of massless virtual sites after calculator binding."""
+        self._require_bound()
+        return self._virtual_sites.indices  # type: ignore[union-attr]
+
+    def bind(self, calculator: Calculator) -> None:
+        """Extract Drude and virtual-site metadata from an OpenMM calculator."""
+        try:
+            omm_system = calculator.potential.base_context.getSystem()
+            context_integrator = calculator.potential.base_context.getIntegrator()
+        except AttributeError as error:
+            raise TypeError(
+                "DrudeLangevinIntegrator requires an OpenMM-backed calculator.",
+            ) from error
+        if isinstance(context_integrator, openmm.DrudeSCFIntegrator):
+            raise RuntimeError(
+                "DrudeLangevinIntegrator requires MMHamiltonian(..., "
+                "drude_engine='native') so OpenMM does not independently "
+                "relax the Drude particles.",
+            )
+        data = extract_drude_data(omm_system)
+        cm_removers = [
+            force for force in omm_system.getForces()
+            if isinstance(force, openmm.CMMotionRemover)
+        ]
+        if len(cm_removers) > 1:
+            raise ValueError("Expected at most one OpenMM CMMotionRemover.")
+        object.__setattr__(self, "_drude_indices", data.drude_indices)
+        object.__setattr__(self, "_parent_indices", data.parent_indices)
+        paired = set(data.drude_indices.tolist()) | set(data.parent_indices.tolist())
+        normal_indices = np.asarray([
+            index for index in range(omm_system.getNumParticles())
+            if index not in paired
+            and omm_system.getParticleMass(index)/openmm.unit.dalton > 0
+        ], dtype=np.int64)
+        object.__setattr__(self, "_normal_indices", normal_indices)
+        object.__setattr__(self, "_virtual_sites", extract_virtual_sites(omm_system))
+        object.__setattr__(
+            self,
+            "_cm_motion_frequency",
+            cm_removers[0].getFrequency() if cm_removers else None,
+        )
+
+    def _require_bound(self) -> None:
+        if self._drude_indices is None or self._virtual_sites is None:
+            raise RuntimeError(
+                "DrudeLangevinIntegrator has not been bound to a calculator.",
+            )
+
+    @staticmethod
+    def _scales(
+            timestep: float,
+            friction: float,
+            temperature: float,
+    ) -> tuple[float, float, float]:
+        """Return velocity, force, and thermal-noise scales."""
+        if friction < 0:
+            raise ValueError("Friction coefficients must be nonnegative.")
+        if friction == 0:
+            return 1.0, timestep, 0.0
+        velocity_scale = np.exp(-timestep*friction)
+        force_scale = (1.0 - velocity_scale)/friction
+        # KB is J/mol/K.  1e-2 converts sqrt(kJ/mol/Da) from nm/ps
+        # to Angstrom/fs, and 1e-3 converts J to kJ.
+        noise_scale = np.sqrt(
+            KB*1e-3*temperature*(1.0 - velocity_scale**2),
+        ) * 1e-2
+        return velocity_scale, force_scale, noise_scale
+
+    def update_virtual_sites(
+            self,
+            positions: NDArray[np.float64],
+            box: NDArray[np.float64],
+    ) -> NDArray[np.float64]:
+        """Return positions with all virtual sites recomputed."""
+        self._require_bound()
+        # System.box stores lattice vectors as columns; VirtualSiteData uses
+        # OpenMM's row-vector convention.
+        return self._virtual_sites.compute_positions(positions, box.T)  # type: ignore[union-attr]
+
+    def defer_post_step(self) -> None:
+        """Defer hard-wall and virtual-site updates until after constraints."""
+        object.__setattr__(self, "_defer_post_step", True)
+
+    def finalize_positions(
+            self,
+            positions: NDArray[np.float64],
+            velocities: NDArray[np.float64],
+            masses: NDArray[np.float64],
+            box: NDArray[np.float64],
+    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
+        """Apply post-constraint hard-wall and virtual-site operations."""
+        self._apply_hard_wall(positions, velocities, masses)
+        positions = self.update_virtual_sites(positions, box)
+        return positions, velocities
+
+    def _apply_hard_wall(
+            self,
+            positions: NDArray[np.float64],
+            velocities: NDArray[np.float64],
+            masses: NDArray[np.float64],
+    ) -> None:
+        """Apply OpenMM's Drude hard-wall bounce in Angstrom/fs units."""
+        maximum = float(self.max_drude_distance)
+        if maximum <= 0:
+            return
+        thermal_speed = np.sqrt(
+            KB*1e-3*float(self.drude_temperature),
+        ) * 1e-2
+        dt = float(self.timestep)
+        for drude, parent in zip(self.drude_indices, self.parent_indices):
+            delta = positions[drude] - positions[parent]
+            distance = np.linalg.norm(delta)
+            if distance <= maximum:
+                continue
+            if distance > 2*maximum:
+                raise RuntimeError(
+                    "Drude particle moved too far beyond hard wall constraint.",
+                )
+            direction = delta/distance
+            v1 = velocities[drude].copy()
+            v2 = velocities[parent].copy()
+            m1, m2 = masses[drude], masses[parent]
+            excess = distance - maximum
+            radial1 = float(v1 @ direction)
+            transverse1 = v1 - radial1*direction
+            if m2 == 0:
+                crossing_time = min(
+                    dt if radial1 == 0 else excess/abs(radial1), dt,
+                )
+                if radial1 != 0:
+                    radial1 = -np.copysign(thermal_speed/np.sqrt(m1), radial1)
+                positions[drude] += direction*(-excess + crossing_time*radial1)
+                velocities[drude] = transverse1 + direction*radial1
+                continue
+            inv_total = 1.0/(m1 + m2)
+            radial2 = float(v2 @ direction)
+            transverse2 = v2 - radial2*direction
+            center_velocity = (m1*radial1 + m2*radial2)*inv_total
+            relative1 = radial1 - center_velocity
+            relative2 = radial2 - center_velocity
+            crossing_time = min(
+                dt if relative1 == relative2
+                else excess/abs(relative1 - relative2),
+                dt,
+            )
+            bond_speed = thermal_speed/np.sqrt(m1)
+            if relative1 != 0:
+                relative1 = -np.copysign(
+                    bond_speed*m2*inv_total, relative1,
+                )
+            if relative2 != 0:
+                relative2 = -np.copysign(
+                    bond_speed*m1*inv_total, relative2,
+                )
+            positions[drude] += direction*(
+                -excess*m2*inv_total + crossing_time*relative1
+            )
+            positions[parent] += direction*(
+                excess*m1*inv_total + crossing_time*relative2
+            )
+            velocities[drude] = transverse1 + direction*(relative1 + center_velocity)
+            velocities[parent] = transverse2 + direction*(relative2 + center_velocity)
+
+    @pluggable_method
+    def integrate(self, system: System) -> Returns:
+        """Advance one dual-Langevin Drude dynamics step."""
+        self._require_bound()
+        masses = np.asarray(system.masses).reshape((-1, 1))
+        positions = np.array(system.positions, copy=True)
+        velocities = np.array(system.velocities, copy=True)
+        forces = np.asarray(system.forces)
+        inverse_masses = np.zeros_like(masses)
+        np.divide(1.0, masses, out=inverse_masses, where=masses != 0)
+
+        if (
+                self._cm_motion_frequency is not None
+                and self._step_count % self._cm_motion_frequency == 0
+        ):
+            massive = masses[:, 0] > 0
+            center_velocity = np.sum(
+                masses[massive]*velocities[massive], axis=0,
+            )/np.sum(masses[massive])
+            velocities[massive] -= center_velocity
+
+        normal = self._normal_indices
+        vscale, fscale, noise = self._scales(
+            float(self.timestep), float(self.friction), float(self.temperature),
+        )
+        if normal is not None and normal.size:
+            velocities[normal] = (
+                vscale*velocities[normal]
+                + fscale*inverse_masses[normal]*forces[normal]*1e-4
+                + noise*np.sqrt(inverse_masses[normal])
+                * self._rng.standard_normal((len(normal), 3))
+            )
+
+        dvscale, dfscale, dnoise = self._scales(
+            float(self.timestep),
+            float(self.drude_friction),
+            float(self.drude_temperature),
+        )
+        for drude, parent in zip(self.drude_indices, self.parent_indices):
+            m1 = masses[drude, 0]
+            m2 = masses[parent, 0]
+            if m1 <= 0 or m2 <= 0:
+                raise ValueError("Drude particles and parents must have positive mass.")
+            inv_total = 1.0/(m1 + m2)
+            inv_reduced = (m1 + m2)/(m1*m2)
+            mass1_fraction = m1*inv_total
+            mass2_fraction = m2*inv_total
+            center_velocity = (
+                mass1_fraction*velocities[drude]
+                + mass2_fraction*velocities[parent]
+            )
+            relative_velocity = velocities[parent] - velocities[drude]
+            center_force = forces[drude] + forces[parent]
+            relative_force = (
+                mass1_fraction*forces[parent]
+                - mass2_fraction*forces[drude]
+            )
+            center_velocity = (
+                vscale*center_velocity
+                + fscale*inv_total*center_force*1e-4
+                + noise*np.sqrt(inv_total)*self._rng.standard_normal(3)
+            )
+            relative_velocity = (
+                dvscale*relative_velocity
+                + dfscale*inv_reduced*relative_force*1e-4
+                + dnoise*np.sqrt(inv_reduced)*self._rng.standard_normal(3)
+            )
+            velocities[drude] = center_velocity - mass2_fraction*relative_velocity
+            velocities[parent] = center_velocity + mass1_fraction*relative_velocity
+
+        massive = masses[:, 0] > 0
+        positions[massive] += float(self.timestep)*velocities[massive]
+        if not self._defer_post_step:
+            positions, velocities = self.finalize_positions(
+                positions,
+                velocities,
+                masses[:, 0],
+                np.asarray(system.box),
+            )
+        object.__setattr__(self, "_step_count", self._step_count + 1)
+        return positions, velocities
+
+    @pluggable_method
+    def compute_kinetic_energy(self, system: System) -> float:
+        """Compute leapfrog kinetic energy while excluding virtual sites."""
+        masses = np.asarray(system.masses).reshape((-1, 1))
+        massive = masses[:, 0] > 0
+        velocities = np.array(system.velocities, copy=True)
+        velocities[massive] += (
+            0.5*float(self.timestep)*np.asarray(system.forces)[massive]
+            * 1e-4/masses[massive]
+        )
+        return float(np.sum(0.5*masses[massive]*velocities[massive]**2)*1e4)
diff --git a/pydft_qmmm/integrators/drude_scf_integrator.py b/pydft_qmmm/integrators/drude_scf_integrator.py
new file mode 100644
index 0000000..6e3e1e9
--- /dev/null
+++ b/pydft_qmmm/integrators/drude_scf_integrator.py
@@ -0,0 +1,177 @@
+"""Nuclear equations of motion for self-consistent Drude dynamics."""
+from __future__ import annotations
+
+__all__ = ["DrudeSCFIntegrator"]
+
+from typing import TYPE_CHECKING
+
+import numpy as np
+from numpy.typing import NDArray
+import openmm
+
+from pydft_qmmm.plugins.drude.drude_data import extract_drude_data
+from pydft_qmmm.plugins.drude.virtual_sites import VirtualSiteData
+from pydft_qmmm.plugins.drude.virtual_sites import extract_virtual_sites
+from pydft_qmmm.utils import pluggable_method
+
+from .integrator import Integrator
+
+if TYPE_CHECKING:
+    from typing import Any
+    from pydft_qmmm import System
+    from pydft_qmmm.calculators import Calculator
+    from .integrator import Returns
+
+
+class _PropagationSystem:
+    """Array-isolated system view used by a wrapped base integrator."""
+
+    def __init__(self, system: System, excluded: NDArray[np.int64]) -> None:
+        self._system = system
+        self.positions = np.array(system.positions, copy=True)
+        self.velocities = np.array(system.velocities, copy=True)
+        self.forces = np.array(system.forces, copy=True)
+        self.masses = np.array(system.masses, copy=True)
+        self.velocities[excluded] = 0.0
+        self.forces[excluded] = 0.0
+        # Generic pydft-qmmm integrators divide by every mass.  Give massless
+        # virtual sites an inert placeholder mass in this isolated view.
+        self.masses[self.masses == 0] = 1.0
+
+    def __getattr__(self, name: str) -> Any:
+        return getattr(self._system, name)
+
+
+class DrudeSCFIntegrator(Integrator):
+    """Propagate nuclei while leaving Drude relaxation to the calculator.
+
+    This wrapper deliberately separates the two operations in SCF dynamics:
+
+    1. ``integrate()`` advances only physical nuclei with ``base_integrator``.
+    2. The subsequent calculator call relaxes Drudes, either with the native
+       :class:`~pydft_qmmm.plugins.drude.DrudeSCF` plugin for MM or with a
+       QM/MM plugin that interleaves Drude updates with the electronic SCF.
+
+    Drude particles and virtual sites are excluded from EOM propagation and
+    kinetic energy.  Virtual-site coordinates are reconstructed after
+    constraints; Drude coordinates are not modified by this class.
+    """
+
+    def __init__(self, base_integrator: Integrator) -> None:
+        object.__setattr__(self, "timestep", base_integrator.timestep)
+        object.__setattr__(self, "_plugins", [])
+        object.__setattr__(self, "base_integrator", base_integrator)
+        object.__setattr__(self, "_drude_indices", None)
+        object.__setattr__(self, "_virtual_sites", None)
+        object.__setattr__(self, "_defer_post_step", False)
+
+    @property
+    def drude_indices(self) -> NDArray[np.int64]:
+        """Indices of SCF-relaxed Drude particles."""
+        self._require_bound()
+        return self._drude_indices
+
+    @property
+    def virtual_site_indices(self) -> NDArray[np.int64]:
+        """Indices of derived massless particles."""
+        self._require_bound()
+        return self._virtual_sites.indices
+
+    @property
+    def kinetic_exclusion_indices(self) -> NDArray[np.int64]:
+        """Particles excluded from physical nuclear kinetic energy."""
+        return np.union1d(self.drude_indices, self.virtual_site_indices)
+
+    def _find_openmm_system(
+            self,
+            calculator: Calculator,
+    ) -> tuple[openmm.System, openmm.Integrator]:
+        """Locate an OpenMM potential in a simple or composite calculator."""
+        calculators = getattr(calculator, "calculators", (calculator,))
+        for component in calculators:
+            potential = getattr(component, "potential", None)
+            context = getattr(potential, "base_context", None)
+            if context is not None:
+                omm_system = context.getSystem()
+                if any(
+                    isinstance(force, openmm.DrudeForce)
+                    for force in omm_system.getForces()
+                ):
+                    return omm_system, context.getIntegrator()
+        raise TypeError(
+            "DrudeSCFIntegrator requires an OpenMM Drude potential in the "
+            "calculator.",
+        )
+
+    def bind(self, calculator: Calculator) -> None:
+        """Bind Drude and virtual-site topology from the MM calculator."""
+        omm_system, context_integrator = self._find_openmm_system(calculator)
+        if isinstance(context_integrator, openmm.DrudeSCFIntegrator):
+            raise RuntimeError(
+                "DrudeSCFIntegrator requires MMHamiltonian(..., "
+                "drude_engine='native') so relaxation is owned by the "
+                "pydft-qmmm/QM-SCF plugin.",
+            )
+        data = extract_drude_data(omm_system)
+        object.__setattr__(self, "_drude_indices", data.drude_indices)
+        object.__setattr__(self, "_virtual_sites", extract_virtual_sites(omm_system))
+
+    def _require_bound(self) -> None:
+        if self._drude_indices is None or self._virtual_sites is None:
+            raise RuntimeError(
+                "DrudeSCFIntegrator has not been bound to a calculator.",
+            )
+
+    def defer_post_step(self) -> None:
+        """Let a constraint plugin perform virtual-site reconstruction."""
+        object.__setattr__(self, "_defer_post_step", True)
+
+    def update_virtual_sites(
+            self,
+            positions: NDArray[np.float64],
+            box: NDArray[np.float64],
+    ) -> NDArray[np.float64]:
+        """Reconstruct virtual-site coordinates without relaxing Drudes."""
+        self._require_bound()
+        return self._virtual_sites.compute_positions(positions, box.T)
+
+    def finalize_positions(
+            self,
+            positions: NDArray[np.float64],
+            velocities: NDArray[np.float64],
+            masses: NDArray[np.float64],
+            box: NDArray[np.float64],
+    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
+        """Perform post-constraint virtual-site reconstruction only."""
+        del masses
+        return self.update_virtual_sites(positions, box), velocities
+
+    @pluggable_method
+    def integrate(self, system: System) -> Returns:
+        """Advance physical nuclei by one step and freeze auxiliary sites."""
+        excluded = self.kinetic_exclusion_indices
+        propagation_system = _PropagationSystem(system, excluded)
+        positions, velocities = self.base_integrator.integrate(
+            propagation_system,
+        )
+        positions = np.asarray(positions)
+        velocities = np.asarray(velocities)
+        positions[excluded] = np.asarray(system.positions)[excluded]
+        velocities[excluded] = 0.0
+        if not self._defer_post_step:
+            positions, velocities = self.finalize_positions(
+                positions,
+                velocities,
+                np.asarray(system.masses),
+                np.asarray(system.box),
+            )
+        return positions, velocities
+
+    @pluggable_method
+    def compute_kinetic_energy(self, system: System) -> float:
+        """Compute physical nuclear kinetic energy with the base integrator."""
+        propagation_system = _PropagationSystem(
+            system,
+            self.kinetic_exclusion_indices,
+        )
+        return self.base_integrator.compute_kinetic_energy(propagation_system)
diff --git a/pydft_qmmm/plugins/drude/__init__.py b/pydft_qmmm/plugins/drude/__init__.py
index 8420cd8..e2864ff 100644
--- a/pydft_qmmm/plugins/drude/__init__.py
+++ b/pydft_qmmm/plugins/drude/__init__.py
@@ -9,6 +9,8 @@ __all__ = [
     "OpenMMDrudeForceOracle",
     "drude_relaxation_step",
     "extract_drude_data",
+    "extract_virtual_sites",
+    "VirtualSiteData",
 ]
 
 from .drude_data import DrudeData
@@ -18,3 +20,5 @@ from .drude_solver import DrudeSolver
 from .drude_solver import DrudeStepInfo
 from .drude_solver import drude_relaxation_step
 from .openmm_oracle import OpenMMDrudeForceOracle
+from .virtual_sites import VirtualSiteData
+from .virtual_sites import extract_virtual_sites
diff --git a/pydft_qmmm/plugins/drude/drude_scf.py b/pydft_qmmm/plugins/drude/drude_scf.py
index 79e944e..9c130b9 100644
--- a/pydft_qmmm/plugins/drude/drude_scf.py
+++ b/pydft_qmmm/plugins/drude/drude_scf.py
@@ -29,6 +29,8 @@ class DrudeSCF(CalculatorPlugin):
             one iteration required for convergence, in Angstrom.
         max_iterations: Maximum fixed-point iterations.
         damping: Scalar multiplier applied to each diagonal-Newton step.
+        stagnation_ratio: Optional OpenMM-compatible force-stagnation
+            threshold.  Use 0.9 to match OpenMM trajectory behavior.
     """
 
     def __init__(
@@ -37,11 +39,13 @@ class DrudeSCF(CalculatorPlugin):
             displacement_tolerance: float = 1e-6,
             max_iterations: int = 100,
             damping: float = 1.0,
+            stagnation_ratio: float | None = None,
     ) -> None:
         self.force_tolerance = force_tolerance
         self.displacement_tolerance = displacement_tolerance
         self.max_iterations = max_iterations
         self.damping = damping
+        self.stagnation_ratio = stagnation_ratio
         self._solver: DrudeSolver | None = None
         self.last_info: DrudeSCFInfo | None = None
 
@@ -66,6 +70,7 @@ class DrudeSCF(CalculatorPlugin):
             displacement_tolerance=self.displacement_tolerance,
             max_iterations=self.max_iterations,
             damping=self.damping,
+            stagnation_ratio=self.stagnation_ratio,
         )
         return self._solver
 
diff --git a/pydft_qmmm/plugins/drude/drude_solver.py b/pydft_qmmm/plugins/drude/drude_solver.py
index b9fb6d8..cf62d8a 100644
--- a/pydft_qmmm/plugins/drude/drude_solver.py
+++ b/pydft_qmmm/plugins/drude/drude_solver.py
@@ -80,6 +80,9 @@ class DrudeSolver:
             one iteration required for convergence, in Angstrom.
         max_iterations: Maximum fixed-point iterations.
         damping: Scalar multiplier applied to each diagonal-Newton step.
+        stagnation_ratio: If provided, accept the current step when the sum
+            of squared Drude forces exceeds this fraction of its value in the
+            preceding iteration.  OpenMM's Drude-SCF minimizer uses 0.9.
     """
 
     def __init__(
@@ -91,6 +94,7 @@ class DrudeSolver:
             displacement_tolerance: float = 1e-6,
             max_iterations: int = 100,
             damping: float = 1.0,
+            stagnation_ratio: float | None = None,
     ) -> None:
         self.data = data
         self.force_oracle = force_oracle
@@ -98,6 +102,7 @@ class DrudeSolver:
         self.displacement_tolerance = displacement_tolerance
         self.max_iterations = max_iterations
         self.damping = damping
+        self.stagnation_ratio = stagnation_ratio
 
     def step(
             self,
@@ -136,8 +141,15 @@ class DrudeSolver:
         relaxed = np.array(positions, dtype=float, copy=True)
         final_max_force = np.inf
         final_max_displacement = np.inf
+        previous_force_squared = np.inf
         for iteration in range(1, self.max_iterations + 1):
-            stepped, step_info = self.step(relaxed)
+            forces = self.force_oracle(relaxed)
+            stepped, step_info = drude_relaxation_step(
+                self.data,
+                relaxed,
+                forces,
+                damping=self.damping,
+            )
             final_max_force = step_info.max_force
             final_max_displacement = step_info.max_displacement
             if (
@@ -150,6 +162,20 @@ class DrudeSolver:
                     final_max_displacement=final_max_displacement,
                     converged=True,
                 )
+            force_squared = float(np.sum(forces*forces))
+            if (
+                    self.stagnation_ratio is not None
+                    and iteration > 1
+                    and force_squared
+                    > self.stagnation_ratio*previous_force_squared
+            ):
+                return stepped, DrudeSCFInfo(
+                    iterations=iteration,
+                    final_max_force=final_max_force,
+                    final_max_displacement=final_max_displacement,
+                    converged=True,
+                )
+            previous_force_squared = force_squared
             relaxed = stepped
         raise RuntimeError(
             "Drude SCF did not converge after "
diff --git a/pydft_qmmm/plugins/drude/virtual_sites.py b/pydft_qmmm/plugins/drude/virtual_sites.py
new file mode 100644
index 0000000..7e68f8b
--- /dev/null
+++ b/pydft_qmmm/plugins/drude/virtual_sites.py
@@ -0,0 +1,186 @@
+"""OpenMM-compatible virtual-site position construction."""
+from __future__ import annotations
+
+__all__ = ["VirtualSiteData", "extract_virtual_sites"]
+
+from dataclasses import dataclass
+from typing import Any
+
+import numpy as np
+from numpy.typing import NDArray
+import openmm
+import openmm.unit
+
+
+@dataclass(frozen=True)
+class _VirtualSite:
+    """A single virtual-site definition in dependency order."""
+
+    index: int
+    kind: str
+    particles: tuple[int, ...]
+    parameters: tuple[Any, ...]
+
+
+@dataclass(frozen=True)
+class VirtualSiteData:
+    """Virtual sites extracted from an OpenMM System."""
+
+    sites: tuple[_VirtualSite, ...]
+
+    @property
+    def indices(self) -> NDArray[np.int64]:
+        """Return virtual-site particle indices."""
+        return np.asarray([site.index for site in self.sites], dtype=np.int64)
+
+    def compute_positions(
+            self,
+            positions: NDArray[np.float64],
+            box: NDArray[np.float64] | None = None,
+    ) -> NDArray[np.float64]:
+        """Compute virtual-site coordinates in dependency order.
+
+        Args:
+            positions: Particle positions in Angstrom.
+            box: Box vectors in Angstrom, stored as row vectors.  This is
+                required only by a periodic ``SymmetrySite``.
+        """
+        result = np.array(positions, dtype=float, copy=True)
+        for site in self.sites:
+            p = site.particles
+            if site.kind in {"two_average", "three_average"}:
+                weights = np.asarray(site.parameters[0])
+                result[site.index] = weights @ result[list(p)]
+            elif site.kind == "out_of_plane":
+                w12, w13, wcross = site.parameters
+                v12 = result[p[1]] - result[p[0]]
+                v13 = result[p[2]] - result[p[0]]
+                result[site.index] = (
+                    result[p[0]] + w12*v12 + w13*v13
+                    + wcross*np.cross(v12, v13)
+                )
+            elif site.kind == "local_coordinates":
+                origin_weights, x_weights, y_weights, local_position = (
+                    site.parameters
+                )
+                source = result[list(p)]
+                origin = np.asarray(origin_weights) @ source
+                xdir = np.asarray(x_weights) @ source
+                ydir = np.asarray(y_weights) @ source
+                zdir = np.cross(xdir, ydir)
+                norm_x = np.linalg.norm(xdir)
+                norm_z = np.linalg.norm(zdir)
+                if norm_x > 0:
+                    xdir /= norm_x
+                if norm_z > 0:
+                    zdir /= norm_z
+                ydir = np.cross(zdir, xdir)
+                local = np.asarray(local_position)
+                result[site.index] = (
+                    origin + local[0]*xdir + local[1]*ydir + local[2]*zdir
+                )
+            elif site.kind == "symmetry":
+                rotation, offset, use_box = site.parameters
+                coordinate = result[p[0]]
+                if use_box:
+                    if box is None or not np.any(box):
+                        raise ValueError(
+                            "A periodic SymmetrySite requires box vectors.",
+                        )
+                    coordinate = coordinate @ np.linalg.inv(box)
+                coordinate = np.asarray(rotation) @ coordinate + offset
+                if use_box:
+                    coordinate = coordinate @ box
+                result[site.index] = coordinate
+            else:  # pragma: no cover - guarded by extraction
+                raise TypeError(f"Unsupported virtual-site type: {site.kind}")
+        return result
+
+
+def _dependency_order(system: openmm.System) -> list[int]:
+    """Topologically order virtual sites like OpenMM ReferenceVirtualSites."""
+    remaining = {
+        i for i in range(system.getNumParticles()) if system.isVirtualSite(i)
+    }
+    order = []
+    while remaining:
+        previous_size = len(remaining)
+        for index in sorted(tuple(remaining)):
+            site = system.getVirtualSite(index)
+            dependencies = {site.getParticle(i) for i in range(site.getNumParticles())}
+            if dependencies.isdisjoint(remaining):
+                order.append(index)
+                remaining.remove(index)
+        if len(remaining) == previous_size:
+            raise ValueError("Virtual site definitions are circular.")
+    return order
+
+
+def _vec3(value: Any, unit: Any | None = None) -> tuple[float, float, float]:
+    """Convert an OpenMM Vec3 or Quantity<Vec3> to a float tuple."""
+    if unit is not None:
+        value = value.value_in_unit(unit)
+    return tuple(float(value[i]) for i in range(3))
+
+
+def extract_virtual_sites(system: openmm.System) -> VirtualSiteData:
+    """Extract supported virtual sites from an OpenMM System."""
+    sites = []
+    for index in _dependency_order(system):
+        site = system.getVirtualSite(index)
+        particles = tuple(
+            int(site.getParticle(i)) for i in range(site.getNumParticles())
+        )
+        if isinstance(site, openmm.TwoParticleAverageSite):
+            sites.append(_VirtualSite(
+                index, "two_average", particles,
+                (tuple(float(site.getWeight(i)) for i in range(2)),),
+            ))
+        elif isinstance(site, openmm.ThreeParticleAverageSite):
+            sites.append(_VirtualSite(
+                index, "three_average", particles,
+                (tuple(float(site.getWeight(i)) for i in range(3)),),
+            ))
+        elif isinstance(site, openmm.OutOfPlaneSite):
+            sites.append(_VirtualSite(
+                index, "out_of_plane", particles,
+                (
+                    float(site.getWeight12()),
+                    float(site.getWeight13()),
+                    float(site.getWeightCross()),
+                ),
+            ))
+        elif isinstance(site, openmm.LocalCoordinatesSite):
+            sites.append(_VirtualSite(
+                index, "local_coordinates", particles,
+                (
+                    tuple(float(x) for x in site.getOriginWeights()),
+                    tuple(float(x) for x in site.getXWeights()),
+                    tuple(float(x) for x in site.getYWeights()),
+                    _vec3(site.getLocalPosition(), openmm.unit.angstrom),
+                ),
+            ))
+        elif hasattr(openmm, "SymmetrySite") and isinstance(
+                site, openmm.SymmetrySite,
+        ):
+            rotation = np.asarray([
+                _vec3(site.getRotationMatrix()[i]) for i in range(3)
+            ])
+            use_box = bool(site.getUseBoxVectors())
+            offset = np.asarray(_vec3(site.getOffsetVector()))
+            if not use_box:
+                # SymmetrySite stores Cartesian offsets in OpenMM's native nm.
+                offset *= 10.0
+            sites.append(_VirtualSite(
+                index, "symmetry", particles,
+                (
+                    rotation,
+                    offset,
+                    use_box,
+                ),
+            ))
+        else:
+            raise TypeError(
+                f"Unsupported OpenMM virtual-site type: {type(site).__name__}",
+            )
+    return VirtualSiteData(tuple(sites))
diff --git a/pydft_qmmm/plugins/settle/settle.py b/pydft_qmmm/plugins/settle/settle.py
index 7abd904..8f7813b 100644
--- a/pydft_qmmm/plugins/settle/settle.py
+++ b/pydft_qmmm/plugins/settle/settle.py
@@ -5,7 +5,6 @@ from __future__ import annotations
 __all__ = ["SETTLE"]
 
 from collections.abc import Callable
-from functools import lru_cache
 from typing import TYPE_CHECKING
 
 import numpy as np
@@ -16,6 +15,7 @@ from .settle_utils import settle_velocities
 from pydft_qmmm.integrators import IntegratorPlugin
 
 if TYPE_CHECKING:
+    from pydft_qmmm.integrators import Integrator
     from pydft_qmmm.integrators import Returns
     from pydft_qmmm import System
 
@@ -44,6 +44,14 @@ class SETTLE(IntegratorPlugin):
         self.query = "(" + query + ") and not subsystem I"
         self.oh_distance = oh_distance
         self.hh_distance = hh_distance
+        self._residue_cache: tuple[int, list[list[int]]] | None = None
+
+    def modify(self, integrator: Integrator) -> None:
+        """Register SETTLE and defer Drude post-step operations."""
+        super().modify(integrator)
+        defer_post_step = getattr(integrator, "defer_post_step", None)
+        if defer_post_step is not None:
+            defer_post_step()
 
     def constrain_velocities(self, system: System) -> NDArray[np.float64]:
         """Apply the SETTLE algorithm to system velocities.
@@ -64,33 +72,47 @@ class SETTLE(IntegratorPlugin):
         )
         return velocities
 
-    @lru_cache
     def _get_hoh_residues(
             self,
-            residues: tuple[int, ...],
-            residue_set: frozenset[tuple[int, frozenset[int]]],
-            select: Callable[[str], frozenset[int]],
+            system: System,
     ) -> list[list[int]]:
         """Get the water residues from the system.
 
         Args:
-            residues: The indices of the residue to which the atoms
-                of the system belong.
-            residue_set: The residue index and the corresponding sets of
-                atoms.
-            select: The select method of the system.
+            system: The system containing selected water residues.
 
         Returns:
             A list of list of atom indices, representing the all water
             residues in the system.
         """
+        if self._residue_cache is not None:
+            system_id, residues = self._residue_cache
+            if system_id == id(system):
+                return residues
         residue_indices = np.unique(
-            np.array(residues)[sorted(select(self.query))],
+            np.asarray(system.residues)[sorted(system.select(self.query))],
+        )
+        residue_map = system.residue_map
+        drudes = frozenset(getattr(self.integrator, "drude_indices", ()))
+        virtual_sites = frozenset(
+            getattr(self.integrator, "virtual_site_indices", ()),
         )
-        residue_map = dict(residue_set)
-        hoh_residues = [sorted(residue_map[i]) for i in residue_indices]
-        if any([len(residue) != 3 for residue in hoh_residues]):
-            raise ValueError("Some SETTLE residues do not have 3 atoms")
+        excluded = drudes | virtual_sites
+        hoh_residues = []
+        for residue_index in residue_indices:
+            atoms = [
+                atom for atom in residue_map[residue_index]
+                if atom not in excluded
+            ]
+            if len(atoms) != 3:
+                raise ValueError(
+                    "SETTLE residues must contain exactly three non-Drude, "
+                    "non-virtual-site atoms.",
+                )
+            # settle_positions expects the central, heavy oxygen first.
+            oxygen = max(atoms, key=lambda atom: system.masses[atom])
+            hoh_residues.append([oxygen] + sorted(set(atoms) - {oxygen}))
+        self._residue_cache = (id(system), hoh_residues)
         return hoh_residues
 
     def _modify_integrate(
@@ -108,11 +130,7 @@ class SETTLE(IntegratorPlugin):
         """
         def inner(system: System) -> Returns:
             positions, velocities = integrate(system)
-            residues = self._get_hoh_residues(
-                tuple(system.residues),
-                frozenset(system.residue_map.items()),
-                system.select,
-            )
+            residues = self._get_hoh_residues(system)
             if residues:
                 positions = settle_positions(
                     residues,
@@ -128,6 +146,16 @@ class SETTLE(IntegratorPlugin):
                         - system.positions[residues, :]
                     ) / self.integrator.timestep
                 )
+            finalize_positions = getattr(
+                self.integrator, "finalize_positions", None,
+            )
+            if finalize_positions is not None:
+                positions, velocities = finalize_positions(
+                    positions,
+                    velocities,
+                    np.asarray(system.masses),
+                    np.asarray(system.box),
+                )
             return positions, velocities
         return inner
 
@@ -147,18 +175,23 @@ class SETTLE(IntegratorPlugin):
         """
         def inner(system: System) -> float:
             masses = system.masses.reshape(-1, 1)
+            accelerations = np.zeros_like(system.forces)
+            np.divide(
+                system.forces*(10**-4),
+                masses,
+                out=accelerations,
+                where=masses != 0,
+            )
             velocities = (
                 system.velocities
-                + (
-                    0.5*self.integrator.timestep
-                    * system.forces*(10**-4)/masses
-                )
+                + 0.5*self.integrator.timestep*accelerations
             )
-            residues = self._get_hoh_residues(
-                tuple(system.residues),
-                frozenset(system.residue_map.items()),
-                system.select,
+            excluded = getattr(
+                self.integrator, "kinetic_exclusion_indices", (),
             )
+            if len(excluded):
+                velocities[excluded] = 0.0
+            residues = self._get_hoh_residues(system)
             if residues:
                 velocities = settle_velocities(
                     residues,
diff --git a/pydft_qmmm/wrappers/simulation.py b/pydft_qmmm/wrappers/simulation.py
index b3e62e1..c980bc4 100644
--- a/pydft_qmmm/wrappers/simulation.py
+++ b/pydft_qmmm/wrappers/simulation.py
@@ -71,15 +71,25 @@ class Simulation(Loggable):
             self.calculator = hamiltonian.build_calculator(system)
         else:
             raise TypeError
+        bind = getattr(integrator, "bind", None)
+        if bind is not None:
+            bind(self.calculator)
         # Perform additional simulation setup.
         self._offset = np.zeros(system.positions.shape)
         if system.box.any():
             self.calculator.register_plugin(CalculatorWrap(), 0)
         if system.select("subsystem I"):
             self.calculator.register_plugin(CalculatorCenter(), 0)
-        if system.masses[system.masses == 0].size > 0:
+        virtual_sites = frozenset(
+            getattr(integrator, "virtual_site_indices", ()),
+        )
+        stationary_atoms = [
+            int(atom) for atom in np.where(system.masses.base == 0)[0]
+            if atom not in virtual_sites
+        ]
+        if stationary_atoms:
             query = "atom"
-            for atom in np.where(system.masses.base == 0)[0]:
+            for atom in stationary_atoms:
                 query += f" {atom}"
                 system.masses[atom] = ELEMENT_TO_MASS.get(
                     system.elements[atom],
diff --git a/tests/drude_test.py b/tests/drude_test.py
index b21e3b2..a77b393 100644
--- a/tests/drude_test.py
+++ b/tests/drude_test.py
@@ -10,6 +10,13 @@ from pydft_qmmm.plugins.drude import DrudeData
 from pydft_qmmm.plugins.drude import DrudeSolver
 from pydft_qmmm.plugins.drude import drude_relaxation_step
 from pydft_qmmm.plugins.drude import extract_drude_data
+from pydft_qmmm.plugins.drude import extract_virtual_sites
+from pydft_qmmm.integrators import DrudeLangevinIntegrator
+from pydft_qmmm.integrators import DrudeSCFIntegrator
+from pydft_qmmm.integrators import VerletIntegrator
+from pydft_qmmm import Atom
+from pydft_qmmm import System
+from pydft_qmmm.plugins import SETTLE
 
 
 def test_extract_drude_data():
@@ -66,6 +73,33 @@ def test_drude_solver_relaxes_against_force_oracle():
     assert info.converged
 
 
+def test_drude_solver_can_accept_openmm_style_stagnation():
+    data = DrudeData(
+        drude_indices=np.array([0]),
+        parent_indices=np.array([1]),
+        charges=np.array([-1.0]),
+        polarizabilities=np.array([1.0]),
+        force_constants=np.array([10.0]),
+    )
+
+    def stalled_oracle(positions):
+        return np.array([[1.0, 0.0, 0.0]])
+
+    solver = DrudeSolver(
+        data,
+        stalled_oracle,
+        force_tolerance=1e-12,
+        displacement_tolerance=1e-12,
+        max_iterations=10,
+        stagnation_ratio=0.9,
+    )
+    positions, info = solver.relax(np.zeros((2, 3)))
+
+    assert info.converged
+    assert info.iterations == 2
+    assert positions[0, 0] == pytest.approx(2.0)
+
+
 def test_drude_relaxation_step_is_standalone():
     data = DrudeData(
         drude_indices=np.array([0]),
@@ -148,3 +182,284 @@ def test_openmm_drude_force_oracle_masks_selected_source_charges():
     assert masked[0, 0] == pytest.approx(0.0)
     assert masked[1, 1] == pytest.approx(unmasked[1, 1])
     assert restored == pytest.approx(unmasked)
+
+
+def test_virtual_site_positions_match_openmm():
+    system = openmm.System()
+    for mass in (1.0, 1.0, 0.0, 0.0):
+        system.addParticle(mass)
+    system.setVirtualSite(2, openmm.TwoParticleAverageSite(0, 1, 0.25, 0.75))
+    system.setVirtualSite(
+        3,
+        openmm.OutOfPlaneSite(0, 1, 2, 0.2, 0.3, 0.1),
+    )
+    positions = np.array([
+        [0.1, 0.2, 0.3],
+        [1.1, -0.4, 0.7],
+        [9.0, 9.0, 9.0],
+        [8.0, 8.0, 8.0],
+    ])
+    context = openmm.Context(system, openmm.VerletIntegrator(0.001))
+    context.setPositions(positions*openmm.unit.angstrom)
+    context.computeVirtualSites()
+    expected = context.getState(getPositions=True).getPositions(asNumpy=True)
+    expected = np.asarray(expected.value_in_unit(openmm.unit.angstrom))
+    actual = extract_virtual_sites(system).compute_positions(positions)
+    assert np.allclose(actual, expected, atol=1e-13)
+
+
+def test_local_coordinate_virtual_site_matches_openmm():
+    system = openmm.System()
+    for mass in (1.0, 1.0, 1.0, 0.0):
+        system.addParticle(mass)
+    system.setVirtualSite(
+        3,
+        openmm.LocalCoordinatesSite(
+            [0, 1, 2],
+            [1.0, 0.0, 0.0],
+            [-1.0, 1.0, 0.0],
+            [-1.0, 0.0, 1.0],
+            openmm.Vec3(0.4, 0.3, 0.2),
+        ),
+    )
+    positions = np.array([
+        [0.1, 0.2, 0.3],
+        [1.1, -0.4, 0.7],
+        [0.4, 1.2, -0.2],
+        [9.0, 9.0, 9.0],
+    ])
+    context = openmm.Context(system, openmm.VerletIntegrator(0.001))
+    context.setPositions(positions*openmm.unit.angstrom)
+    context.computeVirtualSites()
+    expected = context.getState(getPositions=True).getPositions(asNumpy=True)
+    expected = np.asarray(expected.value_in_unit(openmm.unit.angstrom))
+    actual = extract_virtual_sites(system).compute_positions(positions)
+    assert np.allclose(actual, expected, atol=1e-13)
+
+
+def test_drude_langevin_deterministic_step_matches_openmm():
+    omm_system = openmm.System()
+    drude = omm_system.addParticle(0.4)
+    parent = omm_system.addParticle(15.6)
+    normal = omm_system.addParticle(1.0)
+    drude_force = openmm.DrudeForce()
+    drude_force.addParticle(drude, parent, -1, -1, -1, -1.0, 0.001, 1.0, 1.0)
+    omm_system.addForce(drude_force)
+    external = openmm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
+    external.addGlobalParameter("k", 10.0)
+    for index in range(3):
+        external.addParticle(index, [])
+    omm_system.addForce(external)
+    omm_system.addForce(openmm.CMMotionRemover(1))
+
+    positions = np.array([
+        [0.02, 0.00, 0.00],
+        [0.00, 0.00, 0.00],
+        [0.10, 0.20, -0.10],
+    ])
+    velocities = np.array([
+        [0.01, -0.02, 0.03],
+        [-0.01, 0.01, 0.00],
+        [0.04, 0.02, -0.03],
+    ])
+    reference_integrator = openmm.DrudeLangevinIntegrator(
+        0.0, 1.0, 0.0, 20.0, 0.001,
+    )
+    reference_integrator.setMaxDrudeDistance(0.0)
+    context = openmm.Context(omm_system, reference_integrator)
+    context.setPositions(positions*openmm.unit.nanometer)
+    context.setVelocities(
+        velocities*openmm.unit.nanometer/openmm.unit.picosecond,
+    )
+    initial = context.getState(getForces=True)
+    forces = np.asarray(initial.getForces(asNumpy=True).value_in_unit(
+        openmm.unit.kilojoule_per_mole/openmm.unit.nanometer,
+    ))
+    reference_integrator.step(1)
+    expected = context.getState(getPositions=True, getVelocities=True)
+
+    atoms = [
+        Atom(
+            position=position*10,
+            velocity=velocity*0.01,
+            force=force/10,
+            mass=mass,
+            element="H",
+            name=f"A{index}",
+            residue_name="MOL",
+        )
+        for index, (position, velocity, force, mass) in enumerate(zip(
+            positions, velocities, forces, (0.4, 15.6, 1.0),
+        ))
+    ]
+    system = System(atoms)
+    integrator = DrudeLangevinIntegrator(
+        timestep=1.0,
+        temperature=0.0,
+        friction=0.001,
+        drude_temperature=0.0,
+        drude_friction=0.02,
+        max_drude_distance=0.0,
+    )
+
+    class Potential:
+        base_context = context
+
+    class Calculator:
+        potential = Potential()
+
+    integrator.bind(Calculator())
+    actual_positions, actual_velocities = integrator.integrate(system)
+    expected_positions = np.asarray(
+        expected.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom),
+    )
+    expected_velocities = np.asarray(
+        expected.getVelocities(asNumpy=True).value_in_unit(
+            openmm.unit.angstrom/openmm.unit.femtosecond,
+        ),
+    )
+    assert np.allclose(actual_positions, expected_positions, atol=2e-7)
+    assert np.allclose(actual_velocities, expected_velocities, atol=2e-7)
+
+
+def test_settle_handles_drude_water_and_updates_virtual_site():
+    omm_system = openmm.System()
+    for mass in (15.6, 1.0, 1.0, 0.0, 0.4):
+        omm_system.addParticle(mass)
+    weights = (0.589781071, 0.2051094645, 0.2051094645)
+    omm_system.setVirtualSite(
+        3,
+        openmm.ThreeParticleAverageSite(0, 1, 2, *weights),
+    )
+    drude_force = openmm.DrudeForce()
+    drude_force.addParticle(4, 0, -1, -1, -1, -1.0, 0.001, 1.0, 1.0)
+    omm_system.addForce(drude_force)
+    context = openmm.Context(omm_system, openmm.VerletIntegrator(0.001))
+
+    angle = np.deg2rad(104.52)
+    positions = np.array([
+        [0.0, 0.0, 0.0],
+        [0.9572, 0.0, 0.0],
+        [0.9572*np.cos(angle), 0.9572*np.sin(angle), 0.0],
+        [9.0, 9.0, 9.0],
+        [0.01, 0.0, 0.0],
+    ])
+    masses = (15.6, 1.0, 1.0, 0.0, 0.4)
+    names = ("O", "H1", "H2", "M", "OD")
+    elements = ("O", "H", "H", "EP", "EP")
+    atoms = [
+        Atom(
+            position=position,
+            velocity=np.array([0.001*index, -0.001*index, 0.0]),
+            mass=mass,
+            residue=0,
+            element=element,
+            name=name,
+            residue_name="HOH",
+        )
+        for index, (position, mass, name, element) in enumerate(zip(
+            positions, masses, names, elements,
+        ))
+    ]
+    system = System(atoms)
+    integrator = DrudeLangevinIntegrator(
+        1.0, 0.0, 0.0, 0.0, 0.0, max_drude_distance=0.0,
+    )
+
+    class Potential:
+        base_context = context
+
+    class Calculator:
+        potential = Potential()
+
+    integrator.bind(Calculator())
+    settle = SETTLE(oh_distance=0.9572, hh_distance=1.5139006545)
+    integrator.register_plugin(settle)
+    updated, _ = integrator.integrate(system)
+
+    assert settle._get_hoh_residues(system) == [[0, 1, 2]]
+    assert np.linalg.norm(updated[1] - updated[0]) == pytest.approx(0.9572)
+    assert np.linalg.norm(updated[2] - updated[0]) == pytest.approx(0.9572)
+    assert np.linalg.norm(updated[2] - updated[1]) == pytest.approx(1.5139006545)
+    assert updated[3] == pytest.approx(np.asarray(weights) @ updated[:3])
+
+
+def test_drude_scf_integrator_separates_eom_from_relaxation():
+    omm_system = openmm.System()
+    drude = omm_system.addParticle(0.4)
+    parent = omm_system.addParticle(10.0)
+    drude_force = openmm.DrudeForce()
+    drude_force.addParticle(drude, parent, -1, -1, -1, -1.0, 0.001, 1.0, 1.0)
+    omm_system.addForce(drude_force)
+    context = openmm.Context(omm_system, openmm.VerletIntegrator(0.001))
+
+    atoms = [
+        Atom(
+            position=np.array([0.0, 0.0, 0.0]),
+            velocity=np.array([0.5, 0.0, 0.0]),
+            force=np.array([1000.0, 0.0, 0.0]),
+            mass=0.4,
+            element="EP",
+            name="D",
+            residue_name="MOL",
+        ),
+        Atom(
+            position=np.array([1.0, 0.0, 0.0]),
+            velocity=np.array([0.1, 0.0, 0.0]),
+            force=np.array([2.0, 0.0, 0.0]),
+            mass=10.0,
+            element="O",
+            name="P",
+            residue_name="MOL",
+        ),
+    ]
+    system = System(atoms)
+
+    class Potential:
+        base_context = context
+
+    class Calculator:
+        potential = Potential()
+        calculate_calls = 0
+
+        def calculate(self):
+            self.calculate_calls += 1
+
+    calculator = Calculator()
+    integrator = DrudeSCFIntegrator(VerletIntegrator(1.0))
+    integrator.bind(calculator)
+    positions, velocities = integrator.integrate(system)
+
+    # The EOM stage neither calls the calculator nor moves the Drude.
+    assert calculator.calculate_calls == 0
+    assert positions[drude] == pytest.approx(system.positions[drude])
+    assert velocities[drude] == pytest.approx(np.zeros(3))
+    # The physical parent follows the wrapped Verlet EOM.
+    assert velocities[parent, 0] == pytest.approx(0.10002)
+    assert positions[parent, 0] == pytest.approx(1.10002)
+
+
+def test_drude_scf_integrator_finds_mm_potential_in_composite():
+    omm_system = openmm.System()
+    omm_system.addParticle(0.4)
+    omm_system.addParticle(10.0)
+    force = openmm.DrudeForce()
+    force.addParticle(0, 1, -1, -1, -1, -1.0, 0.001, 1.0, 1.0)
+    omm_system.addForce(force)
+    context = openmm.Context(omm_system, openmm.VerletIntegrator(0.001))
+
+    class MM:
+        class Potential:
+            base_context = context
+        potential = Potential()
+
+    class QM:
+        potential = object()
+
+    class Composite:
+        calculators = [QM(), MM()]
+
+    integrator = DrudeSCFIntegrator(VerletIntegrator(1.0))
+    integrator.bind(Composite())
+
+    assert integrator.drude_indices.tolist() == [0]
```
