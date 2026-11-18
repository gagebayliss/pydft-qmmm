# Virtual-site-only patch

Source: master `f3b13c378a25096f9727ddd5aa3dfb65e426905c`.
Baseline: upstream `3daf587` (Suppress Psi4 task planner logging).

`virtual-sites.patch` selectively ports the virtual-site features from master
onto that baseline. It is not a whole-file snapshot or a sequence of mixed
feature commits. Full baseline blob hashes are included for three-way merging.
The recipient's branch is unknown, so compatibility with that branch has not
been verified; its agent can resolve differences using this manifest.

From the recipient's repository root:

```sh
git apply --check /path/to/virtual-sites.patch
git apply /path/to/virtual-sites.patch
```

If the check fails because their branch has diverged, with the baseline Git
objects available and a clean worktree:

```sh
git apply --3way /path/to/virtual-sites.patch
```

The three-way form stages changes and may leave conflicts to resolve. Apply
from the repository root: Git can omit paths outside the current subdirectory.

## Included

- `system/system.py`: virtual-site metadata, helper attachment, and automatic
  coordinate reconstruction when positions change. The System notifier runs
  before subsequently registered interface notifiers.
- `utils/virtual_sites.py`, `utils/__init__.py`: master's definitions,
  extraction, batched position calculation, force propagation, and exports.
  Optional CuPy acceleration retains its NumPy fallback.
- `interfaces/openmm/openmm_factory.py`: Ep/Lp topology handling, modeller
  quantity/extra-particle handling, site extraction and registration **before**
  auxiliary-system construction, and auxiliary virtual-site definitions.
  This includes the latest subtraction-context fix.
- `interfaces/openmm/openmm_interface.py`: zero the auxiliary forces-only mask
  along with the base and auxiliary energy-group masks. No additional force
  redistribution is added: OpenMM already distributes its site forces.
- `interfaces/psi4/psi4_interface.py`: omit Ep/Lp from QM nuclei and QM-gradient
  indexing; propagate external-charge forces to parent atoms; clone density
  matrices before adding spin densities so force evaluation does not mutate
  the cached wavefunction.
- `potentials/pme_potential.py`: omit Ep/Lp consistently from nuclear charges,
  nuclear evaluation coordinates, force indexing, and excluded-energy indexing.
- `integrators/{integrator,verlet_integrator,langevin_integrator}.py`: avoid
  division by zero mass in integration and kinetic energy. Extracted from
  master's logic while retaining NumPy instead of importing the general GPU
  integrator refactor. Zero-mass particles receive zero velocity updates;
  System reconstructs their dependent coordinates upon position assignment.
- `utils/misc.py`: mass-safe initial velocity generation only. Zero-mass
  particles get zero initial velocity instead of NaN.
- `plugins/settle/settle.py`: exclude Ep/Lp from three-atom water constraints;
  make kinetic energy mass-safe; use the existing residue-selection signature
  consistently in `constrain_velocities`.
- Dedicated TIP4P-Ew fixtures and `tests/virtual_sites_regression_test.py`.
  Tests use OpenMM Reference and isolate QM output in pytest temporary paths.
- Master's `tests/virtual_test.py` and all of `tests/tip4pew_data/`, including
  both PDB files, the force-field XML, and the subsystem-II JSON. Existing test
  logic and data values are preserved; trailing whitespace is normalized.

## Deliberately omitted

No numerical-gradient changes: System already reconstructs sites on position
updates. No Drude SCF changes, thermostats/barostats, logging/timing, general
CuPy integrator or SETTLE refactors, PME alpha/grid defaults or warnings,
cache/box-invalidation changes, OpenMM charge-update refactors, editor swap
files, scratch output, or exploratory force-reference binaries.

The unfinished `System.copy`, `__deepcopy__`, and `copy_selection` additions on
master are not dependencies of this feature and are omitted. No legacy virtual
plugin is added: the System position notifier owns that behavior. Branches
with an older virtual plugin/Drude virtual-site helper should reconcile those
paths and imports to avoid duplicate propagation. No coupling-mode or shared
force-matrix fixes from the earlier investigation are included.

Existing `file_manager.py`, selection utilities, constants, SETTLE math
utilities, simulation wrapper, and Hamiltonians need no additional edits on
the stated baseline for these features.

## Validation

Checked the patch on a clean archive of baseline `3daf587`, applied it, parsed
the Python sources, and verified the resulting package against the isolated
test tree. Combined results for `virtual_test.py` and the dedicated regression
tests: **9 passed, 1 skipped** with Psi4 1.12a1.dev9,
OpenMM 8.5.2, NumPy, CPU execution, and the Reference OpenMM platform.

Checks cover auxiliary-site definitions, notifier-driven positions in both
contexts, Verlet/Langevin massless dynamics and kinetic energy, SETTLE,
the MM force versus an energy finite difference, and subsystem-II agreement
between electrostatic/mechanical, electrostatic/none, and electrostatic/cutoff.
The PME nuclear-charge check is skipped because `helpme_py` is unavailable;
full QM/MM/PME and GPU execution were not validated here.

To rerun after applying:

```sh
OPENMM_CPU_THREADS=1 OMP_NUM_THREADS=1 python -B -m pytest -q \
  -p no:cacheprovider tests/virtual_test.py tests/virtual_sites_regression_test.py
```
