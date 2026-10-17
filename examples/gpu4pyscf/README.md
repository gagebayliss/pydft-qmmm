# GPU4PySCF with virtual sites and MM PME

The molecular PySCF interface was adapted from
[Jackcarzy/pydft-qmmm](https://github.com/Jackcarzy/pydft-qmmm/tree/2cfe4d055c83fdc0d67bb54cdd7d8a4a2ccf603d/pydft_qmmm/interfaces/pyscf).
Use `interface="pyscf-mol"` and `device="gpu"` on the cluster, or
`device="cpu"` to run the same interface with CPU PySCF.

Install this checkout with `python -m pip install -e '.[pyscf,tests]'`
in your existing PyDFT-QMMM environment. Install GPU4PySCF and CuPy
following the [GPU4PySCF installation instructions](https://github.com/pyscf/gpu4pyscf#installation)
for the cluster's CUDA runtime. GPU4PySCF is optional and is imported only
when `device="gpu"` is requested. There is no automatic CPU fallback.

```python
from pydft_qmmm import MMHamiltonian, QMHamiltonian, QMMMHamiltonian

qm = QMHamiltonian(
    interface="pyscf-mol",
    device="gpu",
    basis="def2-SVP",
    functional="PBE",
    charge=0,
    multiplicity=1,
    density_fit=True,
    conv_tol=1e-10,
)
mm = MMHamiltonian(
    forcefield="water.xml",  # Your force field, including virtual sites.
    nonbonded_method="PME",
    platform="CUDA",  # Independent of the QM device; CPU also works.
)
coupling = QMMMHamiltonian(
    "electrostatic", "mechanical", cutoff=8.0,
)
# Example: first four particles form the QM water, including its M site.
hamiltonian = qm[:4] + mm[4:] + coupling
calculator = hamiltonian.build_calculator(system)
```

This selects QM electrostatic embedding for subsystem II and force-field
charges with OpenMM PME for long range (`long_range="mechanical"`).
The near-region MM electrostatic term is subtracted by the existing
coupling code to avoid double counting. `long_range="electrostatic"`
requests a different mode that puts a reciprocal potential into the QM
Hamiltonian; this interface deliberately does not implement that mode.
No helPME-py dependency is needed for the MM PME mode above.

Load virtual-site definitions with `System.load(pdb, forcefield_xml)`.
Include all particles of each QM residue in the QM selection. `Ep` particles
(virtual sites and Drudes) are excluded from the QM molecule. MM virtual
sites and Drudes in subsystem II remain point charges, and forces on
supported virtual sites are distributed to their parents. As with the
existing Psi4 interface, site slots retain the unredistributed force;
physical parent forces already include it. Use the existing `Virtual`
integrator plugin to update site positions during dynamics and the
appropriate stationary-particle/Drude plugins for your model. Direct
coordinate perturbations must also recompute virtual-site positions.
The repository currently supports two- and three-particle average sites.

## Validation

From the repository root, CPU checks require no GPU:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
    tests/pyscf_virtual_test.py tests/pyscf_backend_test.py -q
```

They check physical forces against energy differences, including virtual
site parent motion, interleaved QM extra particles, an independent MM
Drude charge, cache invalidation, copying, backend dispatch, and a periodic
TIP4P-Ew calculation with near and far MM waters.

On an allocated CUDA node run:

```bash
PYDFT_TEST_GPU=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
    tests/pyscf_virtual_test.py tests/pyscf_backend_test.py -q
```

The opt-in tests compare CPU and GPU energies and all forces with and
without density fitting. Setting `PYDFT_TEST_GPU=1` makes an unavailable
GPU/backend fail the checks instead of silently skipping them. Local CPU
checks establish the mapping and force chain rule; actual CUDA execution
still needs this cluster check.
