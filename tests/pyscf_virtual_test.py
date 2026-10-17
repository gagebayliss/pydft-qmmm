"""CPU gradient checks and optional cluster GPU parity for extra particles."""
from copy import deepcopy
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyscf")

from pydft_qmmm import Atom, System, QMHamiltonian
from pydft_qmmm.interfaces.pyscf.pyscf_factory import pyscf_interface_factory
from pydft_qmmm.interfaces.pyscf.pyscf_utils import qm_atom_indices
from pydft_qmmm.utils import Subsystem, virtual_sites
from pydft_qmmm.utils.virtual_sites import VirtualSite


@pytest.fixture
def embedded():
    # Interleave an Ep in subsystem I to exercise gradient index mapping.
    elements = ["O", "Ep", "H", "H", "O", "H", "H", "Ep", "Ep"]
    positions = [[0, 0, 0], [0, 0, 0], [.96, 0, 0], [-.24, .93, 0],
                 [3, .4, .2], [3.96, .4, .2], [2.76, 1.33, .2],
                 [0, 0, 0], [3.1, .5, .3]]
    atoms = [Atom(element=e, position=np.array(p, dtype=float),
                  subsystem=Subsystem.I if i < 4 else Subsystem.II,
                  charge={5: .52, 6: .52, 7: -1.04, 8: -.1}.get(i, 0.))
             for i, (e, p) in enumerate(zip(elements, positions))]
    system = System(atoms, virtual_sites=[
        VirtualSite(1, "three_average", (0, 2, 3), (.8, .1, .1)),
        VirtualSite(7, "three_average", (4, 5, 6), (.8, .1, .1)),
    ])
    system.positions[:] = virtual_sites.compute_positions(system, system.positions)
    return system


def factory(system, **kwargs):
    functional = kwargs.pop("functional", None)
    return pyscf_interface_factory(system, basis="sto-3g", functional=functional,
                                   charge=0, multiplicity=1, conv_tol=1e-12,
                                   **kwargs)


def test_virtual_and_drude_mapping(embedded):
    assert qm_atom_indices(embedded) == (0, 2, 3)
    potential = factory(embedded)
    state = potential._run()
    assert state.mol.natm == 3
    assert state.embed_indices == (4, 5, 6, 7, 8)
    assert potential.compute_forces().shape == (9, 3)
    assert np.all(potential.compute_forces()[1] == 0)


@pytest.mark.parametrize("functional,density_fit", [(None, False), ("PBE", True)])
def test_virtual_parent_forces_match_energy(embedded, functional, density_fit):
    potential = factory(embedded, functional=functional, density_fit=density_fit)
    forces = potential.compute_forces().copy()
    initial = np.asarray(embedded.positions).copy()
    # Probe every physical atom and the independent Drude particle.
    h = 1e-4
    for atom in (0, 2, 3, 4, 5, 6, 8):
        for axis in range(3):
            energies = []
            for sign in (1, -1):
                displaced = initial.copy()
                displaced[atom, axis] += sign*h
                embedded.positions[:] = virtual_sites.compute_positions(embedded, displaced)
                energies.append(potential.compute_energy())
            numerical = -(energies[0] - energies[1]) / (2*h)
            assert forces[atom, axis] == pytest.approx(numerical, abs=2e-3)
    embedded.positions[:] = initial
    # Site slots retain their force under the repository convention.
    assert forces[[0, 2, 3, 4, 5, 6, 8]].sum(axis=0) == pytest.approx(0, abs=2e-5)


def test_cache_and_copy(embedded):
    potential = factory(embedded)
    potential.compute_energy()
    solver = potential.method[0]
    potential.compute_forces()
    assert potential.method[0] is solver
    copied = deepcopy(potential)
    assert copied.system is not embedded
    assert copied.method[0] is None
    assert copied.compute_energy() == pytest.approx(potential.compute_energy(), abs=1e-7)
    embedded.charges[7] *= .9
    potential.compute_energy()
    assert potential.method[0] is not solver


def test_mechanical_embedding_and_discovery(embedded):
    calculator = QMHamiltonian(interface="pyscf-mol", basis="sto-3g",
                               charge=0, multiplicity=1)[0:4].build_calculator(embedded)
    potential = calculator.potential
    potential.configure_electrostatic_embedding(False)
    assert potential._run().embed_indices == ()
    with pytest.raises(NotImplementedError, match="MM PME"):
        potential.add_electronic_potential(SimpleNamespace())


@pytest.mark.skipif(os.environ.get("PYDFT_TEST_GPU") != "1",
                    reason="set PYDFT_TEST_GPU=1 on a CUDA node")
@pytest.mark.parametrize("density_fit", [False, True])
@pytest.mark.parametrize("functional", [None, "PBE"])
def test_gpu_virtual_embedding_parity(embedded, density_fit, functional):
    cpu = factory(embedded, density_fit=density_fit, functional=functional)
    gpu = factory(embedded, device="gpu", density_fit=density_fit, functional=functional)
    assert gpu.compute_energy() == pytest.approx(cpu.compute_energy(), abs=1e-4)
    assert gpu.compute_forces() == pytest.approx(cpu.compute_forces(), abs=2e-3)


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "gpu", marks=pytest.mark.skipif(
        os.environ.get("PYDFT_TEST_GPU") != "1",
        reason="set PYDFT_TEST_GPU=1 on a CUDA node",
    ),
)])
def test_openmm_pme_with_close_qm_embedding(device):
    """Use QM charges nearby and force-field charges for long-range PME."""
    from pydft_qmmm import MMHamiltonian, QMMMHamiltonian
    import openmm

    system = System.load("tests/tip4pew_data/tip4pew_trimer.pdb",
                         "tests/tip4pew_data/tip4pew.xml")
    system.box[:] = np.eye(3)*30
    positions = np.asarray(system.positions).copy()
    for start, origin in ((0, [5, 5, 5]), (4, [8, 5.5, 5.2]), (8, [15, 6, 5])):
        positions[start:start+4] += np.array(origin)-positions[start]
    system.positions[:] = virtual_sites.compute_positions(system, positions)
    mm = MMHamiltonian(forcefield="tests/tip4pew_data/tip4pew.xml",
                       nonbonded_method="PME", nonbonded_cutoff=10,
                       pme_gridnumber=32, pme_alpha=3, platform="Reference")
    qm = QMHamiltonian(interface="pyscf-mol", basis="sto-3g",
                       charge=0, multiplicity=1, conv_tol=1e-12, device=device)
    coupling = QMMMHamiltonian("electrostatic", "mechanical", cutoff=5)
    calculator = (qm[:4]+mm[4:]+coupling).build_calculator(system)
    results = calculator.calculate()
    potential = next(c.potential for c in calculator.calculators if c.name == "PySCF")
    assert potential._run().qm_indices == (0, 1, 2)
    assert potential._run().embed_indices == (4, 5, 6, 7)
    assert system.select("subsystem III") == frozenset(range(8, 12))
    assert set(c.name for c in calculator.calculators) == {"OpenMM", "PySCF"}
    mm_potential = next(c.potential for c in calculator.calculators if c.name == "OpenMM")
    nonbonded = next(f for f in mm_potential.base_context.getSystem().getForces()
                    if isinstance(f, openmm.NonbondedForce))
    assert nonbonded.getNonbondedMethod() == openmm.NonbondedForce.PME
    assert nonbonded.getParticleParameters(3)[0]._value != 0
    assert np.isfinite(results.energy)
    assert np.all(np.isfinite(results.forces))
    # Both the near and far MM virtual-site parents must see forces.
    initial = np.asarray(system.positions).copy()
    h = 1e-4
    for atom in (0, 1, 4, 5, 8, 9):
        for axis in range(3):
            energies = []
            for sign in (1, -1):
                displaced = initial.copy()
                displaced[atom, axis] += sign*h
                system.positions[:] = virtual_sites.compute_positions(system, displaced)
                energies.append(calculator.calculate(return_forces=False).energy)
            assert results.forces[atom, axis] == pytest.approx(
                -(energies[0]-energies[1])/(2*h), abs=3e-3,
            )
    system.positions[:] = initial
