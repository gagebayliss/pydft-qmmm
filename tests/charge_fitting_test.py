import numpy as np
import psi4
import pytest

from pydft_qmmm import MMHamiltonian, QMHamiltonian, QMMMHamiltonian, System
from pydft_qmmm.plugins.charge_fitting import QMChargeFit
from pydft_qmmm.utils import Subsystem


def water_calculator(model, scheme, tmp_path):
    directory = f"tests/{model}_data"
    if model == "swm4ndp":
        system = System.load(f"{directory}/swm4ndp_qmmm_231.pdb")
        size = 5
    else:
        system = System.load(f"{directory}/hoh_123.pdb")
        size = 3
    mm = MMHamiltonian(
        forcefield=[f"{directory}/{model}.xml", f"{directory}/{model}_residues.xml"],
        nonbonded_method="NoCutoff", platform="Reference",
    )
    qm = QMHamiltonian(basis="sto-3g", functional="HF", charge=0, multiplicity=1,
                       guess="sad", output_file=str(tmp_path / "psi4.log"))
    coupling = QMMMHamiltonian("electrostatic", "mechanical", partition=None)
    # Fit the second water, embed the first, and leave all others in III.
    calculator = (qm[size:2*size] + mm[:size, 2*size:] + coupling).build_calculator(system)
    system.subsystems[:size] = Subsystem.II
    calculator.register_plugin(QMChargeFit(scheme))
    return system, calculator, size


@pytest.mark.parametrize("model", ["swm4ndp", "spce"])
@pytest.mark.parametrize("scheme", ["mulliken", "mbis"])
def test_embedded_water_charges(model, scheme, tmp_path):
    system, calculator, size = water_calculator(model, scheme, tmp_path)
    geometry = "0 1\nunits angstrom\nsymmetry c1\nno_com\nno_reorient\n"
    for i in range(size, size + 3):
        x, y, z = system.positions[i]
        geometry += f"{system.elements[i]} {x} {y} {z}\n"
    xyz_bohr = system.positions[:size] / psi4.constants.bohr2angstroms
    points = np.column_stack((system.charges[:size], xyz_bohr))
    _, wfn = psi4.energy("hf/sto-3g", molecule=psi4.geometry(geometry),
                         external_potentials=[points, None, None], return_wfn=True)
    psi4.oeprop(wfn, scheme.upper() + "_CHARGES")
    expected = np.asarray(wfn.variable(scheme.upper() + " CHARGES")).ravel()

    calculator.calculate(return_forces=False, return_components=False)
    np.testing.assert_allclose(system.charges[size:size+3], expected, atol=1e-6, rtol=0)

@pytest.mark.parametrize("model", ["swm4ndp", "spce"])
@pytest.mark.parametrize("scheme", ["mulliken", "mbis"])
def test_correct_charge_order(model, scheme, tmp_path):
    """testing vs. saved reference charges; intended as regression test
    in case order of charges output by psi4 changes
    """
    system, calculator, size = water_calculator(model, scheme, tmp_path)
    calculator.calculate(return_forces=False, return_components=False)
    expected = np.load(f"tests/charge_fitting_data/{model}_{scheme}_charges.npy")
    np.testing.assert_allclose(system.charges[size:size+3], expected, atol=1e-6, rtol=0)

def test_swm4ndp_charge_indices(tmp_path):
    system, calculator, _ = water_calculator("swm4ndp", "mulliken", tmp_path)
    before = system.charges.copy()
    calculator.calculate(return_forces=False, return_components=False)
    assert np.all(system.charges[5:8] != before[5:8])
    np.testing.assert_array_equal(system.charges[8:10], [0, 0])
    np.testing.assert_array_equal(system.charges[:5], before[:5])
    np.testing.assert_array_equal(system.charges[10:], before[10:])
