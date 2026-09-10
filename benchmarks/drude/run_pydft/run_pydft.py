import numpy as np
from pathlib import Path

from pydft_qmmm import MMHamiltonian
from pydft_qmmm import QMHamiltonian
from pydft_qmmm import QMMMHamiltonian
from pydft_qmmm import System
from pydft_qmmm import VerletIntegrator
from pydft_qmmm.plugins import SETTLE
from pydft_qmmm.utils import ELEMENT_TO_MASS
from pydft_qmmm.utils import Subsystem
from pydft_qmmm.plugins import Virtual
from pydft_qmmm.plugins import Stationary
from pydft_qmmm.plugins import SETTLE

from pydft_qmmm.plugins import DrudeSCF
from pydft_qmmm.plugins.drude import extract_drude_data

drude_indices = [4,9,14]

forcefield_dir = str(Path(__file__).resolve().parents[1] / "swm4ndp_data") + "/"
system = System.load(
    str(Path(__file__).with_name("hoh_trimer.pdb")),
    forcefield_dir + "swm4ndp.xml"
)


mm = MMHamiltonian(
    forcefield=[
        forcefield_dir + "swm4ndp.xml",
        forcefield_dir + "swm4ndp_residues.xml",
    ],
    nonbonded_method="NoCutoff",
)

qm = QMHamiltonian(
    "psi4",
    functional="HF", 
    charge=0,
    multiplicity=1,
    basis="sto-3g",
    guess="read",
    scf_type="DF",
    output_file="output/qmmm_psi4",
)

# qmmm = QMMMHamiltonian("electrostatic", "none")

total = mm 
# total = mm[5:] + qm[0:5] + qmmm
calculator = total.build_calculator(system)
# mm_calc = None
# for calc in calculator.calculators:
#     if calc.calculator_group == "MM":
#         mm_calc = calc

integrator = VerletIntegrator(1.0)
# Real atoms remain fixed during this SCF benchmark; no SETTLE projection.

system.masses[:] = 0.0

stationary_atoms = [
    int(atom) for atom in np.where(system.masses.base == 0)[0]
]

if stationary_atoms:
    # fake_atoms = system.select("element Ep Lp")
    query = "atom"
    for atom in stationary_atoms:
        query += f" {atom}"
        system.masses[atom] = ELEMENT_TO_MASS.get(
            system.elements[atom],
            0.1,
        )
    integrator.register_plugin(Stationary(query), 0)
integrator.register_plugin(Virtual(),0)
drude_data = extract_drude_data(calculator.potential.base_context.getSystem())
drude_indices = drude_data.drude_indices
drude_scf = DrudeSCF(calculator, drude_data, force_tolerance=1e-6)
integrator.register_plugin(drude_scf,0)

results = calculator.calculate()
print("results")
print(results)

system.forces = results.forces

print("system.positions[drude_indices]")
print(system.positions[drude_indices])
old_pos = system.positions.base.copy()

positions, velocities = integrator.integrate(system)
system.positions[:] = positions
system.velocities[:] = velocities
# The SCF iterator evaluates forces before its final update; refresh them here.
results = calculator.calculate()
system.forces[:] = results.forces
print("post-SCF Drude forces (kJ/mol/Angstrom)")
print(results.forces[drude_indices])

print("system.positions[drude_indices]")
print(system.positions[drude_indices])
new_pos = system.positions.base.copy()

print("np.array_equal(new_pos,old_pos)")
print(np.array_equal(new_pos,old_pos))

