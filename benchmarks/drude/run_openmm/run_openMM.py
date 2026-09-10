from openmm.app import *
from openmm import *
from simtk.unit import *
from sys import stdout

import numpy as np
from pathlib import Path

# temperature=300*kelvin
# pressure = 1.0*atmosphere
# barofreq = 100

write_output = 10000 # frequency for writing trajectory output

ffdir = str(Path(__file__).resolve().parents[1] / "swm4ndp_data") + "/"

# integrator input: temperature, friction, timestep 
integrator = DrudeSCFIntegrator(
    0.001 * picoseconds,
)

# OpenMM uses kJ/mol/nm; PyDFT-QMMM uses kJ/mol/Angstrom.
integrator.setMinimizationErrorTolerance(1e-5)

residue_xml_list = [ ffdir + "swm4ndp_residues.xml", ]

# add residue definitions to Topology
for residue_file in residue_xml_list:
    Topology().loadBondDefinitions(residue_file)

# load pdb after adding definitions to Topology
pdb = PDBFile(str(Path(__file__).with_name("hoh_trimer.pdb")))

modeller = Modeller(pdb.topology, pdb.positions)
forcefield = ForceField( ffdir + "swm4ndp.xml")
modeller.addExtraParticles(forcefield)

# create system class
system = forcefield.createSystem(modeller.topology, nonbondedCutoff=1.4*nanometer, rigidWater=False, constraints=None )

for i in range(system.getNumParticles()):
    system.setParticleMass(i, 0.0 * dalton)

# get force objects to set calculation method
nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == NonbondedForce][0]
# customNonbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == CustomNonbondedForce][0]

# use PME for electrostatics
nbondedForce.setNonbondedMethod(NonbondedForce.NoCutoff)
# Cutoff for LJ
print( "setting CustomNonbondedForce(LJ) method to CutoffPeriodic" )
# customNonbondedForce.setNonbondedMethod(min(nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic))

# this chooses the platform for executing kernels.  CUDA is fastest for GPUs
platform = Platform.getPlatformByName('CPU')

# create the simulations object
simmd = Simulation(modeller.topology, system, integrator, platform)
simmd.context.setPositions(modeller.positions)

# Match the unconstrained PyDFT-QMMM energy for Cartesian gradients.
# Zero masses freeze the real atoms; DrudeSCFIntegrator relaxes the Drudes.
simmd.context.computeVirtualSites()

# # this is for writing binary trajectory file of coordinates
# simmd.reporters = []
# simmd.reporters.append(DCDReporter('md_npt_atm.dcd', write_output))

print()

drude_indices = np.array([4,9,14])
state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
print("Potential Energy" , str(state.getPotentialEnergy()))
print("drude positions")
print(state.getPositions(asNumpy=True)[drude_indices])
simmd.step(1)
state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
print("Potential Energy" , str(state.getPotentialEnergy()))
print("drude positions")
print(state.getPositions(asNumpy=True)[drude_indices])
simmd.step(1)
state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
print("Potential Energy" , str(state.getPotentialEnergy()))
print("drude positions")
print(state.getPositions(asNumpy=True)[drude_indices])

print("post-SCF Drude forces (kJ/mol/Angstrom)")
print(state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole/angstrom)[drude_indices])
