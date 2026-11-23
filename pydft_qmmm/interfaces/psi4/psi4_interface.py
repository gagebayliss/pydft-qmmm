"""The Psi4 software interface and potential.

This module contains the software interface for storing and manipulating
Psi4 data types and the potential using Psi4 to calculate energies
and forces.
"""
from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np
import psi4

from pydft_qmmm.interfaces import QMInterface
from pydft_qmmm.potentials import AtomicPotential
from pydft_qmmm.utils import BOHR_PER_ANGSTROM
from pydft_qmmm.utils import KJMOL_PER_EH
from pydft_qmmm.utils import PyDFTQMMMException
from pydft_qmmm.utils import system_cache

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray
    from pydft_qmmm.potentials import ElectronicPotential
    from pydft_qmmm import System  # noqa: F401
    from . import psi4_utils


@dataclass(frozen=True)
class Psi4Interface(QMInterface):
    r"""A mix-in for storing and manipulating Psi4 data types.

    Args:
        system: The system that will inform the interface to the
            external software.
        functional: The name of the functional to use in QM
            calculations.
        charge: The net charge (:math:`e`) of the QM subsystem.
        multiplicity: The spin multiplicity of the QM subsystem.
        output_file: The file to which Psi4 output is written.
        output_interval: The interval at which Psi4 output should be
            written, e.g., the default value of 1 means that output
            will be written every calculation.

    Attributes:
        potentials: A list of electronic potentials to incorporate into
            QM calculations.
        frame: The estimated current frame for output writing purposes.
    """
    functional: str
    charge: int
    multiplicity: int
    output_file: str
    output_interval: int
    potentials: list[ElectronicPotential] = field(
        default_factory=list,
        init=False,
    )
    components: list[str] = field(
        default_factory=list,
        init=False,
    )
    frame: list[int] = field(
        default_factory=lambda: [0],
        init=False,
    )

    def __post_init__(self) -> None:
        """Perform post-initialization setup.
        """
        # This currently suppresses the Psi4 task planner output.
        logging.getLogger("psi4").setLevel(logging.CRITICAL + 1)

    def add_electronic_potential(self, potential: ElectronicPotential) -> None:
        """Add an electronic potential to apply before calculations.

        Args:
            potential: The electronic potential to incorporate into
                QM calculations.
        """
        self.potentials.append(potential)

    def fit_mm_charges(self,charge_fitting_scheme: str) -> NDArray[np.float64]:
        """Fit MM point charges to QM electron density.
                    
        Args:
            charge_fitting_scheme: one of "mulliken", "mbis".
                                            
        Returns:
            An array of charges for the MM atoms corresponding to QM nuclei.
        """
        charge_fitting_scheme = charge_fitting_scheme.lower()
        wfn = self._generate_wavefunction()
        charges = None
        if charge_fitting_scheme == "mulliken":
            psi4.oeprop(wfn, "MULLIKEN_CHARGES")
            charges = wfn.variable(f"MULLIKEN CHARGES")
        elif charge_fitting_scheme == "mbis":
            psi4.oeprop(wfn, "MBIS_CHARGES")
            charges = np.asarray(wfn.variable(f"MBIS CHARGES")).reshape((-1))
        else:
            raise ValueError(f"Unsupported charge fitting scheme: {charge_fitting_scheme}")
        return charges

    def return_external_ixn_component(self) -> None:
        """Return the external energy interaction as a component.
        """
        if "External Interaction Energy" not in self.components:
            self.components.append("External Interaction Energy")

    def get_density_matrix(self) -> psi4.core.Matrix:
        """Return the spin-summed AO density from the current SCF."""
        wfn = self._generate_wavefunction()
        density = wfn.Da().clone()
        density.add(wfn.Db())
        return density

    def compute_electric_field(
            self,
            points: NDArray[np.float64],
            density_matrix: psi4.core.Matrix | None = None,
    ) -> NDArray[np.float64]:
        """Return the QM field at points in angstrom, in kJ/mol/angstrom/e."""
        if density_matrix is None:
            density_matrix = self.get_density_matrix()
        points = np.asarray(points) * BOHR_PER_ANGSTROM
        if len(points) == 0:
            return np.empty((0, 3))
        basis = self._generate_basis()
        mints = psi4.core.MintsHelper(basis)
        field = mints.electric_field_value(
            psi4.core.Matrix.from_array(points), density_matrix,
        ).np
        molecule = basis.molecule()
        for atom, position in enumerate(molecule.geometry().np):
            displacement = points - position
            distance = np.linalg.norm(displacement, axis=1, keepdims=True)
            field += molecule.Z(atom) * displacement / distance**3
        return field * KJMOL_PER_EH * BOHR_PER_ANGSTROM

    def compute_embedding_energy(
            self,
            density_matrix: psi4.core.Matrix | None = None,
            *,
            point_charges: bool = True,
            diffuse_charges: bool = True,
            arbitrary_potential: bool = True,
    ) -> float:
        """Compute the selected embedding interactions in kJ/mol.
        """
        if not (point_charges or diffuse_charges or arbitrary_potential):
            return 0.0

        if density_matrix is None:
            wfn = self._generate_wavefunction()
            density_matrix = self.get_density_matrix()
            basis = wfn.basisset()

        external = psi4.core.ExternalPotential()

        # point charges
        charges = self._embedding_charges() if point_charges else []
        if not charges and not (arbitrary_potential and self.potentials):
            return 0.0
        external.appendCharges(charges)

        # diffuse charges
        #TODO: add diffuse code here when we support this
        if diffuse_charges:
            pass

        # arbitrary potential
        if arbitrary_potential and self.potentials:
            matrix = self._generate_external_potentials()[2]
            external.setMatrix(psi4.core.Matrix.from_array(matrix))
        
        # evaluate
        basis = self._generate_basis()
        potential = external.computePotentialMatrix(basis)
        energy = density_matrix.vector_dot(potential)
        energy += external.computeNuclearEnergy(basis.molecule())
        return energy * KJMOL_PER_EH

    @system_cache("positions", "charges", "elements", "subsystems", "box")
    def _generate_wavefunction(self) -> psi4.core.Wavefunction:
        """Generate the Psi4 Wavefunction object.

        Returns:
            The Psi4 Wavefunction object, which contains the energy
            and coefficients determined through SCF.
        """
        if not self.frame[0] % self.output_interval:
            psi4.core.set_output_file(self.output_file, True)
        molecule = self._generate_molecule()
        _, wfn = psi4.energy(
            self.functional,
            return_wfn=True,
            molecule=molecule,
            external_potentials=self._generate_external_potentials(),
        )
        wfn.to_file(
            wfn.get_scratch_filename(180),
        )
        if not self.frame[0] % self.output_interval:
            psi4.core.set_output_file("/dev/null", True)
        self.frame[0] += 1
        return wfn

    @system_cache("positions", "elements", "subsystems")
    def _generate_molecule(self) -> psi4.core.Molecule:
        """Generate the Psi4 Molecule object.

        Returns:
            The Psi4 Molecule object, which contains the positions,
            net charge, and net spin of atoms in the QM subsystem.
        """
        geometrystring = """\n"""
        geometrystring += str(self.charge) + " "
        geometrystring += str(self.multiplicity) + "\n"
        geometrystring += "symmetry c1\n"
        geometrystring += "no_reorient\nno_com\n"
        atoms = sorted(self.system.select("subsystem I and not element Ep Lp"))
        for atom in atoms:
            geometrystring = (
                geometrystring
                + str(self.system.elements[atom]) + str(atom) + " "
                + str(self.system.positions[atom][0]) + " "
                + str(self.system.positions[atom][1]) + " "
                + str(self.system.positions[atom][2]) + "\n"
            )
        molecule = psi4.geometry(geometrystring)
        c1_molecule = molecule.clone()
        c1_molecule._initial_cartesian = molecule._initial_cartesian.clone()
        c1_molecule.set_geometry(c1_molecule._initial_cartesian)
        c1_molecule.reset_point_group("c1")
        c1_molecule.fix_orientation(True)
        c1_molecule.fix_com(True)
        c1_molecule.update_geometry()
        return c1_molecule

    @system_cache("positions", "charges", "subsystems", "box")
    def _generate_external_potentials(self) -> list[Any] | None:
        r"""Generate the data structure needed to perform embedding.

        Returns:
            The list of coordinates (:math:`\mathrm{a.u.}`) and charges
            (:math:`e`) that will be electrostatically embedded by Psi4
            during calculations.
        """
        charges = self._embedding_charges()
        matrix = None
        if self.potentials:
            numint = self._generate_numinthelper()
            matrix = numint.potential_integral(self._generate_potential()).np
        # The second slot is reserved for diffuse embedding.
        return [np.array(charges) if charges else None, None, matrix]

    def _embedding_charges(self) -> list[list[float]]:
        """Return subsystem II charges and their current coordinates in bohr."""
        charges = []
        for i in sorted(self.system.select("subsystem II")):
            x, y, z = self.system.positions[i] * BOHR_PER_ANGSTROM
            charges.append([self.system.charges[i], x, y, z])
        return charges

    @system_cache("positions", "elements", "subsystems")
    def _generate_basis(self) -> psi4.core.BasisSet:
        """Build the AO basis at the current QM geometry."""
        return psi4.core.BasisSet.build(
            self._generate_molecule(), "BASIS", psi4.core.get_global_option("BASIS"),
        )

    @system_cache("positions", "elements", "subsystems")
    def _generate_numinthelper(self) -> psi4.core.NumIntHelper:
        """Build the integration grid at the current QM geometry."""
        basis_set = self._generate_basis()
        molecule = basis_set.molecule()
        grid = psi4.core.DFTGrid.build(molecule, basis_set)
        numinthelper = psi4.core.NumIntHelper(grid)
        return numinthelper

    @system_cache("positions", "elements", "subsystems")
    def _generate_potential(self) -> list[psi4.core.Vector]:
        """Generate the potential grid from ElectronicPotential objects.

        Returns:
            A list of Psi4 Vector objects containing the potential
            evaluated at the points on the Psi4 DFTGrid object.
        """
        numint = self._generate_numinthelper()
        blocks = []
        indices = []
        i = 0
        for block in numint.numint_grid().blocks():
            x = block.x().np
            y = block.y().np
            z = block.z().np
            blocks.append(np.stack((x, y, z), axis=-1))
            indices.append(i)
            i += block.npoints()
        xyz = np.concatenate(tuple(blocks), axis=0)
        v = np.zeros_like(xyz[:, 0])
        for potential in self.potentials:
            v += potential.compute_potential(
                xyz / BOHR_PER_ANGSTROM,
            ).flatten()
        v_split = np.split(v, indices)
        v_grid = [psi4.core.Vector.from_array(x) for x in v_split[1:]]
        return v_grid

    def update_options(self, **kwargs: psi4_utils.Psi4Options) -> None:
        """Set additional options for Psi4.

        Args:
            kwargs: Additional options to provide to Psi4.  See
                `Psi4 options`_ for additional Psi4 options.
        """
        psi4.set_options(kwargs)

class Psi4Potential(Psi4Interface, AtomicPotential):
    """A potential wrapping Psi4 functionality.

    Args:
        system: The system that will inform the interface to the
            external software.
        functional: The name of the functional to use in QM
            calculations.
        charge: The net charge (:math:`e`) of the QM subsystem.
        multiplicity: The spin multiplicity of the QM subsystem.
        output_file: The file to which Psi4 output is written.
        output_interval: The interval at which Psi4 output should be
            written, e.g., the default value of 1 means that output
            will be written every calculation.

    Attributes:
        potentials: A list of electronic potentials to incorporate into
            QM calculations.
        frame: The estimated current frame for output writing purposes.
    """

    def compute_energy(self) -> float:
        r"""Compute the energy of the system using Psi4.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the system.
        """
        wfn = self._generate_wavefunction()
        return wfn.energy() * KJMOL_PER_EH

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system using Psi4.

        Returns:
            The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            acting on atoms in the system.
        """
        qm_indices = sorted(self.system.select("subsystem I and not element Ep Lp"))
        forces_temp = np.zeros(self.system.positions.shape)
        wfn = self._generate_wavefunction()
        grads = psi4.gradient(
            self.functional,
            ref_wfn=wfn,
        )
        forces = grads.np * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
        forces_temp[qm_indices, :] += forces
        if self._generate_external_potentials() is None:
            return forces_temp
        if self._generate_external_potentials()[0] is not None:
            embed_indices = sorted(self.system.select("subsystem II"))
            grads = wfn.external_pot().gradient_on_charges()
            # `grads` will be `None` if a numerical gradient is
            # performed in Psi4, and so the following block
            # restores the analytic gradient on from and on the
            # embedded point charges.  This requires at least Psi4 1.11.
            if grads is None:
                if not hasattr(
                        wfn.external_pot(),
                        "computePotentialGradients",
                ):
                    raise PyDFTQMMMException(
                        textwrap.fill(
                            "\nGradients on embedded point charges "
                            "were not calculated, likely because Psi4 "
                            "opted for finite-difference gradients.  "
                            "PyDFT-QMMM can still calculate the "
                            "gradients on embedded point charges, but "
                            "only using Psi4 v1.11 or higher.  The "
                            "current Psi4 installation does not meet "
                            f"this requirement (v{psi4.__version__} is "
                            "currently installed).",
                        ),
                    )
                D = self.get_density_matrix()
                grads = wfn.external_pot().computePotentialGradients(
                    wfn.basisset(),
                    D,
                )
                forces = grads.np * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
                forces_temp[qm_indices, :] += forces
                grads = wfn.external_pot().gradient_on_charges()
            forces = grads.np * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
            forces_temp[embed_indices, :] += forces
        if self._generate_external_potentials()[2] is not None:
            numint = self._generate_numinthelper()
            v_grid = self._generate_potential()
            D = self.get_density_matrix()
            grads = numint.potential_gradient(v_grid, D)
            forces = grads.np * -KJMOL_PER_EH * BOHR_PER_ANGSTROM
            forces_temp[qm_indices, :] += forces
        # this has to run after everything
        forces_temp = self.system.vsite_helper.distribute_forces(forces_temp)
        return forces_temp

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy using Psi4.

        Returns:
            The components of the energy (:math:`\mathrm{kJ\;mol^{-1}}`)
            of the system.
        """
        components: dict[str, float] = {}
        wfn = self._generate_wavefunction()
        energy = wfn.energy() * KJMOL_PER_EH
        components["Nuclear Energy"] = (
            wfn.molecule().nuclear_repulsion_energy() * KJMOL_PER_EH
        )
        components["Electronic Energy"] = energy - components["Nuclear Energy"]
        if (
            "External Interaction Energy" in self.components
            and wfn.external_pot() is not None
        ):
            components["External Interaction Energy"] = self.compute_embedding_energy()
            components["Electronic Energy"] -= components["External Interaction Energy"]
        return components
