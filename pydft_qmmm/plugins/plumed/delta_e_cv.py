from __future__ import annotations

__all__ = ["DeltaECV"]

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .collective_variable import CollectiveVariable
from pydft_qmmm.calculators import CompositeCalculator

if TYPE_CHECKING:
    from pydft_qmmm.calculators import Calculator
    from pydft_qmmm.calculators import Results
    from numpy.typing import NDArray


class DeltaECV(CollectiveVariable):
    """The generalized solvation collective variable class.

    Args:
        calculator: The calculator whose system state and energies/forces
            will be used to calculate collective variables.
        scheme: a string of the form "{state 1} - {state 2}", e.g.
            "anion - neutral"; used as functional form of the CV.
        states: If MM only, a dict mapping the above states to dicts containing
            arrays of "indices" and "charges". If QM/MM, a dict mapping the above
            states to tuples for charge and multiplicity.
    """
    def __init__(
            self,
            scheme: str,
            states: dict[str,dict[str,NDArray[np.float64] | NDArray[np.int64]]] |\
                    dict[str,tuple],
            ):
        self.states = states
        self._cv_gradient = None
        self._state_names = (key.strip() for key in scheme.split("-"))
        if len(states) != 2:
            raise ValueError("...")
        if not (self._state_names[0] in states.keys() and\
                self._state_names[1] in states.keys()):
            raise ValueError("...")

    def get_scalar_cv(
            self,
            calculator: Calculator,
            results: Results,
    ) -> Results:
        r"""Calculate a scalar collective variable from system state
            and calculator results.

        Args:
            results: The energies and forces from a calculation.

        Returns:
            The scalar value of the collective variable.
        """
        system = calculator.system
        original_charges = system.charges.copy()

        skip_state = ""
        for state_name, state in self.states.items():
            if np.array_equal(original_charges,state):
                skip_state = state_name

        energies = list()
        forces = list() 

        for state_name in self._state_names:
            if skip_state == state_name:
                result = results
            elif isinstance(calculator,CompositeCalculator): #e.g., QM/MM calculation
                raise NotImplementedError("Cannot use QM/MM with DeltaECV")
            else:
                solvent_indices = self.states[state_name]["indices"]
                system.charges[solvent_indices] = self.states[state_name]["charges"]
                result = calculator.calculate()
            energies.append(result.energy)
            forces.append(result.forces)

        print(self._state_names)
        delta_E = energies[self._state_names[0]] - energies[self._state_names[1]]

        # used in chain_rule()
        self._cv_gradient = -(forces[self._state_names[0]] - forces[self._state_names[1]])

        system.charges[:] = original_charges
        return delta_E

    def chain_rule(
            self,
    ) -> NDArray[np.float64]:
        r"""Calculate change in atomic position per unit change in collective variable.

        Returns:
            An array of forces.
        """
        return self._cv_gradient.copy()
    
    @staticmethod
    def load_charge_state(xml_file, pdb_file, start: int, stop: int) -> dict:
        """Return {'indices': [...], 'charges': [...]} for PDB atoms[start:stop].

        Match PDB residue and atom names to XML Residues templates. Charges come
        from NonbondedForce Atom type/class rules, or residue Atom charge attributes
        when UseAttributeFromResidue requests them. Relative XML Includes are read.
        Charges are in elementary-charge units. No simulation libraries are needed.

        Requires explicit nonnegative bounds and a nonempty range. Ambiguous or
        missing templates/parameters raise ValueError rather than guessing. Alternate
        locations in the selected atoms are rejected; supply a single-conformer PDB.
        XML residue/atom names must match the PDB for every charge state.
        """
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop:
            raise ValueError('Require integer bounds 0 <= start < stop (stop is exclusive)')
        atoms = []
        for line in Path(pdb_file).read_text().splitlines():
            if line.startswith('ENDMDL'):
                break
            if line.startswith(('ATOM  ', 'HETATM')):
                atoms.append((line[17:20].strip(), line[12:16].strip(), line[16:17].strip()))
        if stop > len(atoms):
            raise ValueError(f'stop={stop} exceeds the {len(atoms)} PDB atoms')

        roots, visited, active = [], set(), set()
        def read_xml(path):
            path = Path(path).resolve()
            if path in active:
                raise ValueError(f'Cyclic XML Include: {path}')
            if path in visited:
                return
            active.add(path)
            root = ET.parse(path).getroot()
            if root.tag != 'ForceField':
                raise ValueError(f'{path} is not a ForceField XML file')
            for include in root.findall('Include'):
                read_xml(path.parent / include.attrib['file'])
            roots.append(root)
            active.remove(path)
            visited.add(path)
        read_xml(xml_file)
        templates, types, forces = {}, {}, []
        for root in roots:
            for residue in root.findall('./Residues/Residue'):
                templates.setdefault(residue.attrib['name'], []).append(residue)
            for atom_type in root.findall('./AtomTypes/Type'):
                name = atom_type.attrib['name']
                if name in types and types[name] != atom_type.attrib.get('class', ''):
                    raise ValueError(f'Conflicting classes for atom type {name}')
                types[name] = atom_type.attrib.get('class', '')
            forces.extend(root.findall('NonbondedForce'))
        if not forces:
            raise ValueError('No NonbondedForce found in XML')

        charges = []
        for index in range(start, stop):
            residue_name, atom_name, altloc = atoms[index]
            label = f'PDB atom {index} ({residue_name}:{atom_name})'
            if altloc:
                raise ValueError(f'{label}: alternate locations are unsupported')
            matches = templates.get(residue_name, [])
            if len(matches) != 1:
                raise ValueError(f'{label}: expected one residue template, found {len(matches)}')
            matches = [a for a in matches[0].findall('Atom') if a.attrib['name'] == atom_name]
            if len(matches) != 1:
                raise ValueError(f'{label}: expected one matching XML atom, found {len(matches)}')
            atom = matches[0]
            atom_type = atom.attrib['type']
            candidates = []
            for force in forces:
                residue_charge = any(a.attrib.get('name') == 'charge'
                                    for a in force.findall('UseAttributeFromResidue'))
                rules = [rule for rule in force.findall('Atom')
                        if ('type' in rule.attrib and rule.attrib['type'] == atom_type)
                        or ('class' in rule.attrib and rule.attrib['class'] == types.get(atom_type))]
                for rule in rules:
                    source = atom if residue_charge else rule
                    if 'charge' not in source.attrib:
                        raise ValueError(f'{label}: missing charge attribute')
                    candidates.append(float(source.attrib['charge']))
            if not candidates or len(set(candidates)) != 1:
                raise ValueError(f'{label}: missing or conflicting NonbondedForce charges {candidates}')
            if not math.isfinite(candidates[0]):
                raise ValueError(f'{label}: charge must be finite')
            charges.append(candidates[0])
        return {'indices': list(range(start, stop)), 'charges': charges}

