.. _`sec:getting_started`:

===============
Getting Started
===============

Installation
============

PyDFT-QMMM can be installed directly from github using ``pip``:

.. code-block:: bash

    $ python -m pip install git+https://github.com/johnppederson/pydft-qmmm

Alternatively, you can clone the repository and install using ``pip``:

.. code-block:: bash

    $ git clone https://github.com/johnppederson/pydft-qmmm
    $ cd pydft-qmmm
    $ pip install .

Examples
========

Several cases are provided in the example suite.  The CLI and Python API
are demonstrated in :example:`0`.  Enhanced sampling functionality with
Plumed is demonstrated in :example:`1`. Using AMBER or GROMACS forcefield
parameter files is demonstrated in :example:`2`.

Templates
=========

Templates for writing third-party plugins and interfaces are provided in
the templates folder in the project root directory.

Temperature and pressure coupling
================================

``AndersenThermostat`` and ``MonteCarloBarostat`` are integrator plugins.
For example, given a periodic ``system`` and a ``hamiltonian``:

.. code-block:: python

    from pydft_qmmm import Simulation, VerletIntegrator
    from pydft_qmmm.plugins import AndersenThermostat, MonteCarloBarostat

    calculator = hamiltonian.build_calculator(system)
    integrator = VerletIntegrator(timestep=1.0)  # fs
    simulation = Simulation(
        system=system,
        integrator=integrator,
        calculator=calculator,
        plugins=[
            MonteCarloBarostat(
                calculator=calculator, pressure=1.0, temperature=300.0,
                interval=25, volume_step=10.0, seed=42,
            ),
            AndersenThermostat(
                temperature=300.0, collision_frequency=0.001,
                query="all", seed=43,
            ),
        ],
    )

Temperature is in K, pressure in bar, collision frequency in inverse fs,
and ``volume_step`` in cubic angstroms. Each plugin has an independent
random generator. Zero collision frequency disables thermal collisions;
massless sites are never resampled. The thermostat uses collision probability
``1 - exp(-collision_frequency * timestep)`` and samples Maxwell–Boltzmann
velocities. These algorithms follow the `OpenMM theory guide`_.

The barostat requires the same system as its constructor's calculator and
evaluates the full potential energy for trial moves. It attempts a symmetric
volume change every ``interval`` calls, translates residue centroids while
preserving internal geometry, and uses the molecular volume Jacobian in its
Metropolis acceptance rule. Accepted moves refresh forces before integration;
rejected moves and failed evaluations restore box, coordinates, velocities,
and forces. The pressure plugin requires a separate thermostat at the same
temperature, and its volume step is fixed rather than automatically tuned.

Each residue must represent one whole molecule, including any virtual sites,
with coordinates unwrapped within that molecule. Molecules spanning multiple
residues and fixed atoms are unsupported for pressure coupling. Calculators
must update all energy terms when the box changes. History-dependent biases
and calculator plugins that change state irreversibly during evaluation are
unsupported, since trial calculations cannot undo their internal state.

Register the barostat before other user integrator plugins. Thermal collisions
act on input velocities before integration so SETTLE and Stationary can
constrain the resulting step (Stationary supports temperature coupling alone).
Simulation's automatic Virtual plugin updates virtual sites after integration.
Box updates are propagated to both OpenMM contexts; other backends must also
honor observed box updates.

.. _OpenMM theory guide: https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html
