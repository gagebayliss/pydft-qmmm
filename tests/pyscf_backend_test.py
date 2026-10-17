"""Backend dispatch and device array transfer checks without CUDA."""
import sys
from types import ModuleType

import numpy as np
import pytest

pytest.importorskip("pyscf")
from pydft_qmmm.interfaces.pyscf import pyscf_backend as backend


def test_gpu_dispatch_uses_gpu_qmmm(monkeypatch):
    from pydft_qmmm.interfaces.pyscf.pyscf_embedding import add_finite_embedding
    package = ModuleType("gpu4pyscf")
    calls = []
    solver = object()
    module = ModuleType("gpu4pyscf.dft")
    module.RKS = lambda mol, xc: (mol, xc)
    qmmm = ModuleType("gpu4pyscf.qmmm")
    qmmm.add_mm_charges = lambda method, positions, charges, unit: calls.append(
        (method, positions, charges, unit),
    ) or solver
    monkeypatch.setitem(sys.modules, "gpu4pyscf", package)
    monkeypatch.setitem(sys.modules, "gpu4pyscf.dft", module)
    monkeypatch.setitem(sys.modules, "gpu4pyscf.qmmm", qmmm)
    assert backend.load_backend("gpu") is package
    assert backend.build_solver(package, "rks", "molecule", "PBE") == ("molecule", "PBE")
    system = ModuleType("system")
    system.positions = np.arange(12).reshape(4, 3)
    system.charges = np.arange(4)*.1
    assert add_finite_embedding(package, solver, system, (1, 3)) is solver
    np.testing.assert_array_equal(calls[0][1], system.positions[[1, 3]])
    np.testing.assert_array_equal(calls[0][2], system.charges[[1, 3]])
    assert calls[0][3] == "Angstrom"


def test_tagged_device_arrays_transfer_explicitly(monkeypatch):
    host = np.eye(2)
    # GPU4PySCF tags CuPy arrays with subclasses in its own modules.
    ndarray = type("ndarray", (), {"__module__": "cupy", "get": lambda self: host})
    tagged = type("TaggedArray", (ndarray,), {"__module__": "gpu4pyscf.lib"})()
    cupy = ModuleType("cupy")
    cupy.asarray = lambda array: ("device", array)
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    assert backend.to_numpy(tagged) is host
    assert backend.to_numpy(host) is host
    assert backend.to_like(host, tagged)[0] == "device"


@pytest.mark.parametrize("method,functional,spin", [("rhf", "PBE", 0),
                        ("rks", None, 0), ("rhf", None, 2), ("unknown", None, 0)])
def test_invalid_method_combination(method, functional, spin):
    with pytest.raises(ValueError):
        backend.resolve_method(method, functional, spin)
