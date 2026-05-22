import importlib
import os
import sys
import types

import pytest


class FakeDevice:
    def __init__(self, kind, index=None):
        self.type = kind
        self.index = index

    def __eq__(self, other):
        return isinstance(other, FakeDevice) and (
            self.type,
            self.index,
        ) == (other.type, other.index)

    def __repr__(self):
        return f"{self.type}:{self.index}"


class FakeCuda:
    def __init__(self):
        self.initialized = False
        self.count = 2
        self.current = 1
        self.synchronized = None

    def is_initialized(self):
        return self.initialized

    def is_available(self):
        return self.count > 0

    def device_count(self):
        return self.count

    def current_device(self):
        return self.current

    def synchronize(self, selected):
        self.synchronized = selected


class FakeTorch(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.cuda = FakeCuda()

    def device(self, kind, index=None):
        return FakeDevice(kind, index)


@pytest.fixture
def scuda_module(monkeypatch):
    fake_torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("SCUDA_SERVER", raising=False)
    monkeypatch.setattr("ctypes.CDLL", lambda *args, **kwargs: None)
    import scuda

    yield importlib.reload(scuda), fake_torch
    importlib.reload(scuda)


def test_connect_sets_env_and_returns_devices(scuda_module):
    scuda, _ = scuda_module

    with scuda.connect(host="host-a") as session:
        assert os.environ["SCUDA_SERVER"] == "host-a:14833"
        assert session.devices() == [FakeDevice("cuda", 0)]
        assert session.device() == FakeDevice("cuda", 0)


def test_connect_loads_explicit_libcuda(scuda_module, monkeypatch, tmp_path):
    scuda, _ = scuda_module
    loaded = []
    libcuda = tmp_path / "libcuda.so.1"
    libcuda.write_bytes(b"")
    monkeypatch.setattr(scuda.ctypes, "CDLL", lambda *args, **kwargs: loaded.append(args))

    with scuda.connect(host="host-a", libcuda=libcuda):
        pass

    assert loaded == [(str(libcuda),)]


def test_connect_accepts_multiple_hosts_in_order(scuda_module):
    scuda, _ = scuda_module

    with scuda.connect(host=["host-a:15000", "host-b:16000"]) as session:
        assert session.servers == ("host-a:15000", "host-b:16000")
        assert session.devices() == [FakeDevice("cuda", 0), FakeDevice("cuda", 1)]
        assert session.device(1) == FakeDevice("cuda", 1)


def test_connect_restores_env_when_cuda_was_not_initialized(scuda_module, monkeypatch):
    scuda, _ = scuda_module

    with scuda.connect(host="host-a"):
        assert os.environ["SCUDA_SERVER"] == "host-a:14833"

    assert "SCUDA_SERVER" not in os.environ


def test_connect_leaves_env_when_cuda_initialized_inside_context(scuda_module):
    scuda, fake_torch = scuda_module

    with scuda.connect(host="host-a"):
        fake_torch.cuda.initialized = True

    assert os.environ["SCUDA_SERVER"] == "host-a:14833"


def test_connect_accepts_matching_preconfigured_env(scuda_module, monkeypatch):
    scuda, _ = scuda_module
    monkeypatch.setenv("SCUDA_SERVER", "host-a:14833")

    with scuda.connect(host="host-a:14833") as session:
        assert session.devices() == [FakeDevice("cuda", 0)]


def test_connect_rejects_different_preconfigured_env(scuda_module, monkeypatch):
    scuda, _ = scuda_module
    monkeypatch.setenv("SCUDA_SERVER", "other:14833")

    with pytest.raises(scuda.ScudaError, match="already configured differently"):
        with scuda.connect(host="host-a:14833"):
            pass


def test_connect_refuses_after_cuda_init(scuda_module):
    scuda, fake_torch = scuda_module
    fake_torch.cuda.initialized = True

    with pytest.raises(scuda.ScudaError, match="before PyTorch initializes CUDA"):
        with scuda.connect(host="host-a"):
            pass


def test_devices_use_current_env(scuda_module, monkeypatch):
    scuda, _ = scuda_module
    monkeypatch.setenv("SCUDA_SERVER", "host-a:14833,host-b:14833")

    assert scuda.devices() == [FakeDevice("cuda", 0), FakeDevice("cuda", 1)]
    assert scuda.device(1) == FakeDevice("cuda", 1)


def test_device_bounds_check(scuda_module):
    scuda, _ = scuda_module

    with scuda.connect(host="host-a") as session:
        with pytest.raises(scuda.ScudaError, match="out of range"):
            session.device(1)


def test_duplicate_hosts_are_rejected(scuda_module):
    scuda, _ = scuda_module

    with pytest.raises(scuda.ScudaError, match="unique"):
        scuda.connect(host=["host-a:14833", "host-a"])
