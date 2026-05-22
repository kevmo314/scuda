"""PyTorch adapter helpers for SCUDA.

The adapter returns ordinary ``torch.device("cuda:N")`` objects. PyTorch stays
on its built-in CUDA dispatch path while SCUDA handles CUDA driver calls below
it.
"""

from __future__ import annotations

import os
import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PORT = 14833


class ScudaError(RuntimeError):
    """Raised when the SCUDA adapter cannot select a usable device."""


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ScudaError("PyTorch is required to use SCUDA devices.") from exc
    return torch


def _cuda_initialized() -> bool:
    try:
        return bool(_torch().cuda.is_initialized())
    except ScudaError:
        return False


def _require_mutable_config() -> None:
    if _cuda_initialized():
        raise ScudaError("connect to SCUDA before PyTorch initializes CUDA")


def _normalize_server(host: str, port: int | None = None) -> str:
    host = str(host).strip()
    if not host:
        raise ScudaError("host must not be empty")
    if port is not None:
        return f"{host}:{int(port)}"
    if host.startswith("[") and "]:" in host:
        return host
    if host.count(":") == 1:
        return host
    return f"{host}:{DEFAULT_PORT}"


def _normalize_hosts(host: str | Sequence[str], port: int | None = None) -> tuple[str, ...]:
    if isinstance(host, str):
        servers = (_normalize_server(host, port),)
    else:
        servers = tuple(_normalize_server(item, port) for item in host)
    if not servers:
        raise ScudaError("at least one SCUDA host is required")
    if len(set(servers)) != len(servers):
        raise ScudaError("SCUDA hosts must be unique")
    return servers


def _set_server_env(servers: Sequence[str]) -> None:
    os.environ["SCUDA_SERVER"] = ",".join(servers)


def _default_libcuda() -> Path | None:
    override = os.environ.get("SCUDA_LIBCUDA")
    if override:
        return Path(override)
    repo_candidate = Path(__file__).resolve().parents[2] / "build" / "libcuda.so.1"
    if repo_candidate.exists():
        return repo_candidate
    return None


def _load_libcuda(path: str | os.PathLike[str] | None) -> None:
    libcuda = Path(path) if path is not None else _default_libcuda()
    if libcuda is None:
        return
    if not libcuda.exists():
        raise ScudaError(f"SCUDA libcuda does not exist: {libcuda}")
    ctypes.CDLL(str(libcuda), mode=ctypes.RTLD_GLOBAL)


def _servers_from_env() -> tuple[str, ...]:
    value = os.environ.get("SCUDA_SERVER", "")
    return tuple(server.strip() for server in value.split(",") if server.strip())


def _cuda_device(index: int, *, require_available: bool = False) -> Any:
    torch = _torch()
    index = int(index)
    if require_available:
        count = int(torch.cuda.device_count())
        if count <= 0:
            raise ScudaError(
                "PyTorch does not see any CUDA devices. Check that the SCUDA "
                "client library is selected and SCUDA_SERVER is configured."
            )
        if index < 0 or index >= count:
            raise ScudaError(f"CUDA device index {index} is out of range for {count} devices")
    return torch.device("cuda", index)


@dataclass
class Session:
    """A process-local SCUDA connection declaration."""

    servers: tuple[str, ...]
    require_available: bool = False
    libcuda: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        if not self.servers:
            raise ScudaError("at least one SCUDA host is required")

    def __enter__(self) -> "Session":
        _require_mutable_config()
        self._previous_server = os.environ.get("SCUDA_SERVER")
        configured = _servers_from_env()
        if configured and configured != self.servers:
            raise ScudaError(
                "SCUDA_SERVER is already configured differently; start a new "
                "process or pass the same hosts to scuda.connect()."
            )
        if not configured:
            _set_server_env(self.servers)
        _load_libcuda(self.libcuda)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if not _cuda_initialized():
            self._restore_env()
        return False

    def _restore_env(self) -> None:
        if getattr(self, "_previous_server", None) is None:
            os.environ.pop("SCUDA_SERVER", None)
        else:
            os.environ["SCUDA_SERVER"] = self._previous_server
    def devices(self, *, require_available: bool | None = None) -> list[Any]:
        """Return all declared SCUDA GPUs as ``torch.device("cuda:N")``."""

        check = self.require_available if require_available is None else require_available
        return [
            _cuda_device(index, require_available=check)
            for index in range(len(self.servers))
        ]

    def device(self, index: int = 0, *, require_available: bool | None = None) -> Any:
        """Return one declared SCUDA GPU as ``torch.device("cuda:N")``."""

        index = int(index)
        if index < 0 or index >= len(self.servers):
            raise ScudaError(
                f"SCUDA device index {index} is out of range for {len(self.servers)} hosts"
            )
        check = self.require_available if require_available is None else require_available
        return _cuda_device(index, require_available=check)


def connect(
    *,
    host: str | Sequence[str],
    port: int | None = None,
    require_available: bool = False,
    libcuda: str | os.PathLike[str] | None = None,
) -> Session:
    """Create a SCUDA session for one or more remote GPU hosts.

    Use the session before any PyTorch CUDA operation:

    ``with scuda.connect(host=["a:14833", "b:14833"]) as s:``

    ``s.devices()`` then returns ``[torch.device("cuda:0"), torch.device("cuda:1")]``.
    """

    return Session(
        servers=_normalize_hosts(host, port),
        require_available=require_available,
        libcuda=libcuda,
    )


def devices(*, require_available: bool = True) -> list[Any]:
    """Return devices for the current ``SCUDA_SERVER`` environment."""

    servers = _servers_from_env()
    if not servers:
        raise ScudaError("SCUDA_SERVER is not configured")
    return [
        _cuda_device(index, require_available=require_available)
        for index in range(len(servers))
    ]


def device(index: int = 0, *, require_available: bool = True) -> Any:
    """Return one device for the current ``SCUDA_SERVER`` environment."""

    return devices(require_available=require_available)[int(index)]


def servers() -> tuple[str, ...]:
    """Return configured SCUDA servers from ``SCUDA_SERVER``."""

    return _servers_from_env()


def is_configured() -> bool:
    """Return true when ``SCUDA_SERVER`` names at least one server."""

    return bool(servers())


def is_available() -> bool:
    """Return true when PyTorch sees at least one CUDA device."""

    try:
        torch = _torch()
    except ScudaError:
        return False
    return bool(torch.cuda.is_available())


def device_count() -> int:
    """Return PyTorch's CUDA device count."""

    torch = _torch()
    return int(torch.cuda.device_count())


def current_device() -> int:
    """Return PyTorch's current CUDA device index."""

    torch = _torch()
    return int(torch.cuda.current_device())


def synchronize(index: int = 0) -> None:
    """Synchronize a SCUDA-backed CUDA device."""

    torch = _torch()
    torch.cuda.synchronize(_cuda_device(index, require_available=False))


__all__ = [
    "DEFAULT_PORT",
    "ScudaError",
    "Session",
    "connect",
    "current_device",
    "device",
    "device_count",
    "devices",
    "is_available",
    "is_configured",
    "servers",
    "synchronize",
]
