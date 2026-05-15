#!/usr/bin/env python3
import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone


DEFAULT_UUIDS = {
    "cublas_init_f8cf": "f8cff95121468b4eb9e2fb469e7c0dd9",
    "cublas_cufft_2131": "21318c60971432488ca641ff7324c8f2",
    "cudart_interface": "6bd5fb6c5bf4e74a8987d93912fd9df9",
    "tools_tls": "42d85a8123f6cb478298f6e78a3aecdc",
    "tools_runtime_callbacks": "a094798c2e742e7493f20800200c0a66",
    "context_local_storage": "c693336e1121df11a8c368f355d89593",
    "context_checks": "263e88607cd2614392f6bbd5006dfa7e",
    "integrity_check": "d4082055bde6704b8d34ba123c66e1f2",
}


TRACE_UUID_RE = re.compile(
    r"SCUDA cuGetExportTable requested UUID:((?: [0-9a-fA-F]{2}){16})"
)


class CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


class DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def clean_uuid(value):
    value = value.replace("-", "").replace(":", "").replace(" ", "").lower()
    if len(value) != 32:
        raise ValueError(f"UUID must be 16 bytes/32 hex chars: {value}")
    return value


def uuid_from_trace_line(line):
    match = TRACE_UUID_RE.search(line)
    if not match:
        return None
    return clean_uuid(match.group(1))


def load_trace_uuids(paths):
    uuids = {}
    for path in paths:
        with open(path, "r", errors="replace") as f:
            for line in f:
                raw = uuid_from_trace_line(line)
                if raw:
                    uuids.setdefault(f"trace_{raw}", raw)
    return uuids


def sha256_file(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def dladdr(libdl, address):
    info = DlInfo()
    ok = libdl.dladdr(ctypes.c_void_p(address), ctypes.byref(info))
    if not ok:
        return {}
    result = {}
    if info.dli_fname:
        result["object"] = info.dli_fname.decode(errors="replace")
    if info.dli_fbase:
        result["object_base"] = info.dli_fbase
    if info.dli_sname:
        result["symbol"] = info.dli_sname.decode(errors="replace")
    if info.dli_saddr:
        result["symbol_addr"] = info.dli_saddr
    return result


def read_bytes(path, offset, size):
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read(size)
    except OSError:
        return b""


def objdump_one(path, offset, size):
    try:
        out = subprocess.check_output(
            [
                "objdump",
                "-d",
                "-Mintel",
                f"--start-address=0x{offset:x}",
                f"--stop-address=0x{offset + size:x}",
                path,
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    lines = []
    for line in out.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return lines[-20:]


def resolve_library_path(name):
    found = ctypes.util.find_library(name)
    if found:
        return found
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Interrogate cuGetExportTable on a real libcuda and emit JSON metadata."
    )
    parser.add_argument("--libcuda", default="libcuda.so.1")
    parser.add_argument("--uuid", action="append", default=[],
                        help="name=hex or hex. May be repeated.")
    parser.add_argument("--trace-log", action="append", default=[],
                        help="SCUDA_TRACE log to parse for requested UUIDs.")
    parser.add_argument("--no-default-uuids", action="store_true",
                        help="Only query UUIDs from --uuid/--trace-log.")
    parser.add_argument("--library", action="append", default=[],
                        help="Additional CUDA library path to fingerprint.")
    parser.add_argument("--max-slots", type=int, default=160)
    parser.add_argument("--bytes-per-slot", type=int, default=32)
    parser.add_argument("--disassemble", action="store_true")
    parser.add_argument("--emit-private-env", action="store_true",
                        help="Print SCUDA_PRIVATE_EXPORT_TABLES for trusted tables to stderr.")
    args = parser.parse_args()

    requested = {} if args.no_default_uuids else dict(DEFAULT_UUIDS)
    requested.update(load_trace_uuids(args.trace_log))
    for item in args.uuid:
        if "=" in item:
            name, raw = item.split("=", 1)
        else:
            raw = item
            name = clean_uuid(raw)
        requested[name] = clean_uuid(raw)

    libcuda = ctypes.CDLL(args.libcuda)
    libdl = ctypes.CDLL(resolve_library_path("dl"))
    libdl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    libdl.dladdr.restype = ctypes.c_int

    libcuda.cuInit.argtypes = [ctypes.c_uint]
    libcuda.cuInit.restype = ctypes.c_int
    libcuda.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    libcuda.cuDriverGetVersion.restype = ctypes.c_int
    libcuda.cuGetExportTable.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(CUuuid),
    ]
    libcuda.cuGetExportTable.restype = ctypes.c_int

    init_rc = libcuda.cuInit(0)
    driver_version = ctypes.c_int()
    driver_version_rc = libcuda.cuDriverGetVersion(ctypes.byref(driver_version))

    get_export_addr = ctypes.cast(libcuda.cuGetExportTable, ctypes.c_void_p).value
    cuda_info = dladdr(libdl, get_export_addr)
    libcuda_path = cuda_info.get("object", args.libcuda)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "libcuda": {
            "requested": args.libcuda,
            "path": libcuda_path,
            "sha256": sha256_file(libcuda_path),
            "cuInit": init_rc,
            "cuDriverGetVersion_rc": driver_version_rc,
            "cuDriverGetVersion": driver_version.value,
        },
        "libraries": [],
        "tables": {},
    }
    for path in args.library:
        result["libraries"].append({
            "path": path,
            "sha256": sha256_file(path),
        })

    for name, raw in requested.items():
        uuid = CUuuid((ctypes.c_ubyte * 16).from_buffer_copy(bytes.fromhex(raw)))
        table_ptr = ctypes.c_void_p()
        rc = libcuda.cuGetExportTable(ctypes.byref(table_ptr), ctypes.byref(uuid))
        table = {
            "uuid": raw,
            "cuGetExportTable": rc,
            "address": table_ptr.value,
            "slots": [],
        }
        if rc == 0 and table_ptr.value:
            arr = ctypes.cast(table_ptr, ctypes.POINTER(ctypes.c_void_p))
            byte_size = arr[0] or 0
            table["byte_size"] = byte_size
            if byte_size > 0 and byte_size % ctypes.sizeof(ctypes.c_void_p) == 0:
                slot_count = byte_size // ctypes.sizeof(ctypes.c_void_p)
                trusted_size = slot_count <= args.max_slots
            else:
                slot_count = args.max_slots
                trusted_size = False
            if not trusted_size:
                table["byte_size_is_trusted"] = False
                slot_count = args.max_slots
            else:
                table["byte_size_is_trusted"] = True
            table["slot_count"] = slot_count
            for slot in range(slot_count):
                value = arr[slot]
                entry = {"slot": slot, "address": value}
                if slot == 0:
                    table["slots"].append(entry)
                    continue
                if value:
                    info = dladdr(libdl, value)
                    entry.update(info)
                    object_path = info.get("object")
                    object_base = info.get("object_base")
                    if object_path and object_base:
                        offset = value - object_base
                        entry["object_offset"] = offset
                        code = read_bytes(object_path, offset, args.bytes_per_slot)
                        entry["code_sha256_32"] = hashlib.sha256(code).hexdigest()
                        entry["code_prefix"] = code.hex()
                        if args.disassemble:
                            entry["disassembly"] = objdump_one(
                                object_path, offset, args.bytes_per_slot
                            )
                table["slots"].append(entry)
        result["tables"][name] = table

    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    if args.emit_private_env:
        entries = []
        for table in result["tables"].values():
            if table.get("cuGetExportTable") == 0 and table.get("byte_size_is_trusted"):
                entries.append(f"{table['uuid']}:{table['byte_size']}")
        print("SCUDA_PRIVATE_EXPORT_TABLES=" + ",".join(entries), file=sys.stderr)


if __name__ == "__main__":
    main()
