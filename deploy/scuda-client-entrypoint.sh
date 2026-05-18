#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  cat >&2 <<'EOF'
usage: scuda-client <command> [args...]

Runs the command with the SCUDA libcuda shim preloaded.
Set SCUDA_SERVER=<host:port> to point the client at a running SCUDA server.
EOF
  exit 64
fi

scuda_lib="${SCUDA_LIB:-/opt/scuda/lib/libcuda.so.1}"

case ":${LD_PRELOAD:-}:" in
  *":${scuda_lib}:"*) ;;
  *) export LD_PRELOAD="${scuda_lib}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
esac

case ":${LD_LIBRARY_PATH:-}:" in
  *":/opt/scuda/lib:"*) ;;
  *) export LD_LIBRARY_PATH="/opt/scuda/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

exec "$@"
