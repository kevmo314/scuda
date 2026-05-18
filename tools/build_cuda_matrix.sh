#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEFAULT_MATRIX=(
  "13.1.0:24.04"
  "13.0.2:24.04"
  "12.9.1:24.04"
  "12.8.1:24.04"
  "12.6.2:24.04"
  "12.5.1:22.04"
  "12.4.1:22.04"
  "11.8.0:22.04"
  "11.7.1:22.04"
)

if [[ $# -gt 0 ]]; then
  matrix=("$@")
else
  matrix=("${DEFAULT_MATRIX[@]}")
fi

for entry in "${matrix[@]}"; do
  cuda_version="${entry%%:*}"
  ubuntu_version="${entry#*:}"
  tag="scuda-builder:cuda-${cuda_version}-ubuntu${ubuntu_version}"

  echo "==> Building $tag"
  docker build \
    -f "$repo_root/Dockerfile.build" \
    --build-arg CUDA_VERSION="$cuda_version" \
    --build-arg UBUNTU_VERSION="$ubuntu_version" \
    -t "$tag" \
    "$repo_root"
done
