#!/bin/bash

build() {
  echo "rebuilding client..."
  cargo build
}

server() {
  echo "starting server..."
  build

  cargo run --features=disable-preload
}

run() {
  build
  local libscuda_path="$(pwd)/target/debug/libscuda.so"

  echo "Full path to libscuda.so: $libscuda_path"

  if [ ! -f "$libscuda_path" ]; then
      echo "libscuda.so not found. build may have failed."
      exit 1
  fi

  LD_PRELOAD="$libscuda_path" nvidia-smi
}

# Main script logic using a switch case
case "$1" in
  build)
      build
      ;;
  run)
      run
      ;;
  server)
      server
      ;;
  *)
      echo "Usage: $0 {build|run|server}"
      exit 1
      ;;
esac