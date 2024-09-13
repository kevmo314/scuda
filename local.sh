#!/bin/bash

libscuda_path="$(pwd)/libscuda.so"
client_path="$(pwd)/client.cu"

build() {
  echo "building client..."

  if [[ "$(uname)" == "Linux" ]]; then
    nvcc -shared -Xcompiler -fPIC -o $libscuda_path $client_path -lcudart
  else
    echo "No compiler options set for os "$(uname)""
  fi

  if [ ! -f "$libscuda_path" ]; then
      echo "libscuda.so not found. build may have failed."
      exit 1
  fi
}

server() {
  echo "starting server..."
}

run() {
  build


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