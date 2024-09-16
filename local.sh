#!/bin/bash

libscuda_path="$(pwd)/libscuda.so"
src_dir="$(pwd)/src"
server_out_path="$(pwd)/server.so"

# Find all .cu files in src directory dynamically
client_sources=$(find "$src_dir" -name "*.cu" ! -name "server.cu")
server_sources=$(find "$src_dir" -name "server.cu")

build() {
  echo "building client..."

  if [[ "$(uname)" == "Linux" ]]; then
    nvcc -Xcompiler -fPIC -shared -o "$libscuda_path" $client_sources 
  else
    echo "No compiler options set for os $(uname)"
  fi

  if [ ! -f "$libscuda_path" ]; then
    echo "libscuda.so not found. build may have failed."
    exit 1
  fi
}

server() {
  echo "building server..." 

  if [[ "$(uname)" == "Linux" ]]; then
    nvcc -o "$server_out_path" $server_sources -lnvidia-ml
  else
    echo "No compiler options set for os $(uname)"
  fi

  echo "starting server... $server_out_path"

  "$server_out_path"
}

run() {
  build

  LD_PRELOAD="$libscuda_path" nvidia-smi --query-gpu=name --format=csv
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
