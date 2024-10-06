#!/bin/bash

libscuda_path="$(pwd)/libscuda.so"
client_path="$(pwd)/client.cpp $(pwd)/codegen/gen_client.cpp"
server_path="$(pwd)/server.cu $(pwd)/codegen/gen_server.cpp"
server_out_path="$(pwd)/server.so"

build() {
  echo "building client..."

  if [[ "$(uname)" == "Linux" ]]; then
    gcc -c -fPIC "$(pwd)/client.cpp" -o "$(pwd)/client.o" -I/usr/local/cuda/include
    gcc -c -fPIC "$(pwd)/codegen/gen_client.cpp" -o "$(pwd)/codegen/gen_client.o" -I/usr/local/cuda/include

    echo "linking client files..."

    gcc -shared -o libscuda.so "$(pwd)/client.o" "$(pwd)/codegen/gen_client.o" -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lspdlog -lfmt

  else
    echo "No compiler options set for os "$(uname)""
  fi

  if [ ! -f "$libscuda_path" ]; then
    echo "libscuda.so not found. build may have failed."
    exit 1
  fi
}

server() {
  echo "building server..." 

  if [[ "$(uname)" == "Linux" ]]; then
    nvcc -o $server_out_path $server_path -lnvidia-ml -lcuda
  else
    echo "No compiler options set for os "$(uname)""
  fi

  echo "starting server... $server_out_path"

  "$server_out_path"
}

ansi_format() {
  case "$1" in
  "pass")
    echo -e "\e[32m   ✓ $2\e[0m"
    ;;
  "fail")
    echo -e "\e[31m    ✗ $2\e[0m"
    ;;
  "emph")
    echo -e "\e[1m$2\e[0m"
    ;;
  *)
    echo "$2"
    ;;
  esac
}

#---- lists tests here ----#
test_cuda_available() {
  pass_message=$1
  fail_message=$2

  output=$(LD_PRELOAD="$libscuda_path" python3 -c "import torch; print(torch.cuda.is_available())" | tail -n 1)

  if [[ "$output" == "True" ]]; then
    ansi_format "pass" "$pass_message"
  else
    ansi_format "fail" "CUDA is not available. Expected True but got [$output]."
  fi
}

test_tensor_to_cuda() {
  output=$(LD_PRELOAD="$libscuda_path" python3 -c "
import torch
print('Creating a tensor...')
tensor = torch.zeros(10, 10)
print('Moving tensor to CUDA...')
tensor = tensor.to('cuda:0')
print('Tensor successfully moved to CUDA')
" | tail -n 1)

  if [[ "$output" == "Tensor successfully moved to CUDA" ]]; then
    ansi_format "pass" "$pass_message"
  else
    ansi_format "fail" "Tensor failed. Got [$output]."
  fi
}

#---- declare test cases ----#
declare -A test_cuda_avail=(
  ["function"]="test_cuda_available"
  ["pass"]="CUDA is available."
)

declare -A test_tensor_to_cuda=(
  ["function"]="test_tensor_to_cuda"
  ["pass"]="Tensor moved to CUDA successfully."
)

#---- assign them to our associative array ----#
tests=("test_cuda_avail" "test_tensor_to_cuda")

test() {
  build

  export SCUDA_SERVER=0.0.0.0

  echo -e "\n\033[1mRunning test(s)...\033[0m"

  for test in "${tests[@]}"; do
    func_name=$(eval "echo \${${test}[function]}")
    pass_message=$(eval "echo \${${test}[pass]}")

    eval "$func_name \"$pass_message\""
done
}

run() {
  build

  export SCUDA_SERVER=0.0.0.0

  LD_PRELOAD="$libscuda_path" python3 -c "
import torch
print('Creating a tensor...')
tensor = torch.zeros(10, 10)
print('Moving tensor to CUDA...')
tensor = tensor.to('cuda:0')
print('Tensor successfully moved to CUDA')
"
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
  test)
    test
    ;;
  *)
    echo "Usage: $0 {build|run|server}"
    exit 1
    ;;
esac