#!/bin/bash

echo "Connecting to LUPINE server at: $LUPINE_SERVER"
echo "Using lupine binary at path: $client_lib_path"

if [[ "$1" == "torch" ]]; then
    echo "Running torch example..."
    LD_PRELOAD="$client_lib_path" python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
elif [[ "$1" == "cublas" ]]; then
    echo "Running cublas example..."

    LD_PRELOAD="$client_lib_path" /matrixMulCUBLAS
elif [[ "$1" == "unified" ]]; then
    echo "Running cublas example..."

    LD_PRELOAD="$client_lib_path" /cublas_unified.o
else
    echo "Unknown option: $1. Please specify one of: torch | cublas | unified ."
fi