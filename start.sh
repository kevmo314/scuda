#!/bin/bash

echo "Connecting to SCUDA server at: $SCUDA_SERVER"
echo "Using scuda binary at path: $libscuda_path"

LD_PRELOAD="$libscuda_path" python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())"