#!/bin/bash

echo "Connecting to SCUDA server at: $SCUDA_SERVER"

LD_PRELOAD="$libscuda_path" python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())"