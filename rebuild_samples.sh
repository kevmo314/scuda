#!/bin/bash
# Rebuild CUDA samples with dynamic linking for SCUDA compatibility

CUDA_SAMPLES=/home/kevin/cuda-samples
SCUDA_DIR=/home/kevin/scuda

cd "$CUDA_SAMPLES"

# List of samples to rebuild
SAMPLES=(
    "Samples/0_Introduction/vectorAdd"
    "Samples/0_Introduction/matrixMul"
    "Samples/0_Introduction/asyncAPI"
    "Samples/0_Introduction/clock"
    "Samples/0_Introduction/mergeSort"
    "Samples/0_Introduction/simpleStreams"
    "Samples/0_Introduction/simpleCallback"
    "Samples/0_Introduction/simpleAtomicIntrinsics"
    "Samples/0_Introduction/simpleOccupancy"
    "Samples/0_Introduction/simpleVoteIntrinsics"
    "Samples/0_Introduction/simpleAssert"
    "Samples/0_Introduction/simpleCooperativeGroups"
    "Samples/1_Utilities/deviceQuery"
)

for sample_path in "${SAMPLES[@]}"; do
    sample=$(basename "$sample_path")
    if [ -d "$sample_path" ]; then
        echo "Building $sample..."
        cd "$sample_path"
        # Find all .cu files and compile with dynamic linking
        cu_files=$(ls *.cu 2>/dev/null)
        cpp_files=$(ls *.cpp 2>/dev/null)
        if [ -n "$cu_files" ]; then
            nvcc -cudart=shared -o "${sample}_shared" $cu_files $cpp_files -I../../../Common 2>&1 || echo "Failed: $sample"
        fi
        cd "$CUDA_SAMPLES"
    else
        echo "Skipping $sample (directory not found)"
    fi
done

echo ""
echo "Done. Run samples with:"
echo "  SCUDA_SERVER=127.0.0.1 LD_PRELOAD=$SCUDA_DIR/build/libscuda_13.1.so ./<sample>_shared"
