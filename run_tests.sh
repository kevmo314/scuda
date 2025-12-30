#!/bin/bash
cd /home/kevin/scuda

PASS=0
FAIL=0
CRASH=0

test_sample() {
    local path="$1"
    local name=$(basename "$path")
    local dir=$(dirname "$path")

    cd "$dir" 2>/dev/null || return 1

    output=$(timeout 30 env SCUDA_SERVER=127.0.0.1 \
        LD_PRELOAD=/home/kevin/scuda/build/libscuda_13.1.so \
        LD_LIBRARY_PATH=/usr/lib/wsl/drivers/nvgbdi.inf_amd64_1e259c259f0f0abf:/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
        "./$name" 2>&1)
    code=$?

    if [[ $code -eq 139 ]] || [[ $code -eq 134 ]] || echo "$output" | grep -q "Segmentation fault\|core dumped"; then
        echo "CRASH: $name"
        ((CRASH++))
    elif echo "$output" | grep -aiq "PASS\|passed\|Test passed\|Result = PASS\|completed, returned OK"; then
        echo "PASS:  $name"
        ((PASS++))
    elif [[ $code -eq 0 ]]; then
        echo "OK:    $name"
        ((PASS++))
    else
        echo "FAIL:  $name (exit $code)"
        ((FAIL++))
    fi

    cd /home/kevin/scuda
}

echo "=== SCUDA Regression Tests ==="

samples=(
    "cuda-samples/Samples/0_Introduction/vectorAdd/vectorAdd"
    "cuda-samples/Samples/0_Introduction/matrixMul/matrixMul"
    "cuda-samples/Samples/0_Introduction/asyncAPI/asyncAPI"
    "cuda-samples/Samples/0_Introduction/clock/clock"
    "cuda-samples/Samples/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics"
    "cuda-samples/Samples/0_Introduction/simpleCallback/simpleCallback"
    "cuda-samples/Samples/0_Introduction/mergeSort/mergeSort"
    "cuda-samples/Samples/0_Introduction/simpleVoteIntrinsics/simpleVoteIntrinsics"
    "cuda-samples/Samples/0_Introduction/simpleAssert/simpleAssert"
    "cuda-samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP"
    "cuda-samples/Samples/0_Introduction/fp16ScalarProduct/fp16ScalarProduct"
    "cuda-samples/Samples/0_Introduction/simpleCooperativeGroups/simpleCooperativeGroups"
    "cuda-samples/Samples/0_Introduction/simpleMultiCopy/simpleMultiCopy"
    "cuda-samples/Samples/0_Introduction/simpleHyperQ/simpleHyperQ"
    "cuda-samples/Samples/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT"
    "cuda-samples/Samples/0_Introduction/simpleAWBarrier/simpleAWBarrier"
    "cuda-samples/Samples/0_Introduction/simpleCubemapTexture/build/simpleCubemapTexture"
    "cuda-samples/Samples/0_Introduction/simpleLayeredTexture/build/simpleLayeredTexture"
    "cuda-samples/Samples/0_Introduction/simpleStreams/simpleStreams"
    "cuda-samples/Samples/1_Utilities/deviceQuery/deviceQuery"
    "cuda-samples/Samples/2_Concepts_and_Techniques/reduction/reduction"
    "cuda-samples/Samples/2_Concepts_and_Techniques/scan/scan"
    "cuda-samples/Samples/2_Concepts_and_Techniques/histogram/histogram"
    "cuda-samples/Samples/2_Concepts_and_Techniques/scalarProd/scalarProd"
    "cuda-samples/Samples/2_Concepts_and_Techniques/sortingNetworks/sortingNetworks"
    "cuda-samples/Samples/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable"
    "cuda-samples/Samples/2_Concepts_and_Techniques/inlinePTX/inlinePTX"
    "cuda-samples/Samples/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction"
    "cuda-samples/Samples/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG"
    "cuda-samples/Samples/3_CUDA_Features/simpleCudaGraphs/simpleCudaGraphs"
    "cuda-samples/Samples/3_CUDA_Features/simpleTemplates/simpleTemplates"
    "cuda-samples/Samples/3_CUDA_Features/newdelete/newdelete"
    "cuda-samples/Samples/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy"
    "cuda-samples/Samples/3_CUDA_Features/warpAggregatedAtomicsCG/warpAggregatedAtomicsCG"
    "cuda-samples/Samples/3_CUDA_Features/StreamPriorities/StreamPriorities"
    "cuda-samples/Samples/3_CUDA_Features/binaryPartitionCG/binaryPartitionCG"
    "cuda-samples/Samples/5_Domain_Specific/BlackScholes/build/BlackScholes"
    "cuda-samples/Samples/5_Domain_Specific/fastWalshTransform/fastWalshTransform"
    "cuda-samples/Samples/5_Domain_Specific/SobolQRNG/SobolQRNG"
    "cuda-samples/Samples/5_Domain_Specific/binomialOptions/binomialOptions"
    "cuda-samples/Samples/5_Domain_Specific/quasirandomGenerator/quasirandomGenerator"
    "cuda-samples/Samples/5_Domain_Specific/dwtHaar1D/dwtHaar1D"
    "cuda-samples/Samples/5_Domain_Specific/dxtc/dxtc"
    "cuda-samples/Samples/5_Domain_Specific/stereoDisparity/stereoDisparity"
    "cuda-samples/Samples/5_Domain_Specific/FDTD3d/FDTD3d"
    "cuda-samples/Samples/6_Performance/transpose/transpose"
    "cuda-samples/Samples/6_Performance/alignedTypes/alignedTypes"
    "cuda-samples/Samples/6_Performance/LargeKernelParameter/LargeKernelParameter"
)

for sample in "${samples[@]}"; do
    if [[ -f "$sample" ]] && [[ -x "$sample" ]]; then
        test_sample "$sample"
    fi
done

echo ""
echo "=== Summary ==="
echo "PASS:  $PASS"
echo "FAIL:  $FAIL"
echo "CRASH: $CRASH"
echo "Total: $((PASS + FAIL + CRASH))"
