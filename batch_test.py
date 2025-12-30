#!/usr/bin/env python3
import subprocess
import os
import sys

BASE = "/home/kevin/scuda"
os.chdir(BASE)

samples = [
    "cuda-samples/Samples/0_Introduction/vectorAdd/vectorAdd",
    "cuda-samples/Samples/0_Introduction/matrixMul/matrixMul",
    "cuda-samples/Samples/0_Introduction/asyncAPI/asyncAPI",
    "cuda-samples/Samples/0_Introduction/clock/clock",
    "cuda-samples/Samples/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics",
    "cuda-samples/Samples/0_Introduction/simpleCallback/simpleCallback",
    "cuda-samples/Samples/0_Introduction/mergeSort/mergeSort",
    "cuda-samples/Samples/0_Introduction/simpleVoteIntrinsics/build/simpleVoteIntrinsics",
    "cuda-samples/Samples/0_Introduction/simpleAssert/build/simpleAssert",
    "cuda-samples/Samples/0_Introduction/cudaOpenMP/build/cudaOpenMP",
    "cuda-samples/Samples/0_Introduction/fp16ScalarProduct/build/fp16ScalarProduct",
    "cuda-samples/Samples/0_Introduction/simpleCooperativeGroups/build/simpleCooperativeGroups",
    "cuda-samples/Samples/0_Introduction/simpleMultiCopy/build/simpleMultiCopy",
    "cuda-samples/Samples/0_Introduction/simpleHyperQ/build/simpleHyperQ",
    "cuda-samples/Samples/0_Introduction/matrixMulDynlinkJIT/build/matrixMulDynlinkJIT",
    "cuda-samples/Samples/0_Introduction/simpleAWBarrier/build/simpleAWBarrier",
    "cuda-samples/Samples/0_Introduction/simpleCubemapTexture/build/simpleCubemapTexture",
    "cuda-samples/Samples/0_Introduction/simpleLayeredTexture/build/simpleLayeredTexture",
    "cuda-samples/Samples/0_Introduction/simpleStreams/simpleStreams",
    "cuda-samples/Samples/0_Introduction/simpleOccupancy/build/simpleOccupancy",
    "cuda-samples/Samples/1_Utilities/deviceQuery/deviceQuery",
    "cuda-samples/Samples/2_Concepts_and_Techniques/reduction/build/reduction",
    "cuda-samples/Samples/2_Concepts_and_Techniques/scan/build/scan",
    "cuda-samples/Samples/2_Concepts_and_Techniques/histogram/build/histogram",
    "cuda-samples/Samples/2_Concepts_and_Techniques/scalarProd/build/scalarProd",
    "cuda-samples/Samples/2_Concepts_and_Techniques/sortingNetworks/build/sortingNetworks",
    "cuda-samples/Samples/2_Concepts_and_Techniques/convolutionSeparable/build/convolutionSeparable",
    "cuda-samples/Samples/2_Concepts_and_Techniques/inlinePTX/build/inlinePTX",
    "cuda-samples/Samples/2_Concepts_and_Techniques/threadFenceReduction/build/threadFenceReduction",
    "cuda-samples/Samples/2_Concepts_and_Techniques/reductionMultiBlockCG/build/reductionMultiBlockCG",
    "cuda-samples/Samples/3_CUDA_Features/simpleCudaGraphs/build/simpleCudaGraphs",
    "cuda-samples/Samples/3_CUDA_Features/simpleTemplates/build/simpleTemplates",
    "cuda-samples/Samples/3_CUDA_Features/newdelete/build/newdelete",
    "cuda-samples/Samples/3_CUDA_Features/globalToShmemAsyncCopy/build/globalToShmemAsyncCopy",
    "cuda-samples/Samples/3_CUDA_Features/warpAggregatedAtomicsCG/build/warpAggregatedAtomicsCG",
    "cuda-samples/Samples/3_CUDA_Features/StreamPriorities/build/StreamPriorities",
    "cuda-samples/Samples/3_CUDA_Features/binaryPartitionCG/build/binaryPartitionCG",
    "cuda-samples/Samples/5_Domain_Specific/BlackScholes/build/BlackScholes",
    "cuda-samples/Samples/5_Domain_Specific/fastWalshTransform/build/fastWalshTransform",
    "cuda-samples/Samples/5_Domain_Specific/SobolQRNG/build/SobolQRNG",
    "cuda-samples/Samples/5_Domain_Specific/binomialOptions/build/binomialOptions",
    "cuda-samples/Samples/5_Domain_Specific/quasirandomGenerator/build/quasirandomGenerator",
    "cuda-samples/Samples/5_Domain_Specific/dwtHaar1D/build/dwtHaar1D",
    "cuda-samples/Samples/5_Domain_Specific/dxtc/build/dxtc",
    "cuda-samples/Samples/5_Domain_Specific/stereoDisparity/build/stereoDisparity",
    "cuda-samples/Samples/5_Domain_Specific/FDTD3d/build/FDTD3d",
    "cuda-samples/Samples/6_Performance/transpose/build/transpose",
    "cuda-samples/Samples/6_Performance/alignedTypes/build/alignedTypes",
    "cuda-samples/Samples/6_Performance/LargeKernelParameter/build/LargeKernelParameter",
]

env = os.environ.copy()
env["SCUDA_SERVER"] = "127.0.0.1"
env["LD_PRELOAD"] = "/home/kevin/scuda/build/libscuda_13.1.so"
env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/drivers/nvgbdi.inf_amd64_1e259c259f0f0abf:/usr/lib/wsl/lib:" + env.get("LD_LIBRARY_PATH", "")

passed = 0
failed = 0
crashed = 0
skipped = 0

for sample in samples:
    name = os.path.basename(sample)
    full_path = os.path.join(BASE, sample)
    if not os.path.exists(full_path):
        # Try without /build/
        alt = full_path.replace("/build/", "/")
        if os.path.exists(alt):
            full_path = alt
        else:
            skipped += 1
            continue

    try:
        result = subprocess.run(
            [full_path],
            env=env,
            capture_output=True,
            timeout=30,
            cwd=os.path.dirname(full_path)
        )
        output = result.stdout.decode('utf-8', errors='replace') + result.stderr.decode('utf-8', errors='replace')

        if result.returncode in (139, 134) or "Segmentation fault" in output:
            print(f"CRASH: {name}")
            crashed += 1
        elif any(x in output.lower() for x in ["pass", "passed", "test passed", "completed"]):
            print(f"PASS:  {name}")
            passed += 1
        elif result.returncode == 0:
            print(f"OK:    {name}")
            passed += 1
        else:
            print(f"FAIL:  {name} (exit {result.returncode})")
            failed += 1
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {name}")
        failed += 1
    except Exception as e:
        print(f"ERROR: {name} - {e}")
        failed += 1

print()
print("=== Summary ===")
print(f"PASS:    {passed}")
print(f"FAIL:    {failed}")
print(f"CRASH:   {crashed}")
print(f"SKIPPED: {skipped}")
print(f"Total:   {passed + failed + crashed}")
