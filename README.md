# SCUDA: GPU-over-IP

SCUDA is a GPU over IP bridge allowing GPUs on remote machines to be attached
to CPU-only machines.

## Demo

### CUBLAS Matrix Multiplication using Unified Memory

The below demo displays a NVIDIA GeForce RTX 4090 running on a remote machine (right pane).
Left pane is a Mac running a docker container with nvidia utils installed.

The docker container runs this [matrixMulCUBLAS](./test/cublas_unified.cu) example. This example not only uses cuBLAS, but also takes advantage of unified memory.

You can view the docker image used [here](./deploy/Dockerfile.unified).

https://github.com/user-attachments/assets/b2db5d82-f214-41cf-8274-b913c04080f9

You can see a list of some currently working examples in the [test folder](./test/).

## Quick Start

Use the published GHCR images. The examples below pin CUDA 13.1.0 on
Ubuntu 24.04; other published tags use the same
`cuda-<cuda-version>-ubuntu<ubuntu-version>` format.

Run the server on the GPU machine:

```bash
docker run --rm --gpus all -p 14833:14833 \
  ghcr.io/kevmo314/scuda-server:cuda-13.1.0-ubuntu24.04
```

Run the client pointing at that server:

```bash
docker run --rm -it \
  -e SCUDA_SERVER=<server>:14833 \
  ghcr.io/kevmo314/scuda-client:cuda-13.1.0-ubuntu24.04 \
  nvidia-smi
```

Example output from a real run against a remote RTX 4090:

```text
Mon May 18 15:40:46 2026
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.288.01             Driver Version: 590.48.01    CUDA Version: 13.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        On  | 00000000:01:00.0  On |                  Off |
| 30%   52C    P8              22W / 450W |      8MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

Inside the client container, `LD_LIBRARY_PATH=/opt/scuda/lib` is already set,
so CUDA driver users and NVML users such as `nvidia-smi` pick up the unified
SCUDA shim automatically. The real client library is `libscuda.so.1`;
`libcuda.so.1` and `libnvidia-ml.so.1` are compatibility symlinks to it.

## Multi-GPU Across Multiple Servers

The client accepts a comma-separated `SCUDA_SERVER` list. Devices are exposed as
one local ordinal list in server order: all GPUs from the first server, then all
GPUs from the next server, and so on.

Run a server on each GPU machine:

```bash
# on gpu-host-a
docker run --rm --gpus all -p 14833:14833 \
  ghcr.io/kevmo314/scuda-server:cuda-13.1.0-ubuntu24.04

# on gpu-host-b
docker run --rm --gpus all -p 14833:14833 \
  ghcr.io/kevmo314/scuda-server:cuda-13.1.0-ubuntu24.04
```

Point the client at both servers:

```bash
docker run --rm --network host \
  -e SCUDA_SERVER=gpu-host-a:14833,gpu-host-b:14833 \
  ghcr.io/kevmo314/scuda-client:cuda-13.1.0-ubuntu24.04 \
  nvidia-smi -L
```

Expected output lists both remote GPUs:

```text
GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-...)
GPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-...)
```

CUDA driver applications use the same `SCUDA_SERVER` value:

```bash
docker run --rm --network host \
  -e SCUDA_SERVER=gpu-host-a:14833,gpu-host-b:14833 \
  ghcr.io/kevmo314/scuda-client:cuda-13.1.0-ubuntu24.04 \
  ./your_cuda_program
```

Cross-server peer access and device-to-device copies are not implemented yet.
Same-server operations route by handle ownership.

For a specific CUDA version:

```bash
docker pull ghcr.io/kevmo314/scuda-client:cuda-12.4.1-ubuntu22.04
docker pull ghcr.io/kevmo314/scuda-server:cuda-12.4.1-ubuntu22.04
```

## Slow Start for the Skeptics

This path derives a small PyTorch client image from the published SCUDA client
image and runs the `microgpt_train` test against a remote GPU. It is
intentionally explicit so it is easy to see which side is the CPU-only client
and which side owns the GPU.

Create a PyTorch client Dockerfile in the repo root:

```dockerfile
# Dockerfile.pytorch-scuda
FROM ghcr.io/kevmo314/scuda-client:cuda-13.1.0-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages \
    --index-url https://download.pytorch.org/whl/cu130 \
    torch

COPY test/pytorch_scuda_tests.py /opt/scuda/test/pytorch_scuda_tests.py

ENV LD_LIBRARY_PATH=/opt/scuda/lib:${LD_LIBRARY_PATH}

CMD ["python3", "/opt/scuda/test/pytorch_scuda_tests.py", "microgpt_train"]
```

Build it:

```bash
docker build -f Dockerfile.pytorch-scuda -t scuda-pytorch:cuda-13.1 .
```

Run the server on the GPU machine:

```bash
docker run --rm --gpus all -p 14833:14833 \
  ghcr.io/kevmo314/scuda-server:cuda-13.1.0-ubuntu24.04
```

Run the PyTorch client from the CPU-only machine:

```bash
docker run --rm \
  -e SCUDA_SERVER=<server>:14833 \
  scuda-pytorch:cuda-13.1
```

Expected success looks like:

```text
microgpt first_loss=... last_loss=...
microgpt_train: PASS
```

## Local Development

The Dockerfile is the reference build environment. For a local build, install a
CUDA Toolkit and configure CMake:

```bash
cmake -S . -B build
cmake --build build --parallel --target scuda scuda_driver_server
```

Generated sources are checked in. Only rerun codegen when intentionally
refreshing the generated CUDA Driver API wrappers for a header version:

```bash
cd codegen
python3 ./codegen.py
```

The client build emits one unified shim, `build/libscuda.so`, plus loader
compatibility symlinks:

```text
build/libscuda.so
build/libcuda.so.1 -> libscuda.so.1
build/libnvidia-ml.so.1 -> libscuda.so.1
```

Run the server on the GPU machine:

```bash
SCUDA_PORT=14833 ./build/scuda_driver_server
```

Run a local client against it:

```bash
export SCUDA_SERVER=<server>:14833

LD_LIBRARY_PATH="$PWD/build:${LD_LIBRARY_PATH:-}" \
  LD_PRELOAD="$PWD/build/libscuda.so" \
  python3 -c "import torch; print(torch.cuda.is_available())"

LD_LIBRARY_PATH="$PWD/build:${LD_LIBRARY_PATH:-}" \
  LD_PRELOAD="$PWD/build/libscuda.so" \
  nvidia-smi
```

## Motivations

The goal of SCUDA is to enable developers to easily interact with GPUs over a network in order to take advantage of various pools of distributed GPUs. Obviously TCP is slower than traditional methods, but we have plans to minimize performance impact through various methods.

### Some use cases / motivations:

1. **Local testing** - For testing purposes, the latency added by TCP is acceptable, as the goal is to verify compatibility and performance rather than achieving the lowest latency. The remote GPU can still fully accelerate the application, allowing a developer to run tests they otherwise couldn’t on their local setup.

2. **Aggregated GPU pools** - The goal is to centralize GPU management and resource allocation, making it easier to deploy and scale containerized applications that need GPU support without worrying about GPU availability. SCUDA will eventually handle capacity management and pooling.

3. **Remote model training** - Developers can train models from their laptops or low-power devices, using GPUs optimized for training without needing to deploy a full VM or move the entire development environment to the remote location.

4. **Remote inferencing** - For remote inferencing, devs can set up their application locally but direct all CUDA calls for model inference to a remote GPU server. The application can thus process large batches of images or video frames using the remote GPU’s acceleration capabilities.

5. **Remote data processing** - Developers can run operations like filtering, joining, and aggregating data directly on the remote GPU, while the results are transferred back over the network. Technically, developers can accelerate matrix multiplication or linear algebra computations on large datasets by offloading these computations to a remote GPU; they can run their scripts locally while utilizing the power of a remote machine.

6. **Remote fine-tuning** - Developers can download a pre-trained model (ex: resnet) and fine-tune it. With SCUDA, training is done remotely using the library to route PyTorch CUDA calls over TCP to a remote GPU, allowing the developer to run the fine-tuning process from their local machine or Jupyter Notebook environment.

## Future goals:

See our [TODO](./TODO.md).

## Prior Art

This project is inspired by some existing proprietary solutions:

- https://www.thundercompute.com/
- https://www.juicelabs.co/
- https://en.wikipedia.org/wiki/RCUDA (That's where SCUDA's name comes from, S is the next letter after R!)

## Benchmarks

todo
