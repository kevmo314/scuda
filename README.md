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

## Local development

Building the binaries requires running codegen first. Scuda codegen reads the cuda dependency header files in order to generate rpc calls.

To ensure codegen works properly, the proper cuda packages need to be installed on your OS. Take a look at our [Dockerfile](./Dockerfile.build) to see an example.

Codegen requires cuBLAS, cuDNN, NVML, etc:

```py
cudnn_graph_header = find_header_file("cudnn_graph.h")
cudnn_ops_header   = find_header_file("cudnn_ops.h")
cuda_header        = find_header_file("cuda.h")
cublas_header      = find_header_file("cublas_api.h")
cudart_header      = find_header_file("cuda_runtime_api.h")
annotations_header = find_header_file("annotations.h")
```

### Run codegen

```bash
cd codegen && python3 ./codegen.py
```

Ensure there are no errors in the output of the codegen.

### Run cmake

```sh
cmake .
cmake --build .
```

Cmake will generate a server and a client file depending on your cuda version.

Example:
`libscuda_12_0.so`, `server_12_0.so`

It's required to run scuda server before initiating client commands.

```sh
./local.sh server
```

The above command will grep for the generated libscuda + server files. You can also run the binaries directly.


```sh
./server_12_0.so
```

If successful, the server will start:

```bash
Server listening on port 14833...
```

## Running the client

Scuda requires you to preload the libscuda binary before executing any cuda commands.

Once the server above is running:

```sh
# update to your desired IP/port
export SCUDA_SERVER=0.0.0.0

LD_PRELOAD=./libscuda_12_0.so python3 -c "import torch; print(torch.cuda.is_available())"

# or

LD_PRELOAD=./libscuda_12_0.so nvidia-smi
```

You can also use the local shell script to run your commands.

```
./local.sh run
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
