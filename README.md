# SCUDA: GPU-over-IP

SCUDA is a GPU over IP bridge allowing GPUs on remote machines to be attached
to CPU-only machines.

## Local development

Make the local dev script executable

```sh
chmod +x local.sh
```

Also helpful to alias this local script in your bash profile.

```sh
alias s='/home/brodey/scuda-latest/local.sh'
```

It's required to run scuda server before initiating client commands.

```sh
s server
```

## Running the client

If the server above is running:

```sh
s run
```

The above will rebuild the client and run nvidia-smi for you.

## Installation

To install SCUDA, run the server binary on the GPU host:

```sh
scuda -l 0.0.0.0:0
```

Then, on the client, run:

```sh
scuda <ip>:<port>
```

## Building from source

```sh
nvcc -shared -o libscuda.so client.c
```

This library can then be preloaded

```sh
LD_PRELOAD=libscuda.so nvidia-smi
```

By default, the client library passes calls through to the client. In other words,
it does not connect to a server. To connect to a server, create a file with the
host you wish to connect to

```
~/.config/scuda/host
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

1. Shared storage allocation - Users likely won't want to sync data over the network every time they connect to a remote GPU. To get around this, a user might upload data to SCUDA once and trigger the dataset(s) to be synced to the remote GPU automatically.

2. GPU management plane - SCUDA will eventually track pools of GPUs so that a user can easily swap between devices, run jobs on multiple devices within the same network, prioritize tasks on more equitable hardware, etc.

3. Future network optimizations - Multiplexing and research into various RDMA technologies will help optimize scuda.

4. etc

## Prior Art

This project is inspired by some existing proprietary solutions:

- https://www.thundercompute.com/
- https://www.juicelabs.co/

## Benchmarks

### Without multiplexing

```bash
strace -T -c -e trace=read,write,open,close python3 -c "import torch; print(torch.cuda.is_available())"
```

```
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 69.12    0.004175           1      2141           read
 26.66    0.001610           1      1157           close
  4.22    0.000255           2        93           write
------ ----------- ----------- --------- --------- ----------------
100.00    0.006040           1      3391           total
```



```bash
strace -T -c -e trace=read,write,open,close python3 -c "
import torch
print('Creating a tensor...')
tensor = torch.zeros(10, 10)
print('Moving tensor to CUDA...')
tensor = tensor.to('cuda:0')
print('Tensor successfully moved to CUDA')
"
```

```
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 71.63    0.005715           2      2253           read
 22.00    0.001755           1      1159           close
  6.37    0.000508           2       231           write
------ ----------- ----------- --------- --------- ----------------
100.00    0.007978           2      3643           total
```