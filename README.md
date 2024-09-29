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