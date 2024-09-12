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
