# scuda Python adapter

This package provides small PyTorch helpers for SCUDA. It intentionally returns
ordinary `torch.device("cuda:N")` objects so PyTorch continues to use its normal
CUDA dispatch path while SCUDA handles CUDA driver/NVML calls underneath.

Declare all SCUDA hosts before any PyTorch CUDA work:

```python
import scuda

with scuda.connect(host="<server>:14833") as s:
    device = s.device()
    model = model.to(device)
```

`connect()` loads the SCUDA `libcuda.so.1` from `../build/libcuda.so.1` when
used from this repository. For an installed package, pass `libcuda=...` or set
`SCUDA_LIBCUDA` if the library lives somewhere else:

```python
with scuda.connect(host="<server>:14833", libcuda="/opt/scuda/libcuda.so.1") as s:
    device = s.device()
```

For multiple SCUDA servers, pass the full host list in one call. The order
defines the CUDA ordinals that PyTorch sees:

```python
import scuda

with scuda.connect(host=["<server-a>:14833", "<server-b>:14833"]) as s:
    gpu0, gpu1 = s.devices()
    model0 = model0.to(gpu0)  # cuda:0
    model1 = model1.to(gpu1)  # cuda:1
```

Do not add a second host after tensors have already been moved to the first one.
SCUDA opens connections from `SCUDA_SERVER` when CUDA first initializes, and
later changes to `SCUDA_SERVER` are not picked up by the current process.

Exiting the context restores `SCUDA_SERVER` only if CUDA was not initialized
inside the block. If CUDA was initialized, the process-global SCUDA connection
is already active and cannot be disconnected safely.

The adapter does not create a new PyTorch backend such as
`torch.device("scuda")`. A true custom PyTorch device would require registering
PrivateUse1 kernels and backend support. SCUDA already works best when PyTorch
sees CUDA tensors and the SCUDA library is selected through the dynamic linker.
