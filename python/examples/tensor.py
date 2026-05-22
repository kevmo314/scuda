# /// script
# dependencies = [
#   "numpy",
#   "scuda",
#   "torch",
# ]
#
# [tool.uv.sources]
# scuda = { path = "..", editable = true }
# ///

import torch
import scuda


with scuda.connect(host="inferable-node-008:14833") as session:
    device = session.device()
    props = torch.cuda.get_device_properties(device)
    x = torch.arange(8, device=device, dtype=torch.float32)
    y = (x * 2).cpu()
    print("cuda available:", torch.cuda.is_available())
    print("device:", device)
    print("count:", torch.cuda.device_count())
    print("gpu:", props.name)
    print("result:", y.tolist())
