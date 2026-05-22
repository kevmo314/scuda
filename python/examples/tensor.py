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


def prompt_endpoint() -> str:
    host = input("SCUDA server host: ").strip()
    while not host:
        host = input("SCUDA server host: ").strip()

    port_text = input("SCUDA server port [14833]: ").strip() or "14833"
    port = int(port_text)
    return f"{host}:{port}"


with scuda.connect(host=prompt_endpoint()) as session:
    device = session.device()
    props = torch.cuda.get_device_properties(device)
    x = torch.arange(8, device=device, dtype=torch.float32)
    y = (x * 2).cpu()
    print("cuda available:", torch.cuda.is_available())
    print("device:", device)
    print("count:", torch.cuda.device_count())
    print("gpu:", props.name)
    print("result:", y.tolist())
