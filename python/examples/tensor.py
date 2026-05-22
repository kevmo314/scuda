# /// script
# dependencies = [
#   "numpy",
#   "lupine",
#   "torch",
# ]
#
# [tool.uv.sources]
# lupine = { path = "..", editable = true }
# ///

import torch
import lupine


def prompt_endpoint() -> str:
    host = input("LUPINE server host: ").strip()
    while not host:
        host = input("LUPINE server host: ").strip()

    port_text = input("LUPINE server port [14833]: ").strip() or "14833"
    port = int(port_text)
    return f"{host}:{port}"


with lupine.connect(host=prompt_endpoint()) as session:
    device = session.device()
    props = torch.cuda.get_device_properties(device)
    x = torch.arange(8, device=device, dtype=torch.float32)
    y = (x * 2).cpu()
    print("cuda available:", torch.cuda.is_available())
    print("device:", device)
    print("count:", torch.cuda.device_count())
    print("gpu:", props.name)
    print("result:", y.tolist())
