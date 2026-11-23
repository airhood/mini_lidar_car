"""Export the trained lower policy's CPU weights for Unity."""
import argparse
from pathlib import Path
import struct

import torch

LAYERS = ("scan_encoder.0", "scan_encoder.2", "command_head.0", "command_head.2")
SHAPES = ((16, 4, 9), (32, 16, 5), (64, 774), (2, 64))
TEMPORAL = (
    ("temporal_encoder.weight_ih_l0", (48, 12)),
    ("temporal_encoder.weight_hh_l0", (48, 16)),
    ("temporal_encoder.bias_ih_l0", (48,)),
    ("temporal_encoder.bias_hh_l0", (48,)),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = torch.jit.load(str(args.checkpoint), map_location="cpu")
    state = model.state_dict()
    temporal = getattr(model, "temporal", False)
    stride = getattr(model, "history_stride", 1)
    goal = getattr(model, "use_relative_goal", False)
    if stride == 5 and goal:
        header = b"LIDART05" if temporal else b"LIDARH05"
    elif stride == 1:
        header = b"LIDART01" if temporal else (b"LIDARG01" if goal else b"LIDARP01")
    else:
        raise ValueError(f"Unsupported history stride {stride}")
    shapes = (*SHAPES[:2], (64, 790), SHAPES[3]) if temporal else SHAPES
    tensors = []
    for name, shape in zip(LAYERS, shapes):
        tensors.extend(((f"{name}.weight", shape), (f"{name}.bias", (shape[0],))))
    if temporal:
        tensors.extend(TEMPORAL)
    with args.output.open("xb") as binary:
        binary.write(header)
        for name, expected in tensors:
            tensor = state[name]
            if name == "command_head.0.weight" and tuple(tensor.shape) == (64, 772):
                # The Unity v1 format reserves two columns for previous motor commands.
                tensor = torch.nn.functional.pad(tensor, (0, 2))
            if tuple(tensor.shape) != expected or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid tensor {name}: {tensor.shape}")
            binary.write(struct.pack("<I", tensor.numel()))
            binary.write(tensor.detach().contiguous().numpy().astype("<f4").tobytes())


if __name__ == "__main__":
    main()
