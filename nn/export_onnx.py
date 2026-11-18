"""Export RenjuDualNet (with Player Embeddings + FiLM) checkpoint to ONNX."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nn.features import NUM_INPUT_PLANES
from nn.model import RenjuDualNet


def main() -> None:
    ckpt_path = ROOT / "artifacts" / "renju_dualnet.pt"
    onnx_path = ROOT / "artifacts" / "renju_dualnet.onnx"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = RenjuDualNet(
        channels=ckpt.get("channels", 96),
        num_blocks=ckpt.get("num_blocks", 8),
        player_vocab_size=ckpt.get("player_vocab_size", 1),
        player_emb_dim=ckpt.get("player_emb_dim", 16),
    )
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    dummy_x = torch.randn(1, NUM_INPUT_PLANES, 15, 15, dtype=torch.float32)
    dummy_m = torch.zeros(1, dtype=torch.long)
    dummy_o = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        net,
        (dummy_x, dummy_m, dummy_o),
        str(onnx_path),
        input_names=["input", "mover_id", "opp_id"],
        output_names=["policy_logits", "value_logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "mover_id": {0: "batch"},
            "opp_id": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value_logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Exported {onnx_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
