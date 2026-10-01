"""Экспортирует обученный `model.pth` в ONNX для демо в браузере.

Статический Space (бесплатный) считает модель прямо в браузере через
ONNX Runtime Web, поэтому нужен ONNX-файл. В граф добавлен softmax — тогда
на стороне JS не нужно ничего считать, кроме самого тензора.

    python hf/export_onnx.py [куда.onnx]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "hf" / "space_static" / "model.onnx"
NUM_CLASSES = 52


class WithSoftmax(torch.nn.Module):
    """MobileNet v2 + softmax: на выходе вероятности, а не логиты."""

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.backbone(x), dim=1)


def build_model() -> torch.nn.Module:
    model = torchvision.models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    state = torch.load(ROOT / "model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)

    model = WithSoftmax(build_model())
    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["probabilities"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("ONNX сохранён: %s (%.1f МБ)" % (out, out.stat().st_size / 1024 / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
