"""Gradio demo: recognize a playing card with the published MobileNet v2.

The Space downloads the weights from the model repository (or uses a local
`model.pth` if one was uploaded next to this file), then shows the top-5 classes
for an uploaded image.

Trained for the vision layer of an autonomous agent that plays Microsoft
Solitaire Collection through the real screen: https://github.com/YanChi-pixel/solitaire-computer-agent
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
import torchvision
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

MODEL_REPO = "YanChi-pixel/solitaire-card-recognizer"
LOCAL_WEIGHTS = Path(__file__).parent / "model.pth"
LOCAL_CLASSES = Path(__file__).parent / "classes.json"

NOTE = (
    "Feed it a **fully visible** card (a crop or a screenshot region). "
    "Partially overlapped cards inside a fan are read unreliably — that is a "
    "documented data limitation, not a bug."
)


def _load_classes() -> list[str]:
    if LOCAL_CLASSES.exists():
        import json

        return json.loads(LOCAL_CLASSES.read_text(encoding="utf-8"))["classes"]
    path = hf_hub_download(MODEL_REPO, "classes.json")
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))["classes"]


def _load_model(num_classes: int) -> torch.nn.Module:
    weights_path = LOCAL_WEIGHTS if LOCAL_WEIGHTS.exists() else Path(
        hf_hub_download(MODEL_REPO, "model.pth")
    )
    model = torchvision.models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


CLASSES = _load_classes()
MODEL = _load_model(len(CLASSES))

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SUIT_SYMBOL = {"hearts": "♥", "diamonds": "♦", "clubs": "♣", "spades": "♠"}


def _pretty(class_name: str) -> str:
    rank, suit = class_name.split("_")
    return "%s%s" % (rank, SUIT_SYMBOL.get(suit, suit))


def predict(image: Image.Image | None):
    if image is None:
        return {}, "Upload an image of a card."
    with torch.no_grad():
        logits = MODEL(TRANSFORM(image.convert("RGB")).unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0]
    top = torch.topk(probs, k=5)
    result = {
        _pretty(CLASSES[int(idx)]): float(prob)
        for prob, idx in zip(top.values, top.indices)
    }
    best = max(result, key=result.get)
    return result, "**%s** — confidence %.1f%%" % (best, result[best] * 100)


DEMO = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Card image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Top-5 predictions"),
        gr.Markdown(label="Result"),
    ],
    title="Solitaire card recognizer",
    description=(
        "MobileNet v2 fine-tuned on 52 card classes (99.91 % accuracy on the "
        "training set, 100 % on the holdout split). " + NOTE
    ),
    examples=[],
    allow_flagging="never",
)

if __name__ == "__main__":
    DEMO.launch()
