---
title: Solitaire Card Recognizer
emoji: 🃏
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: mit
---

# Solitaire card recognizer — live demo

Recognise a playing card in the browser: the model runs **client-side** through
ONNX Runtime Web, so there is no server, no cold start and nothing to wake up.

Upload a **whole card** (as it appears on screen, roughly 131×176 px) or click one
of the example cards. The demo shows the top-5 classes with probabilities.

This is the vision layer of
[solitaire-computer-agent](https://github.com/YanChi-pixel/solitaire-computer-agent) —
an autonomous agent that plays Microsoft Solitaire Collection through the real
screen and mouse: computer vision, a deterministic rule engine, an LLM strategist
and WinAPI input, with a verification layer after every move.

| | |
| --- | --- |
| Architecture | MobileNet v2 (torchvision, ImageNet weights) fine-tuned to 52 classes |
| Accuracy | 99.91 % (3251/3254) on the full dataset, 100 % (650/650) on the holdout split |
| Model | 2,290,484 parameters · 9.4 MB (`model.pth`) · 8.7 MB exported to ONNX |
| Weights | [solitaire-card-recognizer](https://huggingface.co/WildFuria/solitaire-card-recognizer) |
| Dataset | [solitaire-cards-dataset](https://huggingface.co/datasets/WildFuria/solitaire-cards-dataset) |
| Export | `torch.onnx.export` with softmax inside the graph (`hf/export_onnx.py`) |

**Known limitation.** A tiny corner crop is out of distribution — the model was
trained on whole cards. A card partially overlapped by another card inside a fan
(only its top strip visible) is read unreliably, which is a data limitation
documented in the project's engineering notes, not a bug in this demo. The agent
works around it by reading only fully visible cards and keeping the rest of the
fan in memory.

MIT licensed. The demo is not affiliated with Microsoft.
