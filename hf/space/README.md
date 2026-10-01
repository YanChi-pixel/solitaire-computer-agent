---
title: Solitaire Card Recognizer
emoji: 🃏
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Solitaire card recognizer — demo

Interactive demo of the vision layer of
[solitaire-computer-agent](https://github.com/YanChi-pixel/solitaire-computer-agent):
an autonomous agent that plays Microsoft Solitaire Collection through the real
screen and mouse.

Upload a **fully visible** card (a crop or a screenshot region) and the model
returns the top-5 classes with probabilities.

* Architecture: MobileNet v2 (torchvision, ImageNet weights) fine-tuned to 52 classes
* Accuracy: 99.91 % (3251/3254) on the full dataset, 100 % (650/650) on the holdout split
* Weights: [solitaire-card-recognizer](https://huggingface.co/YanChi-pixel/solitaire-card-recognizer)
* Dataset: [solitaire-cards-dataset](https://huggingface.co/datasets/YanChi-pixel/solitaire-cards-dataset)

**Known limitation:** a card that is partially overlapped by another card (only
its top strip visible, as happens inside a fan) is read unreliably — that is a
data limitation documented in the project's engineering notes, not a bug in the
demo. The agent works around it by reading only fully visible cards and keeping
the rest of the fan in memory.

MIT licensed; the demo is not affiliated with Microsoft.
