---
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - computer-vision
  - image-classification
  - mobilenet-v2
  - cards
  - solitaire
  - computer-use
datasets:
  - solitaire-cards-dataset
metrics:
  - accuracy
---

# Solitaire Card Recognizer

A 52-class playing-card classifier fine-tuned from **MobileNet v2** (ImageNet
weights from `torchvision`). It was trained for an autonomous computer-use agent
that plays Microsoft Solitaire Collection through the real screen and mouse:
the agent sees only pixels, so this model is its "eyes".

* **Accuracy:** 99.91 % (3251/3254) on the full dataset, **100 %** (650/650) on
  the held-out split
* **Size:** 2,290,484 parameters (9.4 MB, `model.pth`)
* **Latency:** 6.7 ms pure inference, 8.1 ms including crop and preprocessing
  (single crop, RTX 2060S); tens of milliseconds on CPU
* **Input:** a full, fully visible card image (the training crops are 131×176 px;
  the model resizes to 224×224 and applies ImageNet normalization)
* **Output:** one of 52 classes in the order of `classes.json`
  (`10_clubs … Q_spades`, matching `torchvision.datasets.ImageFolder` ordering)

## Usage

```python
import json
from huggingface_hub import hf_hub_download
import torch, torchvision
from PIL import Image
from torchvision import transforms

repo = "YanChi-pixel/solitaire-card-recognizer"     # ← your namespace after upload
weights = hf_hub_download(repo, "model.pth")
classes = json.load(open(hf_hub_download(repo, "classes.json"), encoding="utf-8"))["classes"]

model = torchvision.models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("card.png").convert("RGB")
with torch.no_grad():
    probs = torch.softmax(model(tf(img).unsqueeze(0)), dim=1)[0]
rank_suit, idx = classes[int(probs.argmax())], int(probs.argmax())
print(rank_suit, f"{float(probs[idx]):.3f}")
```

## Training data

Collected and auto-labelled from live play: **3254 full cards (131×176 px)** in 52
balanced classes, plus 3967 earlier corner crops (42×54 px) from the abandoned
first pipeline. Published as
[`solitaire-cards-dataset`](https://huggingface.co/datasets/solitaire-cards-dataset).
Labels were produced semi-automatically (a local vision-language model plus manual
sorting), so expect a small amount of label noise — the reported accuracy is
measured against those labels, not against a human-audited test set.

## Limitations

* Trained on the **Microsoft Solitaire Collection** card art at 1920×1040. Other
  card designs, fonts or resolutions are out of distribution.
* It expects a **fully visible card**. A partially overlapped card inside a fan
  (only its top strip visible) is read unreliably — that is a data limitation, not
  a code bug, and it is documented in the agent's engineering notes.
* It classifies a single card crop; it does not detect cards on a full screenshot
  (that geometry lives in the agent's `vision/` layer).

## Links

* Code, tests and the full agent: [solitaire-computer-agent](https://github.com/YanChi-pixel/solitaire-computer-agent)
* Interactive demo: [Space](https://huggingface.co/spaces/YanChi-pixel/solitaire-card-recognizer-demo)
* Case study: [ai-wiki.tech](https://ai-wiki.tech/en/cases/autonomous-computer-agent/)

## License

MIT. MobileNet v2 architecture comes from `torchvision` (BSD-3-Clause).
