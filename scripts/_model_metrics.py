"""Метрики обученной MobileNet: точность на датасете, 80/20-сплит, латентность.

Запуск: python scripts/_model_metrics.py
"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

MODEL_PATH = ROOT / "model.pth"
DATA_DIR = ROOT / "dataset_full"
NUM_CLASSES = 52

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = "cuda" if torch.cuda.is_available() else "cpu"
ds = datasets.ImageFolder(root=str(DATA_DIR), transform=transform)
print(f"device={device}  всего картинок={len(ds)}  классов={len(ds.classes)}")

model = torchvision.models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device).eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"параметров модели: {n_params:,}  размер model.pth: {MODEL_PATH.stat().st_size/1e6:.1f} MB")


def evaluate(subset, name):
    loader = DataLoader(subset, batch_size=64, shuffle=False)
    correct = total = 0
    conf_sum = 0.0
    wrong_pairs = {}
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            probs = torch.softmax(out, dim=1)
            conf, pred = probs.max(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            conf_sum += conf.sum().item()
            for p, label in zip(pred.tolist(), labels.tolist()):
                if p != label:
                    wrong_pairs[(ds.classes[label], ds.classes[p])] = (
                        wrong_pairs.get((ds.classes[label], ds.classes[p]), 0) + 1)
    acc = correct / total * 100
    print(f"{name}: accuracy={acc:.2f}%  ({correct}/{total})  средняя уверенность={conf_sum/total:.3f}")
    if wrong_pairs:
        top = sorted(wrong_pairs.items(), key=lambda kv: -kv[1])[:6]
        print("   частые путаницы: " + ", ".join(f"{a}->{b} ({n})" for (a, b), n in top))
    return acc


acc_all = evaluate(ds, "весь dataset_full")

torch.manual_seed(42)
val_size = int(len(ds) * 0.2)
train_size = len(ds) - val_size
_, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
acc_val = evaluate(val_ds, "отложенные 20% (seed=42)")

# латентность одного распознавания (как в игре: один кроп карты)
import numpy as np
from PIL import Image

crop = np.array(Image.open(next((DATA_DIR / "8_diamonds").glob("*.png"))).convert("RGB"))
tensor = transform(Image.fromarray(crop)).unsqueeze(0).to(device)
with torch.no_grad():
    for _ in range(5):
        model(tensor)
    t0 = time.perf_counter()
    for _ in range(50):
        model(tensor)
    dt = (time.perf_counter() - t0) / 50 * 1000
print(f"\nлатентность одного распознавания: {dt:.1f} мс ({device})")

# латентность полного прохода по столу (как screen_to_board: 7 колонок + waste + 4 фундамента)
N = 12
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(10):
        model(tensor.repeat(N, 1, 1, 1))
    dt_full = (time.perf_counter() - t0) / 10 * 1000
print(f"латентность полного чтения стола (~12 карт, один батч): {dt_full:.0f} мс")
