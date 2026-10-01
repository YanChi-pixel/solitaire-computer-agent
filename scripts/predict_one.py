"""
Проверка обученной модели на одной картинке.

Берёт картинку из dataset/<rank>_<suit>/, прогоняет через обученную модель
(model.pth) и печатает, что модель предсказала, и правильный ответ.
Сравниваем — видим, угадала ли модель.
"""

from pathlib import Path

import torch
import torchvision
from torchvision import transforms

# ── Настройки ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model.pth"
NUM_CLASSES = 52

# Какую картинку проверяем: берём первую попавшуюся из папки 8_diamonds.
# Поменяйте имя папки, чтобы проверить другую карту (например "K_spades").
TEST_DIR = PROJECT_ROOT / "dataset" / "8_diamonds"
TEST_IMAGE = next(TEST_DIR.glob("*.png"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── 1. То же самое преобразование, что и при обучении ─────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── 2. Загружаем обученную модель ─────────────────────────────────────
# Нам нужны те же 52 класса в том же порядке. Извлекаем их из папок.
class_names = sorted(
    d.name for d in (PROJECT_ROOT / "dataset").iterdir()
    if d.is_dir() and not d.name.startswith("_")
)

model = torchvision.models.mobilenet_v2(weights=None)  # берём архитектуру БЕЗ предобученных весов
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # грузим НАШИ веса
model = model.to(DEVICE)
model.eval()


# ── 3. Прогоняем картинку через модель ────────────────────────────────
from PIL import Image

image = Image.open(TEST_IMAGE).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # добавляем измерение "батч"

with torch.no_grad():
    output = model(input_tensor)
    predicted_idx = output.argmax(dim=1).item()

predicted_class = class_names[predicted_idx]
correct_class = TEST_IMAGE.parent.name  # имя папки = правильный ответ

print(f"Картинка: {TEST_IMAGE.name} (правильный ответ: {correct_class})")
print(f"Модель предсказала: {predicted_class}")
print(f"{'✅ ВЕРНО' if predicted_class == correct_class else '❌ ОШИБКА'}")
