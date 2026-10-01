"""
Обучение (дообучение) MobileNet на датасете карт.

Что делает:
1. Берёт предобученную MobileNet из torchvision.
2. Меняет последний слой под нашу задачу (52 класса = 52 карты).
3. Загружает картинки из dataset/<rank>_<suit>/.
4. Обучает несколько эпох, показывая картинки модели.
5. Сохраняет веса в файл model.pth.

Запуск:  python scripts/train_model.py
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ── Настройки ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset_full"
BATCH_SIZE = 32        # сколько картинок показываем за раз
EPOCHS = 10            # сколько раз пройти по всем картинкам
LEARNING_RATE = 0.001  # насколько сильно подкручиваем веса за шаг
NUM_CLASSES = 52       # 52 карты в колоде

# Используем GPU, если он есть, иначе CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Обучаемся на: {DEVICE}")


# ── 1. Подготовка данных ──────────────────────────────────────────────
# Картинки надо привести к одному размеру и нормализовать, как учили MobileNet.
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # MobileNet ждёт картинки 224x224
    transforms.ToTensor(),              # превращаем картинку в числа (тензор)
    transforms.Normalize(               # нормализация, как у готовой MobileNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Загружаем картинки: имя папки = метка класса (например "8_diamonds").
# В dataset_full лежат только 52 папки-класса (служебные удалены).
dataset = datasets.ImageFolder(root=str(DATASET_DIR), transform=transform)
print(f"Всего картинок: {len(dataset)}")
print(f"Классов: {len(dataset.classes)}")

# ── Разделяем данные: обучающие (train) и проверочные (val) ────────────
# 80% картинок модель увидит при обучении, 20% — НЕ увидит никогда.
# Их мы используем только в конце для честной проверки.
val_size = int(len(dataset) * 0.2)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Обучающих картинок: {train_size}, проверочных: {val_size}")

# DataLoader — подаёт картинки порциями (батчами) и перемешивает.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ── 2. Берём готовую MobileNet и переделываем последний слой ──────────
model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")

# Меняем последний (классификаторный) слой: было 1000 классов, стало 52.
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

model = model.to(DEVICE)


# ── 3. Готовим "учителя" (loss) и "оптимизатор" (как подкручивать веса) ──
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ── 4. Цикл обучения ──────────────────────────────────────────────────
model.train()
for epoch in range(1, EPOCHS + 1):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()          # обнуляем градиенты
        outputs = model(images)        # модель делает предсказание
        loss = criterion(outputs, labels)  # считаем ошибку
        loss.backward()                # вычисляем, как подправить веса
        optimizer.step()               # подкручиваем веса

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total * 100
    print(f"Эпоха {epoch}/{EPOCHS}: ошибка (loss) = {running_loss/len(train_loader):.4f}, "
          f"точность (train) = {acc:.1f}%")


# ── 5. ЧЕСТНАЯ проверка на невиданных картинках (validation) ──────────
model.eval()  # переключаем модель в режим "только предсказывать", без обучения
correct = 0
total = 0
with torch.no_grad():  # отключаем подсчёт градиентов (не учимся, просто проверяем)
    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_acc = correct / total * 100
print(f"\nТочность на проверочных (невиданных) картинках: {val_acc:.1f}%")


# ── 6. Сохраняем обученную модель ─────────────────────────────────────
save_path = PROJECT_ROOT / "model.pth"
torch.save(model.state_dict(), save_path)
print(f"\nМодель сохранена: {save_path}")
