"""
Recognize a face-up card using the trained MobileNet.

Loads model.pth once and provides recognize_corner(image), which crops the
card corner and returns (rank:int, suit:Suit).

Class folder names like "8_diamonds" are mapped to rank and Suit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from orchestrator.rule_engine import Suit

# ── настройки ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODEL_PATH = _PROJECT_ROOT / "model.pth"
_CLASSES_PATH = Path(__file__).resolve().parent / "classes.json"
_NUM_CLASSES = 52

_SUIT_BY_NAME = {
    "hearts": Suit.HEARTS,
    "diamonds": Suit.DIAMONDS,
    "clubs": Suit.CLUBS,
    "spades": Suit.SPADES,
}

_RANK_BY_STR = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13,
}

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _class_to_rank_suit(class_name: str) -> tuple[int, Suit]:
    """'8_diamonds' -> (8, Suit.DIAMONDS)."""
    rank_str, suit_str = class_name.split("_")
    return _RANK_BY_STR[rank_str], _SUIT_BY_NAME[suit_str]


def _load_class_names() -> list[str]:
    """Порядок классов обученной модели.

    Читается из `vision/classes.json` — он совпадает с порядком выходов
    сети (ImageFolder сортирует папки датасета по алфавиту). Если файла
    нет, список собирается из папок `dataset_full/` (полный датасет).
    """
    if _CLASSES_PATH.exists():
        data = json.loads(_CLASSES_PATH.read_text(encoding="utf-8"))
        return list(data["classes"])
    return sorted(
        d.name for d in (_PROJECT_ROOT / "dataset_full").iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )


class Recognizer:
    """Загружает модель и умеет распознавать уголок карты."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.class_names = _load_class_names()

        self.model = torchvision.models.mobilenet_v2(weights=None)
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features, _NUM_CLASSES
        )
        self.model.load_state_dict(
            torch.load(_MODEL_PATH, map_location=self.device, weights_only=True)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def recognize(self, corner_image: np.ndarray) -> tuple[int, Suit]:
        """Принимает RGB-массив уголка карты, возвращает (rank, suit)."""
        rank, suit, _conf = self.recognize_with_confidence(corner_image)
        return rank, suit

    def recognize_with_confidence(self, corner_image: np.ndarray) -> tuple[int, Suit, float]:
        """Return (rank, suit, confidence) — confidence is the softmax prob."""
        img = Image.fromarray(corner_image).convert("RGB")
        tensor = _transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            conf = float(probs.max().item())
            idx = int(output.argmax(dim=1).item())
        rank, suit = _class_to_rank_suit(self.class_names[idx])
        return rank, suit, conf

    def recognize_at(
        self,
        image: np.ndarray,
        cx: int,
        top_y: int,
        card_w: int = 131,
        card_h: int = 176,
    ) -> tuple[int, Suit]:
        """
        Recognize a card by its position on the full screenshot.

        Crops the FULL card (card_w x card_h, centered on cx). The model's
        transform resizes any crop to 224x224 internally, so the crop only
        needs to capture the whole card — its pixel size may vary with screen
        resolution. `card_w`/`card_h` are measured from the layout (see
        vision.screen_to_board.measure_card_size); defaults fit 1920px.
        """
        left = cx - card_w // 2
        top = max(0, top_y)
        crop = image[top:top + card_h, left:left + card_w]
        return self.recognize(crop)

    def recognize_at_with_confidence(
        self,
        image: np.ndarray,
        cx: int,
        top_y: int,
        card_w: int = 131,
        card_h: int = 176,
    ) -> tuple[int, Suit, float]:
        """Same as recognize_at, but also returns the softmax confidence."""
        left = cx - card_w // 2
        top = max(0, top_y)
        crop = image[top:top + card_h, left:left + card_w]
        return self.recognize_with_confidence(crop)


# ── глобальный синглтон (грузим модель один раз) ──────────────────────
_recognizer = None


def get_recognizer() -> Recognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = Recognizer()
    return _recognizer
