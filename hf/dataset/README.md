---
license: mit
task_categories:
  - image-classification
tags:
  - cards
  - solitaire
  - computer-vision
  - computer-use
  - image-classification
size_categories:
  - 1K<n<10K
---

# Solitaire Cards Dataset

Images of playing cards captured from **live Microsoft Solitaire Collection**
games (1920×1040 window, UWP app) for training the vision layer of an autonomous
computer-use agent.

## Configurations

| Folder | Images | Size | What it is |
| --- | --- | --- | --- |
| `full` | 3254 | 131×176 px | complete face-up cards — the training set of the published recognizer |
| `corners` | 3967 | 42×54 px | corner crops from the earlier, abandoned pipeline (kept for comparison) |
| `fixtures` | 3 | 1920×1040 px | real screenshots used as regression tests, with a hand-read ground truth |

All 52 classes are stored as folders named `<rank>_<suit>` (`A_hearts`,
`10_spades`, `2_clubs`, …, `K_diamonds`), which is exactly the class order
`torchvision.datasets.ImageFolder` produces (alphabetical) and the order of the
published model's outputs.

## How it was collected

1. `scripts/collect_full.py` runs while a game is open and saves every fully
   visible card crop to `dataset_full/_unlabeled/`.
2. `scripts/sort_full.py` asks a local vision-language model (`qwen2.5vl:7b` via
   Ollama) for the rank and suit, rejects images that are actually a stack of two
   cards, and moves the rest into `<rank>_<suit>/` folders.
3. Ambiguous cases were sorted manually.

**Label noise is expected.** The labels come from a model plus manual sorting, so
the 99.91 % accuracy of the published recognizer is measured against these labels,
not against a human-audited gold standard.

## Ground truth for the fixtures

`fixtures/` contains three real screenshots with a manually read table state
(written down card by card, including the rank and suit of every visible card):

* `live_frame_2026-10-01.png` — midgame with deep fans (columns 2 and 6 empty)
* `live_frame_empty_cols_2026-10-01.png` — the same game later, when columns 2 and 6 are empty
* `live_frame_fresh_deal_2026-10-01.png` — a fresh deal, used to catch the
  "blue portrait mistaken for a card back" bug (in it the bottom card of column 5
  is Q♣, whose dress is drawn in blue)

The exact expected layout per frame is in
[`tests/test_screen_to_board_real_frame.py`](https://github.com/YanChi-pixel/solitaire-computer-agent/blob/main/tests/test_screen_to_board_real_frame.py).

## Intended use

Training and benchmarking card classifiers and, more generally, perception layers
for computer-use agents. Not suitable for anything that requires human-audited
labels.

## Links

* Agent code: [solitaire-computer-agent](https://github.com/YanChi-pixel/solitaire-computer-agent)
* Recognizer: [solitaire-card-recognizer](https://huggingface.co/YanChi-pixel/solitaire-card-recognizer)
* Case study: [ai-wiki.tech](https://ai-wiki.tech/en/cases/autonomous-computer-agent/)

## License and provenance

MIT for the dataset packaging. The images show the Microsoft Solitaire Collection
card art and are published for research and educational purposes; this dataset is
not affiliated with or endorsed by Microsoft.
