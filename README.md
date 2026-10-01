# solitaire-computer-agent

[![CI](https://github.com/YanChi-pixel/solitaire-computer-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/YanChi-pixel/solitaire-computer-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-solitaire--card--recognizer-yellow)](https://huggingface.co/WildFuria/solitaire-card-recognizer)
[![Demo on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Space-green)](https://huggingface.co/spaces/WildFuria/solitaire-card-recognizer-demo)

**An autonomous agent that plays Microsoft Solitaire Collection (Klondike)
through the real screen and mouse.** No game API, no process-memory reading, no
injected DLLs and no access to the application's internal state: for the agent
the game exists only as pixels in and mouse events out — which is exactly the
situation of any computer-use system (RPA, an agent on someone else's
application, UI automation without integration).

Card recognition **99.91 %** · **8.1 ms** per card with preprocessing · **40
consecutive moves** without a memory↔screen desync · all on a game that fights
back (it is a UWP app that ignores synthetic input).

![Midgame frame](docs/img/frame-midgame.jpg)

## Why Klondike is a fair testbed, not a toy

The game has no API, and Microsoft Solitaire Collection is a **UWP application**,
not a plain `.exe`, so the usual automation tricks simply do not work:

| UWP problem | What happens |
| --- | --- |
| The window is invisible to the ordinary Win32 API | Window enumeration returns empty, `MainWindowHandle = 0` |
| `BitBlt` (GDI) does not capture the content | The screenshot comes out black: average brightness 36, green felt 0 % |
| Synthetic input is ignored | `mouse_event`, `SetCursorPos` and instrumented drag are all dropped |
| The window lives in `ApplicationFrameHost` | Title, geometry and client area are computed differently |

On top of that, the domain is genuinely visual: cards **overlap in a fan**, so
only a strip of each card is visible, and face-down cards reveal neither rank nor
suit — the table state has to be reconstructed and kept in memory between
actions.

## Architecture: Vision → Rules → Planning → Action → Verification

```
        ┌──────────────────────┐
        │  Game screen (UWP)   │◄──────────── physical action ────────┐
        │      1920×1040       │                                      │
        └──────────┬───────────┘                                      │
   screenshot (mss, DWM)                                              │
                   ▼                                                  │
   ┌──────────────────────────────┐                                   │
   │ Vision: geometry + MobileNet │  frame → Board                    │
   └──────────────┬───────────────┘                                   │
                  ▼                                                   │
   ┌──────────────────────────────┐                                   │
   │ Rule Engine: get_valid_moves │  numbered list of legal moves     │
   └──────────────┬───────────────┘                                   │
                  ▼                                                   │
   ┌──────────────────────────────┐                                   │
   │ LLM Planner (qwen2.5:14b)    │  move_index + reasoning           │
   └──────────────┬───────────────┘                                   │
                  ▼                                                   │
   ┌──────────────────────────────┐                                   │
   │ Act: SendInput click / drag  │ ──────────────────────────────────┘
   └──────────────┬───────────────┘
                  │ new frame
                  ▼
   ┌──────────────────────────────┐
   │ Verification: memory vs screen│  match → continue
   │  verify + quantity + loop     │  mismatch → Undo + retry
   └──────────────────────────────┘
```

| Component | Task | Code |
| --- | --- | --- |
| Vision | find the table, columns and cards; read what is actually visible | `vision/` |
| Rule Engine | pure logic: which actions are legal and what follows each | `orchestrator/rule_engine.py` |
| LLM Planner | choose **one** action from the list and explain the choice | `planner/` |
| Act | turn the action into pixels and execute it physically | `act/` |
| Verification | confirm the screen became what memory predicted | `orchestrator/verify.py`, `quantity_guard.py`, `loop_guard.py` |
| Infrastructure | local server + web UI ("move / autoplay / stop"), move log | `server.py`, `web/index.html` |

**The main architectural principle:** the rules compute *which* actions are
legal, the LLM decides *which* of them to take. An illegal move is therefore
impossible by construction — the model is never asked to invent a legal action,
only to choose among the ones that already exist.

## Results in numbers

| Metric | Value |
| --- | --- |
| Card classification accuracy | **99.91 %** (3251/3254) on the full dataset, **100 %** (650/650) on the holdout split |
| Model size | 2,290,484 parameters / 9.4 MB (MobileNet v2, 52 classes) |
| Recognition latency | **6.7 ms** pure inference, **8.1 ms** with crop and preprocessing |
| Full table read (frame → Board) | ≈150 ms (geometry + 12 model inferences); frame capture ≈37 ms |
| Moves without desync | **40 in a row** in one live run, 93 moves in total |
| Score under AI control | 260 (three foundations started) |
| Dataset | 3254 full cards (131×176) + 3967 corner crops, 52 classes |
| Tests | 8 vision + 9 verification + rule-engine set — all green |
| Full move time | 5–20 s, almost all of it the local LLM |

## The bug that looked like a bad neural network

The most instructive part of the project. The agent would suddenly report:

```
⛔ Stuck
chosen = Move 1 card(s) from column 3 to column 4
attempt 0: ok=False reason = col 4 top: mem=10♣ screen=K♠
```

Memory believed the top of column 4 was the ten of clubs, "the screen" showed a
king — and **memory was right**: the actually playable card in that column is
10♣. The vision layer was wrong. Five independent root causes came out of it:

| Cause | What it really was | Evidence |
| --- | --- | --- |
| `open_y` pointed at the **first** face-up card of the fan instead of the bottom, playable one | the model was fed a stitched image of two cards (top strip of K♦ + face of J♦) — a **crop** bug, not a model bug | confidence 0.25 and a wrong suit → **1.000** on all seven columns after the fix |
| An **empty column** broke the whole geometry | the grid step was `median` of gaps; an empty column yields no cluster, so gaps become multiples: `[1,2,2,1]` → median 1.5 steps | detected centers `203…1715` → `455…1463`, "card size" 197×265 → 131×176 |
| The table top was **guessed** (`int(H*0.33)` = 343 instead of the real 350) | a king dropped into an empty column landed 7 px above the slot and was rejected by the count check | move rejected → measured top, move accepted |
| Waste and stock after a **restart mid-game** | the top waste card was re-read only if a placeholder existed, and the "point fix" was a stub that returned "fixed" without writing anything | `waste: mem=J♣ screen=5♦`; 8/10 moves failed → 20/20 passed |
| The **blue portrait** of Q♣ was mistaken for a card back | the "is this a back?" feature was measured on the central strip; a face card's portrait is blue too | blue fraction: back 1.000, Q♣ 0.214, others 0.000 |

![The crop that looked like a model error](docs/img/fan-crop.png)

The invariant *"`open_y` is the top of the bottom, playable card"* was written in
code comments — in the clicks module and in the verification module. The
implementation did not fulfil it, and comments are not checked by a compiler.
That is why regression tests on real frames became the project's final
deliverable: three real screenshots with a hand-read ground truth, down to the
rank and suit of every visible card.

More detail (in Russian, with all five causes, the constants table and the test
log): [`docs/engineering-notes.md`](docs/engineering-notes.md).

## How it was tested

```bash
python tests/test_screen_to_board_real_frame.py   # 8/8  — vision on 3 REAL frames, hand-read ground truth
python tests/test_verify.py                       # 9/9  — move verification geometry
python tests/test_rule_engine.py                  # rule engine: legal move generation
python tests/test_calibration.py                  # column grid + table top on a synthetic table
python scripts/_autoplay_probe.py 10              # N live moves + independent memory-vs-screen diff
```

The live probe makes N moves through the server API and, after each move,
independently captures a frame, reads the table with its own vision and compares
it with the server's memory (column tops, waste top, foundations). Results:
midgame 20 moves — 0 discrepancies, fresh deal 40 moves — 0, continuation 21
moves — 0, control run 12 moves — 0.

## Install

Windows 10/11 (the project uses WinAPI, `mss` and `pywin32`), Python 3.10+:

```bash
git clone https://github.com/YanChi-pixel/solitaire-computer-agent.git
cd solitaire-computer-agent
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch with CUDA is optional but recommended for training; inference runs fine
on CPU (6.7 ms per card is a GPU figure; expect tens of milliseconds on CPU).

The LLM strategist is configurable:

* **Ollama** (default, local, free): `ollama pull qwen2.5:14b`
* **DeepSeek API** (or any OpenAI-compatible endpoint): set `LLM_BACKEND=deepseek`
  and `DEEPSEEK_API_KEY` in `.env`

Copy `.env.example` to `.env` and fill it in. `.env` is never committed.

## Run

Make sure a Klondike game is open and visible on screen, then:

```bash
python server.py        # web UI on http://127.0.0.1:8000
python play.py          # one move from the console
start.bat               # one-click: frees port 8000, starts the server, opens the browser
```

In the web UI: **Make a move**, **Auto play**, **Stop** — every move comes with
the LLM's reasoning in human language, which is the point of using an LLM here
at all.

## Training your own recognizer

The published `model.pth` was trained on 3254 auto-labeled cards collected from
live play. To reproduce or retrain:

```bash
python scripts/collect_full.py          # collect FULL face-up cards while you play
python scripts/sort_full.py             # sort them into dataset_full/<rank>_<suit>/
python scripts/train_model.py           # fine-tune MobileNet v2 → model.pth
python scripts/_model_metrics.py        # accuracy, confusion pairs, latency
```

An older pipeline (42×54 corner crops, folder `dataset/`) is still in the repo:
`vision/dataset_collector.py` → `scripts/label_corners.py` (or `auto_label.py`
with a local vision model) → `dataset/<rank>_<suit>/`. It was abandoned in favour
of full-card crops — the model reads a whole card far more reliably than a tiny
corner.

The class order the model outputs lives in `vision/classes.json` and matches
`torchvision.datasets.ImageFolder` (alphabetical folder order). Keep it in sync
if you retrain on a different class set.

## Vision invariants (do not break these)

1. **`open_y` is the top of the BOTTOM (playable, free) face-up card of a
   column** — never the first face-up card under the backs. `layout.open_y`,
   `apply_move_to_screen`, `verify.check_shift`, `verify_quantity` and every
   "re-read the card" path assume the free card.
2. Only **fully visible** cards are ever recognized: the free card of a column,
   the top of the waste fan, the stock/backs count, the foundation tops.
   Overlapped cards are tracked in memory by `apply_move`.
3. Row order inside a column is structural, not threshold-based: face-down backs
   are a solid group on top, and the first face-up card starts at the first long
   white run.
4. **Column grid step = the MINIMUM gap between detected columns** (an empty
   column yields no cluster, so other gaps are multiples of the step).
5. "Is this card a back?" is decided over the **whole card area** with an inset:
   a back is blue edge to edge (1.000), a face card's portrait is ~0.2 blue.
6. The waste fan is a fan: only its rightmost card is read, and the screen is
   authoritative for it. The stock's presence comes from the screen too.
7. `table_top_y` is **measured** over non-empty columns, never guessed as a
   fraction of the frame height.

## Project layout

```
act/            mouse control (SendInput clicks/drags) + screen capture + layout
vision/         calibration, card extraction, MobileNet recognizer, frame → Board
orchestrator/   rule engine, verification, quantity guard, loop guard
planner/        prompt contract, JSON parsing, LLM backends (mock/ollama/openai)
scripts/        train, label, collect, benchmark, diagnostics, demo runs
tests/          regression tests; tests/data/*.png are REAL frames — keep them
web/index.html  web UI: buttons, board state, LLM reasoning
server.py       local HTTP server wired to the real game
play.py         one move from the console
dataset/        3967 corner crops, 52 classes (the earlier pipeline)
model.pth       trained MobileNet v2 weights (9.4 MB, 52 classes)
docs/           engineering notes (RU), README images
hf/             recipes for publishing the model, dataset and a demo Space
```

The full-card training set (3254 images, 131×176, 52 classes) is not stored in
this repository to keep it light — `hf/` contains the scripts that publish it,
and `scripts/collect_full.py` collects an equivalent set from your own screen.

## Hugging Face

The vision layer is published as a standalone artefact — download the weights in
one line, grab the dataset, or try the demo in a browser:

| | |
| --- | --- |
| 🃏 **Recognizer** | [huggingface.co/WildFuria/solitaire-card-recognizer](https://huggingface.co/WildFuria/solitaire-card-recognizer) — `model.pth`, `classes.json`, metrics and limitations |
| 📦 **Dataset** | [huggingface.co/datasets/WildFuria/solitaire-cards-dataset](https://huggingface.co/datasets/WildFuria/solitaire-cards-dataset) — 3254 full cards + 3967 corners + the regression fixtures |
| 🚀 **Live demo** | [huggingface.co/spaces/WildFuria/solitaire-card-recognizer-demo](https://huggingface.co/spaces/WildFuria/solitaire-card-recognizer-demo) — runs **entirely in the browser** via ONNX Runtime Web |

```python
from huggingface_hub import hf_hub_download

weights = hf_hub_download("WildFuria/solitaire-card-recognizer", "model.pth")
```

The demo is a *static* Space: `model.pth` is exported to ONNX with
`hf/export_onnx.py` and the browser does the inference, so there is no server and
no cold start. Gradio/Docker Spaces on the free `cpu-basic` tier require a PRO
subscription; static ones are free. The recipes for republishing everything live
in [`hf/`](hf/README.md).

## Limitations and what was deliberately left out

* **Covered cards are not read by vision** — that is the physics of the task, not
  a bug. The fan's composition is reconstructed from memory, so starting the
  server mid-game gives "partial" memory that self-recovers as moves are made
  but never becomes perfect.
* **Advertising in the game.** A new deal is started from the game's own menu,
  and an ad may play first; clicking through someone else's ad screen was not
  automated on purpose.
* **Speed is bottlenecked by the LLM**: vision takes fractions of a second while
  a move takes 5–20 s with the local 14B model. A smaller model or a GPU with
  more memory fixes that.
* **Mid-level strategy.** Prompt rules and anti-shuffle filters produce careful
  but not optimal play; a full Klondike solver with backtracking would play
  better. The LLM here is not for strength — it is for explainability.
* **Windows-only**, because of WinAPI input and UWP screen capture.

## Other lessons worth keeping

* **Perception quality matters as much as model quality.** One crop error looked
  exactly like a neural-network error (confidence 0.25, wrong suit) because the
  model was fed a stitched image of two cards. Check what your model is actually
  receiving before you retrain it.
* **State and observation are different things.** The screen reports only what is
  visible now; memory stores the state established by previous actions.
* **An external interface requires verification.** After an action you cannot
  assume it happened — the verification layer turned out to be more important
  than recognition accuracy, because it turns failures into recoverable
  situations.
* **Measurement beats guesswork.** Pixel measurements, move logs, real frames and
  regression tests found the root causes; the phase of "fix the symptom" did not.

## License and disclaimer

MIT — see [LICENSE](LICENSE).

The screenshots in `tests/data/` and `docs/img/` show the Microsoft Solitaire
Collection interface and are used as test fixtures for research and educational
purposes. This project is not affiliated with, endorsed by or connected to
Microsoft. MobileNet v2 comes from `torchvision` (BSD-3-Clause). The agent
controls the mouse and keyboard of your machine — run it on a desktop you can
watch and stop, not on a machine doing something else.

---

По-русски: [README.ru.md](README.ru.md) · Инженерные заметки: [docs/engineering-notes.md](docs/engineering-notes.md)
