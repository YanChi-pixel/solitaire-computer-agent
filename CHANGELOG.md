# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-10-01

Initial public release: the autonomous Klondike agent extracted from the working
project and prepared for reuse.

### Added

- `vision/` — table geometry measured from pixels (column grid, card size, table
  top, row anchors), card extraction and the trained MobileNet v2 recognizer.
- `orchestrator/` — `rule_engine.py` (Board / Card / Move, `get_valid_moves`,
  `apply_move`), `verify.py` (memory vs screen), `quantity_guard.py`
  (fan-step estimation and card-count check), `loop_guard.py` (layout signature
  and "dead deck" detection).
- `planner/` — prompt contract, JSON parsing and three interchangeable LLM
  backends: mock, Ollama (local) and any OpenAI-compatible API (DeepSeek).
- `act/` — UWP-accepted input through WinAPI `SendInput`, window lookup, screen
  capture via `mss` (DWM) and the layout of every screen element.
- `server.py` + `web/index.html` — local server and UI ("Make a move",
  "Auto play", "Stop") with the LLM's reasoning rendered per move.
- `scripts/` — data collection, auto-labeling, training, metrics, benchmarking,
  live probes and diagnostics (including `diag_overlap_probe.py`, which shows how
  the recognizer behaves on a partially overlapped card).
- `tests/` — 8 vision checks on three **real** game frames with a hand-read
  ground truth, 9 verification checks and the rule-engine set.
- `model.pth` — the trained recognizer (MobileNet v2, 52 classes, 99.91 % on the
  full dataset, 100 % on the holdout split).
- `docs/engineering-notes.md` — the incident analysis in Russian: five root
  causes behind "Stuck", with the measurements that found them.
- `hf/` — model card, dataset card and a Gradio Space for publishing the
  artefacts to Hugging Face, plus a one-command upload script.
- CI: ruff, byte-compile, portable import smoke test and the full test suite
  (including the vision regression) on Python 3.10 and 3.12.

### Changed (compared with the private working copy)

- The class list moved from a `dataset_full/` directory listing into
  `vision/classes.json`, so the repository works without the full training set
  (the file still matches the `ImageFolder` order the model was trained with).
- Dead pixel-window code in `count_cards_in_column` removed; that function now
  does only what its invariant says.
- `.env` replaced by `.env.example`; the real API key is not part of the repo.
- Absolute paths, working chat logs, run artefacts and the diagnostics written for
  an external review removed or neutralized.
- Lint debt cleared: 133 ruff findings (import order, unused imports, dead
  locals, one-line statements) fixed so that CI is green from the first commit.

[1.0.0]: https://github.com/YanChi-pixel/solitaire-computer-agent/releases/tag/v1.0.0
