---
license: mit
---

# Hi, I'm Yana 👋

I build **AI automation end to end** — from a client problem to a running pipeline
and an open-source repository. Mostly where language models meet real systems:
CMS platforms that have no API, desktop and browser automation, content pipelines,
and vision models that have to work on someone else's UI.

![Solitaire card recognizer demo](assets/space-demo.png)

## Featured artefacts

| | |
| --- | --- |
| 🃏 **Model** | [solitaire-card-recognizer](https://huggingface.co/WildFuria/solitaire-card-recognizer) — MobileNet v2 fine-tuned to 52 card classes: 99.91% accuracy (3251/3254), 100% on the holdout split, 8.1 ms per card |
| 📦 **Dataset** | [solitaire-cards-dataset](https://huggingface.co/datasets/WildFuria/solitaire-cards-dataset) — 3254 full cards + 3967 corner crops + regression fixtures, collected and auto-labelled from live play |
| 🚀 **Demo** | [solitaire-card-recognizer-demo](https://huggingface.co/spaces/WildFuria/solitaire-card-recognizer-demo) — runs entirely in the browser via ONNX Runtime Web: no server, no cold start |
| 🤖 **Agent** | [solitaire-computer-agent](https://github.com/YanChi-pixel/solitaire-computer-agent) — the agent behind them: vision → deterministic rule engine → LLM planner → WinAPI input → verification after every action |

## How I work

- **Deterministic logic before models.** In the agent, a rule engine computes every
  legal move and the LLM only chooses among them — an illegal action becomes
  impossible by construction. The LLM is there for explainability, not for brute
  force.
- **Measurement before guesswork.** Pixel measurements, move logs and regression
  tests on real frames found five root causes behind one "desync" bug; endless
  prompt tweaking had found none.
- **Ship it as an artefact.** A model is not finished until it has a card with
  metrics and limitations, a dataset with provenance, and a demo anyone can open.

## Toolbox

Python · PyTorch / torchvision · ONNX Runtime · LLMs (DeepSeek, Qwen, Ollama) ·
computer vision · UI automation without APIs (WinAPI, mss, Playwright) · PHP ·
CMS integrations (Diafan, Joomla, Webasyst) · PowerShell · GitHub Actions

## Elsewhere

- GitHub: [github.com/YanChi-pixel](https://github.com/YanChi-pixel)
- Case studies (EN / RU): [ai-wiki.tech](https://ai-wiki.tech/en/cases/)
