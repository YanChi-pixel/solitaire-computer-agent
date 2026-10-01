"""Publishes the model, the dataset and the demo Space to the Hugging Face Hub.

    pip install huggingface_hub
    export HF_TOKEN=hf_...                     # Windows: set HF_TOKEN=hf_...
    python hf/upload.py --what model
    python hf/upload.py --what dataset --full-dataset "D:\\AI Projects\\sol-dev\\dataset_full"
    python hf/upload.py --what space
    python hf/upload.py --what all --full-dataset "D:\\AI Projects\\sol-dev\\dataset_full"

The token is read from the environment and never stored in the repository.
Re-running is safe: existing repositories are updated, not recreated.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_DIR = REPO_ROOT / "hf"

MODEL_NAME = "solitaire-card-recognizer"
DATASET_NAME = "solitaire-cards-dataset"
SPACE_NAME = "solitaire-card-recognizer-demo"

DEFAULT_FULL_DATASET = REPO_ROOT / "dataset_full"
FALLBACK_FULL_DATASET = REPO_ROOT.parent / "dataset_full"


def get_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "Не задан HF_TOKEN.\n"
            "Создайте write-токен: https://huggingface.co/settings/tokens\n"
            "  Windows:  set HF_TOKEN=hf_...\n"
            "  bash:     export HF_TOKEN=hf_...\n"
        )
    return token


def get_api(token: str):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("Нужен пакет huggingface_hub: pip install huggingface_hub")
    return HfApi(token=token)


def publish_model(api, token: str, namespace: str, repo_id_holder: dict) -> str:
    from huggingface_hub import create_repo, upload_file

    repo_id = "%s/%s" % (namespace, MODEL_NAME)
    create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    for local, remote in (
        (REPO_ROOT / "model.pth", "model.pth"),
        (REPO_ROOT / "vision" / "classes.json", "classes.json"),
        (HF_DIR / "model" / "README.md", "README.md"),
        (HF_DIR / "model" / "config.json", "config.json"),
    ):
        if not local.exists():
            raise SystemExit("нет файла: %s" % local)
        upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                    repo_id=repo_id, token=token)
        print("    + %s" % remote)
    repo_id_holder["model"] = repo_id
    return repo_id


def publish_dataset(api, token: str, namespace: str, full_dataset: Path) -> str:
    from huggingface_hub import create_repo, upload_file, upload_folder

    repo_id = "%s/%s" % (namespace, DATASET_NAME)
    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)

    folders = [
        (full_dataset, "full", "полные карты для обучения (3254)"),
        (REPO_ROOT / "dataset", "corners", "уголки прежнего конвейера (3967)"),
        (REPO_ROOT / "tests" / "data", "fixtures", "реальные кадры для регресс-тестов"),
    ]
    for local, remote, description in folders:
        if not local.exists():
            print("    ! пропущено (%s): нет папки %s" % (description, local))
            continue
        count = sum(1 for _ in local.rglob("*") if _.is_file())
        print("    ^ %s/ — %s (%d файлов)" % (remote, description, count))
        upload_folder(folder_path=str(local), path_in_repo=remote,
                      repo_id=repo_id, repo_type="dataset", token=token,
                      ignore_patterns=["_unlabeled/*", "_trash/*", "synthetic_table.png"])

    upload_file(path_or_fileobj=str(HF_DIR / "dataset" / "README.md"),
                path_in_repo="README.md", repo_id=repo_id,
                repo_type="dataset", token=token)
    print("    + README.md")
    return repo_id


def publish_space(api, token: str, namespace: str, model_repo: str) -> str:
    from huggingface_hub import create_repo, upload_folder

    repo_id = "%s/%s" % (namespace, SPACE_NAME)
    create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, token=token)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in ("app.py", "requirements.txt", "README.md"):
            shutil.copy2(HF_DIR / "space" / name, stage / name)

        # Space должен скачивать модель из опубликованного репозитория
        app = (stage / "app.py").read_text(encoding="utf-8")
        app = app.replace('MODEL_REPO = "YanChi-pixel/solitaire-card-recognizer"',
                          'MODEL_REPO = "%s"' % model_repo)
        (stage / "app.py").write_text(app, encoding="utf-8")

        upload_folder(folder_path=str(stage), repo_id=repo_id,
                      repo_type="space", token=token)

    # приложить веса прямо к Space: демо не зависит от сети при старте
    from huggingface_hub import upload_file

    upload_file(path_or_fileobj=str(REPO_ROOT / "model.pth"), path_in_repo="model.pth",
                repo_id=repo_id, repo_type="space", token=token)
    upload_file(path_or_fileobj=str(REPO_ROOT / "vision" / "classes.json"),
                path_in_repo="classes.json", repo_id=repo_id,
                repo_type="space", token=token)
    print("    + app.py, requirements.txt, README.md, model.pth, classes.json")
    return repo_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Публикация артефактов на Hugging Face Hub")
    parser.add_argument("--what", choices=["model", "dataset", "space", "all"],
                        default="all")
    parser.add_argument("--namespace", default=None,
                        help="аккаунт HF; по умолчанию берётся из токена")
    parser.add_argument("--full-dataset", default=None,
                        help="папка dataset_full с полными картами (для датасета)")
    args = parser.parse_args()

    token = get_token()
    api = get_api(token)
    namespace = args.namespace or api.whoami()["name"]
    print("Аккаунт Hugging Face: %s" % namespace)

    full_dataset = Path(args.full_dataset) if args.full_dataset else (
        DEFAULT_FULL_DATASET if DEFAULT_FULL_DATASET.exists() else FALLBACK_FULL_DATASET
    )

    holder: dict = {}
    links = {}

    if args.what in ("model", "all"):
        print("\n[model] %s/%s" % (namespace, MODEL_NAME))
        links["model"] = publish_model(api, token, namespace, holder)

    if args.what in ("dataset", "all"):
        print("\n[dataset] %s/%s" % (namespace, DATASET_NAME))
        print("  источник полного набора: %s%s" % (
            full_dataset, "" if full_dataset.exists() else "  (папки нет — будет пропущена)"))
        links["dataset"] = publish_dataset(api, token, namespace, full_dataset)

    if args.what in ("space", "all"):
        model_repo = holder.get("model") or "%s/%s" % (namespace, MODEL_NAME)
        print("\n[space] %s/%s" % (namespace, SPACE_NAME))
        links["space"] = publish_space(api, token, namespace, model_repo)

    print("\nГотово:")
    for kind, repo_id in links.items():
        if kind == "dataset":
            print("  dataset: https://huggingface.co/datasets/%s" % repo_id)
        elif kind == "space":
            print("  space:   https://huggingface.co/spaces/%s" % repo_id)
        else:
            print("  model:   https://huggingface.co/%s" % repo_id)
    print("\nНе забудьте добавить эти ссылки в README.md (раздел «Hugging Face»).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
