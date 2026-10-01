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
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_DIR = REPO_ROOT / "hf"

MODEL_NAME = "solitaire-card-recognizer"
DATASET_NAME = "solitaire-cards-dataset"
SPACE_NAME = "solitaire-card-recognizer-demo"
SPACE_GRADIO_NAME = "solitaire-card-recognizer-gradio"

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


def namespace_urls(text: str, namespace: str) -> str:
    """Подставляет аккаунт HF только в ссылки Hugging Face.

    Ссылки на GitHub (исходный код) трогать нельзя: там другой аккаунт.
    """
    return re.sub(
        r"(huggingface\.co/(?:datasets/|spaces/)?)YanChi-pixel\b",
        lambda match: match.group(1) + namespace,
        text,
    )


def upload_readme(api, token: str, src: Path, repo_id: str, namespace: str,
                  repo_type: str = "model") -> None:
    """Загружает карточку, подставляя в HF-ссылки актуальный аккаунт."""
    from huggingface_hub import upload_file

    text = namespace_urls(src.read_text(encoding="utf-8"), namespace)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md",
                                     delete=False) as handle:
        handle.write(text)
        tmp_path = handle.name
    try:
        upload_file(path_or_fileobj=tmp_path, path_in_repo="README.md",
                    repo_id=repo_id, repo_type=repo_type, token=token)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def publish_model(api, token: str, namespace: str, repo_id_holder: dict) -> str:
    from huggingface_hub import create_repo, upload_file

    repo_id = "%s/%s" % (namespace, MODEL_NAME)
    create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    for local, remote in (
        (REPO_ROOT / "model.pth", "model.pth"),
        (REPO_ROOT / "vision" / "classes.json", "classes.json"),
        (HF_DIR / "model" / "config.json", "config.json"),
    ):
        if not local.exists():
            raise SystemExit("нет файла: %s" % local)
        upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                    repo_id=repo_id, token=token)
        print("    + %s" % remote)
    upload_readme(api, token, HF_DIR / "model" / "README.md", repo_id, namespace, "model")
    print("    + README.md (карточка модели)")
    repo_id_holder["model"] = repo_id
    return repo_id


def publish_dataset(api, token: str, namespace: str, full_dataset: Path) -> str:
    from huggingface_hub import create_repo, upload_folder

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

    upload_readme(api, token, HF_DIR / "dataset" / "README.md", repo_id, namespace, "dataset")
    print("    + README.md (карточка датасета)")
    return repo_id


def publish_space(api, token: str, namespace: str, model_repo: str) -> str:
    """Static Space: модель считается в браузере (ONNX Runtime Web).

    Gradio/Docker Spaces на бесплатном cpu-basic требуют PRO-подписки, а
    статические — бесплатны для всех и вдобавок не «засыпают».
    """
    from huggingface_hub import create_repo, upload_folder

    repo_id = "%s/%s" % (namespace, SPACE_NAME)
    create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True, token=token)

    stage_dir = HF_DIR / "space_static"
    onnx_path = stage_dir / "model.onnx"
    if not onnx_path.exists():
        print("    ~ model.onnx не найден, экспортирую из model.pth…")
        sys.path.insert(0, str(HF_DIR))
        import export_onnx  # noqa: PLC0415

        export_onnx.main()

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copy2(stage_dir / "index.html", stage / "index.html")
        shutil.copy2(stage_dir / "app.js", stage / "app.js")
        shutil.copy2(stage_dir / "model.onnx", stage / "model.onnx")
        shutil.copy2(REPO_ROOT / "vision" / "classes.json", stage / "classes.json")

        card = (stage_dir / "README.md").read_text(encoding="utf-8")
        (stage / "README.md").write_text(namespace_urls(card, namespace),
                                        encoding="utf-8")

        examples = stage_dir / "examples"
        if examples.is_dir():
            shutil.copytree(examples, stage / "examples")

        upload_folder(folder_path=str(stage), repo_id=repo_id,
                      repo_type="space", token=token)

    print("    + index.html, app.js, model.onnx, classes.json, examples/, README.md")
    return repo_id


def publish_space_gradio(api, token: str, namespace: str, model_repo: str) -> str:
    """Необязательный Gradio-вариант: требует PRO-подписки HF (cpu-basic)."""
    from huggingface_hub import create_repo, upload_file, upload_folder

    repo_id = "%s/%s" % (namespace, SPACE_GRADIO_NAME)
    create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, token=token)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in ("app.py", "requirements.txt", "README.md"):
            shutil.copy2(HF_DIR / "space_gradio" / name, stage / name)
        examples = HF_DIR / "space_static" / "examples"
        if examples.is_dir():
            shutil.copytree(examples, stage / "examples")

        # Space должен брать модель из опубликованного репозитория
        app = (stage / "app.py").read_text(encoding="utf-8")
        app = app.replace('MODEL_REPO = "YanChi-pixel/solitaire-card-recognizer"',
                          'MODEL_REPO = "%s"' % model_repo)
        (stage / "app.py").write_text(app, encoding="utf-8")

        # ссылки внутри карточки Space — на актуальный аккаунт HF
        card = (stage / "README.md").read_text(encoding="utf-8")
        (stage / "README.md").write_text(namespace_urls(card, namespace),
                                         encoding="utf-8")

        upload_folder(folder_path=str(stage), repo_id=repo_id,
                      repo_type="space", token=token)

    # приложить веса прямо к Space: демо не зависит от сети при старте
    upload_file(path_or_fileobj=str(REPO_ROOT / "model.pth"), path_in_repo="model.pth",
                repo_id=repo_id, repo_type="space", token=token)
    upload_file(path_or_fileobj=str(REPO_ROOT / "vision" / "classes.json"),
                path_in_repo="classes.json", repo_id=repo_id,
                repo_type="space", token=token)
    print("    + app.py, requirements.txt, README.md, model.pth, classes.json")
    return repo_id


def publish_profile(api, token: str, namespace: str) -> str:
    """README-визитка профиля: репозиторий с именем самого аккаунта."""
    from huggingface_hub import create_repo, upload_file

    repo_id = "%s/%s" % (namespace, namespace)
    create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    upload_readme(api, token, HF_DIR / "profile" / "README.md", repo_id, namespace, "model")
    print("    + README.md (визитка профиля)")
    shot = REPO_ROOT / "docs" / "img" / "space-demo.png"
    if shot.exists():
        upload_file(path_or_fileobj=str(shot), path_in_repo="assets/space-demo.png",
                    repo_id=repo_id, token=token)
        print("    + assets/space-demo.png")
    return repo_id


def check_local(full_dataset: Path) -> list[str]:
    """Проверяет, что все локальные файлы на месте (без обращения к сети)."""
    problems = []
    required = [
        REPO_ROOT / "model.pth",
        REPO_ROOT / "vision" / "classes.json",
        HF_DIR / "model" / "README.md",
        HF_DIR / "model" / "config.json",
        HF_DIR / "dataset" / "README.md",
        HF_DIR / "space_static" / "index.html",
        HF_DIR / "space_static" / "app.js",
        HF_DIR / "space_static" / "README.md",
        HF_DIR / "space_static" / "examples",
        HF_DIR / "profile" / "README.md",
        REPO_ROOT / "dataset",
        REPO_ROOT / "tests" / "data",
    ]
    for path in required:
        if not path.exists():
            problems.append("нет файла или папки: %s" % path)

    if not full_dataset.exists():
        print("! полный датасет не найден (%s) — папка 'full' будет пропущена" % full_dataset)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Публикация артефактов на Hugging Face Hub")
    parser.add_argument("--what",
                        choices=["model", "dataset", "space", "space-gradio", "profile", "all"],
                        default="all")
    parser.add_argument("--namespace", default=None,
                        help="аккаунт HF; по умолчанию берётся из токена")
    parser.add_argument("--full-dataset", default=None,
                        help="папка dataset_full с полными картами (для датасета)")
    parser.add_argument("--dry-run", action="store_true",
                        help="проверить локальные файлы и показать план, ничего не заливая")
    args = parser.parse_args()

    full_dataset = Path(args.full_dataset) if args.full_dataset else (
        DEFAULT_FULL_DATASET if DEFAULT_FULL_DATASET.exists() else FALLBACK_FULL_DATASET
    )

    if args.dry_run:
        print("Проверка перед публикацией (токен не нужен, ничего не заливается)\n")
        problems = check_local(full_dataset)
        folders = [
            ("model", REPO_ROOT / "model.pth"),
            ("model", REPO_ROOT / "vision" / "classes.json"),
            ("model", HF_DIR / "model" / "README.md"),
            ("model", HF_DIR / "model" / "config.json"),
            ("dataset", full_dataset),
            ("dataset", REPO_ROOT / "dataset"),
            ("dataset", REPO_ROOT / "tests" / "data"),
            ("dataset", HF_DIR / "dataset" / "README.md"),
            ("space", HF_DIR / "space_static" / "index.html"),
            ("space", HF_DIR / "space_static" / "app.js"),
            ("space", HF_DIR / "space_static" / "model.onnx"),
            ("space", HF_DIR / "space_static" / "examples"),
        ]
        print("Что будет загружено:")
        for kind, path in folders:
            if path.is_file():
                print("  %-8s %-52s %8.1f КБ" % (kind, path.name, path.stat().st_size / 1024))
            elif path.is_dir():
                files = [f for f in path.rglob("*") if f.is_file()]
                total = sum(f.stat().st_size for f in files)
                print("  %-8s %-52s %6d файлов, %6.1f МБ"
                      % (kind, path.name + "/", len(files), total / 1024 / 1024))
            else:
                print("  %-8s %-52s ОТСУТСТВУЕТ" % (kind, path.name))
        print()
        if problems:
            print("ПРОБЛЕМЫ:")
            for p in problems:
                print("  " + p)
            return 1
        print("Всё на месте. Запустите без --dry-run, когда будет HF_TOKEN.")
        return 0

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
        print("\n[space] %s/%s (static, инференс в браузере)" % (namespace, SPACE_NAME))
        links["space"] = publish_space(api, token, namespace, model_repo)

    if args.what == "space-gradio":
        model_repo = holder.get("model") or "%s/%s" % (namespace, MODEL_NAME)
        print("\n[space-gradio] %s/%s (нужна PRO-подписка)" % (namespace, SPACE_GRADIO_NAME))
        links["space-gradio"] = publish_space_gradio(api, token, namespace, model_repo)

    if args.what in ("profile", "all"):
        print("\n[profile] %s/%s (визитка профиля)" % (namespace, namespace))
        links["profile"] = publish_profile(api, token, namespace)

    print("\nГотово:")
    for kind, repo_id in links.items():
        if kind == "dataset":
            print("  dataset: https://huggingface.co/datasets/%s" % repo_id)
        elif kind in ("space", "space-gradio"):
            print("  space:   https://huggingface.co/spaces/%s" % repo_id)
        elif kind == "profile":
            print("  profile: https://huggingface.co/%s" % repo_id)
        else:
            print("  model:   https://huggingface.co/%s" % repo_id)
    print("\nНе забудьте добавить эти ссылки в README.md (раздел «Hugging Face»).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
