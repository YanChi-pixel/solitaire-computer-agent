# Hugging Face: модель, датасет и демо

Здесь лежат карточки и скрипты публикации артефактов проекта на Hugging Face
Hub — то, что неудобно (и не нужно) держать в git: полный датасет на 50 МБ, веса
как самостоятельный артефакт и живое демо, на которое можно дать ссылку.

| Что | Куда | Из чего собирается |
| --- | --- | --- |
| **Model** | `solitaire-card-recognizer` | `model.pth` (9,4 МБ), `vision/classes.json`, `config.json`, карточка `hf/model/` |
| **Dataset** | `solitaire-cards-dataset` | `dataset_full/` (3254 полные карты) + `dataset/` (3967 уголков) + `tests/data/` (3 реальных кадра с эталоном) |
| **Space** (static) | `solitaire-card-recognizer-demo` | `hf/space_static/`: `index.html`, `app.js`, `model.onnx`, примеры карт |

## Зачем это вообще нужно

GitHub отвечает на вопрос «как это сделано»: код, тесты, история решений.
Hugging Face отвечает на другой вопрос — «чем этим можно воспользоваться прямо
сейчас»: скачать веса одной строкой, забрать датасет, потыкать демо. Для
портфолио это два разных сигнала, и они не заменяют друг друга:

* **модель с карточкой** показывает, что обучение доведено до публикуемого
  артефакта — с метриками, ограничениями и порядком классов;
* **датасет с карточкой** — что данные описаны и воспроизводимы;
* **Space** — что результат запускается у любого в браузере; именно эту ссылку
  обычно и просят.

## Почему Space статический, а не Gradio

Gradio и Docker Spaces на бесплатном железе `cpu-basic` требуют **PRO-подписки**
(API отвечает `402 Payment Required`). Статические Spaces бесплатны для всех — и
для этой задачи они даже лучше: модель выполняется **прямо в браузере** через
ONNX Runtime Web, поэтому нет сервера, нет холодного старта и нечему «засыпать».

Модель экспортируется в ONNX скриптом `hf/export_onnx.py` (softmax зашит в граф,
так что на стороне JS остаётся только собрать тензор 1×3×224×224 с нормализацией
ImageNet). Проверено, что ONNX даёт те же ответы, что PyTorch: 52/52 совпадений на
выборке по всем классам.

Gradio-вариант (`hf/space_gradio/`) оставлен как альтернатива — он рабочий и
протестирован, но залить его можно только с PRO:

```bash
python hf/upload.py --what space-gradio     # 402 без подписки
```

## Как залить

Нужен **write-токен** Hugging Face: `Settings → Access Tokens → New token → Write`.
Токен не хранится в репозитории — передаётся через переменную окружения.

```bash
pip install huggingface_hub onnx onnxruntime
export HF_TOKEN=hf_...            # Windows: set HF_TOKEN=hf_...
python hf/upload.py --what model
python hf/upload.py --what dataset --full-dataset "D:\AI Projects\sol-dev\dataset_full"
python hf/upload.py --what space
# или всё сразу:
python hf/upload.py --what all --full-dataset "D:\AI Projects\sol-dev\dataset_full"
```

Перед заливкой полезно посмотреть план без обращения к сети:

```bash
python hf/upload.py --dry-run --full-dataset "D:\AI Projects\sol-dev\dataset_full"
```

Скрипт создаёт репозитории под вашим аккаунтом (имя берётся из токена), сам
подставляет его в ссылки внутри карточек, экспортирует `model.onnx`, если его нет,
и печатает ссылки. Повторный запуск обновляет существующие репозитории, а не падает.

## Структура

```
hf/
├── README.md            этот файл
├── upload.py            публикация всего (--what model|dataset|space|all)
├── export_onnx.py       model.pth → model.onnx для демо в браузере
├── model/               карточка модели + config.json
├── dataset/             карточка датасета (три конфигурации: full, corners, fixtures)
├── space_static/        static Space: index.html, app.js, примеры карт
└── space_gradio/        альтернативный Gradio Space (нужна PRO-подписка)
```

`model.onnx` в git не хранится (`.gitignore`): он генерируется из `model.pth`
командой `python hf/export_onnx.py` или автоматически при заливке Space.
